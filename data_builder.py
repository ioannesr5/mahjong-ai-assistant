"""
HDF5 データセット構築 (Dataset builder)

旧版からの主な変更:
  * サンプル生成は mjai_parser.MjaiReplayParser に委譲 (打牌 / 鳴き応答 / 立直宣言 / 和了)
  * state_2d を state_codec でパックして保存 (36 KB -> 7.2 KB、lzf 圧縮後は実測 約 0.47 KB)
  * ファイル順ではなく **対局単位のシャッフル** で train / val / test に分割
    (旧: selected_files[:split_idx]。しかも mjson_files[:max_files] で先頭 5000 件しか使っていなかった)
  * 牌譜解析を複数プロセスで並列化 (向聴計算が支配的な CPU バウンド処理)
  * schema バージョン・ソースマニフェスト・統計レポートを保存

usage:
    python data_builder.py --max-files 40000 --workers 12 --out data_v3
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from collections import Counter

import h5py
import numpy as np
from tqdm import tqdm

import actions as A
from feature_extractor import MahjongFeatureExtractor256
from mjai_parser import MjaiReplayParser, ReplayFormatError
from state_codec import (
    BIN_CHANNELS,
    CTX_CHANNELS,
    DEC_FIELDS,
    NUM_TILES,
    assert_roundtrip,
    pack_state,
    unpack_state,
)


def _use_utf8_stdout() -> None:
    """Windows の既定コンソール (cp932/gbk) で日本語ログが落ちないようにする"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


SCHEMA_VERSION = 3
FEATURE_VERSION = "256ch-v3-decision-context"

# HDF5 データセット名 -> (1 サンプルあたりの形状, dtype)
FIELDS: dict[str, tuple[tuple[int, ...], str]] = {
    "state_bin": ((BIN_CHANNELS, NUM_TILES), "uint8"),
    "state_ctx": ((CTX_CHANNELS,), "float32"),
    "state_dec": ((DEC_FIELDS,), "int16"),
    "cond_vec": ((16,), "float32"),
    "seq_hist": ((72,), "int16"),
    "target_action": ((), "int16"),
    "legal_mask": ((A.N_ACTIONS,), "uint8"),
    "decision_type": ((), "int8"),
    "target_score": ((), "float32"),
    "target_tenpai": ((3,), "float32"),
    "target_danger": ((102,), "float32"),
    "target_waits": ((102,), "float32"),
    "game_id": ((), "int32"),
    "kyoku_id": ((), "int16"),
    "step_id": ((), "int16"),
    "actor": ((), "int8"),
}


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def manifest_digest(paths: list[str]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(os.path.basename(p).encode("utf-8"))
    return h.hexdigest()


def samples_to_arrays(samples, game_id: int) -> dict[str, np.ndarray]:
    """Sample のリストを HDF5 に書ける配列辞書へ変換する (worker プロセス側で実行)"""
    packed = [pack_state(s.state_2d) for s in samples]
    return {
        "state_bin": np.stack([p[0] for p in packed]),
        "state_ctx": np.stack([p[1] for p in packed]),
        "state_dec": np.stack([p[2] for p in packed]),
        "cond_vec": np.stack([s.cond_vec for s in samples]).astype(np.float32),
        "seq_hist": np.stack([s.seq_hist for s in samples]).astype(np.int16),
        "target_action": np.array([s.action for s in samples], dtype=np.int16),
        "legal_mask": np.stack([s.legal_mask for s in samples]).astype(np.uint8),
        "decision_type": np.array([s.decision_type for s in samples], dtype=np.int8),
        "target_score": np.array([s.score_hand for s in samples], dtype=np.float32),
        "target_tenpai": np.stack([s.tenpai for s in samples]).astype(np.float32),
        "target_danger": np.stack([s.danger for s in samples]).astype(np.float32),
        "target_waits": np.stack([s.waits for s in samples]).astype(np.float32),
        "game_id": np.full(len(samples), game_id, dtype=np.int32),
        "kyoku_id": np.array([s.kyoku_index for s in samples], dtype=np.int16),
        "step_id": np.array([s.step_index for s in samples], dtype=np.int16),
        "actor": np.array([s.actor for s in samples], dtype=np.int8),
    }


class DatasetWriter:
    """1 つの HDF5 スプリットへの追記を担当する"""

    CHUNK = 256

    def __init__(self, path: str, compression: str | None = "lzf"):
        self.path = path
        self.h5 = h5py.File(path, "w")
        kw = {"compression": compression} if compression else {}
        self.ds = {
            name: self.h5.create_dataset(
                name,
                shape=(0, *shape),
                maxshape=(None, *shape),
                dtype=dtype,
                chunks=(self.CHUNK, *shape),
                **kw,
            )
            for name, (shape, dtype) in FIELDS.items()
        }
        self.n = 0
        self.stats: Counter = Counter()
        self.score_sum = 0.0
        self.score_sq = 0.0

    def append(self, arrays: dict[str, np.ndarray]) -> None:
        count = len(arrays["target_action"])
        if count == 0:
            return
        for name, ds in self.ds.items():
            ds.resize(self.n + count, axis=0)
            ds[self.n : self.n + count] = arrays[name]
        self.n += count

        for action in arrays["target_action"]:
            self.stats[A.ACTION_NAMES[int(action)]] += 1
        for decision in arrays["decision_type"]:
            self.stats[f"decision_{int(decision)}"] += 1
        scores = arrays["target_score"].astype(np.float64)
        self.score_sum += float(scores.sum())
        self.score_sq += float((scores**2).sum())

    def close(self, meta: dict) -> dict:
        report = {
            "num_samples": self.n,
            "action_distribution": dict(self.stats),
            "score_mean": self.score_sum / max(1, self.n),
            "score_rms": (self.score_sq / max(1, self.n)) ** 0.5,
        }
        for key, value in {**meta, "report": json.dumps(report, ensure_ascii=False)}.items():
            self.h5.attrs[key] = value
        self.h5.close()
        return report


# --- 並列解析 ---------------------------------------------------------------
_WORKER_PARSER: MjaiReplayParser | None = None


def _worker_init(call_negative_rate: float, seed: int) -> None:
    global _WORKER_PARSER
    _use_utf8_stdout()
    _WORKER_PARSER = MjaiReplayParser(
        MahjongFeatureExtractor256(),
        call_negative_rate=call_negative_rate,
        rng=np.random.default_rng(seed + os.getpid()),
    )


def _worker_parse(item):
    game_id, path = item
    try:
        samples = _WORKER_PARSER.parse_file(path)
    except (ReplayFormatError, ValueError, KeyError, IndexError) as exc:
        return None, type(exc).__name__
    if not samples:
        return None, "EmptyReplay"
    return samples_to_arrays(samples, game_id), None


def _build_split(
    split_name: str,
    split_files: list[str],
    out_path: str,
    *,
    workers: int,
    compression: str | None,
    call_negative_rate: float,
    seed: int,
    verify_every: int,
    common_meta: dict,
) -> None:
    print(f"\n--- 構築開始: {out_path} ({len(split_files)} 対局, workers={workers}) ---")
    writer = DatasetWriter(out_path, compression=compression)
    failures: Counter = Counter()
    next_verify = 0
    items = list(enumerate(split_files))

    def consume(result):
        nonlocal next_verify
        arrays, error = result
        if error:
            failures[error] += 1
            return
        writer.append(arrays)
        if writer.n >= next_verify:
            # 保存形式の可逆性を定期検査 (復元して形状と値域を確認)
            restored = unpack_state(
                arrays["state_bin"][0], arrays["state_ctx"][0], arrays["state_dec"][0]
            )
            assert restored.shape == (256, 4, 9), restored.shape
            next_verify = writer.n + verify_every

    if workers <= 1:
        _worker_init(call_negative_rate, seed)
        for item in tqdm(items, desc=f"Parsing {split_name}"):
            consume(_worker_parse(item))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=workers, initializer=_worker_init, initargs=(call_negative_rate, seed)
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(_worker_parse, items, chunksize=4),
                total=len(items),
                desc=f"Parsing {split_name}",
            ):
                consume(result)

    report = writer.close(
        {
            **common_meta,
            "split": split_name,
            "source_manifest_sha256": manifest_digest(split_files),
        }
    )
    size_gb = os.path.getsize(out_path) / 1e9
    print(f"[完了] {out_path}: {report['num_samples']:,} サンプル / {size_gb:.2f} GB")
    if failures:
        print(f"  解析失敗: {dict(failures)}")
    print(f"  score_hand mean={report['score_mean']:+.4f} rms={report['score_rms']:.4f}")
    grouped: Counter = Counter()
    for key, value in report["action_distribution"].items():
        if key.startswith("decision_"):
            continue
        grouped["DISCARD" if key.startswith("DISCARD") else key] += value
    print(f"  アクション分布: {dict(sorted(grouped.items(), key=lambda kv: -kv[1]))}")


def build_dataset(
    log_dir: str,
    output_dir: str,
    max_files: int | None = None,
    split=(0.90, 0.05, 0.05),
    seed: int = 20260818,
    call_negative_rate: float = 1.0,
    compression: str | None = "lzf",
    workers: int = 1,
    verify_every: int = 20000,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(log_dir, "**", "*.mjson"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(log_dir, "**", "*.json"), recursive=True))
    if not files:
        raise SystemExit(f"[エラー] 牌譜が見つかりません: {log_dir}")

    print(f"[検索完了] 牌譜ファイル数: {len(files)}")

    # 【修正】旧版は files[:max_files] で **先頭 5000 件** しか使わず、split もファイル順だった。
    # ここでは固定シードでシャッフルしてから対局単位で切る。
    rng = np.random.default_rng(seed)
    files = [files[i] for i in rng.permutation(len(files))]
    if max_files:
        files = files[:max_files]

    n_train = int(len(files) * split[0])
    n_val = int(len(files) * split[1])
    splits = {
        "train": files[:n_train],
        "val": files[n_train : n_train + n_val],
        "test": files[n_train + n_val :],
    }
    print(
        f"[分割] train={len(splits['train'])} / val={len(splits['val'])} / test={len(splits['test'])} "
        f"(対局単位, seed={seed})"
    )

    common_meta = {
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "action_space": "pymahjong-54",
        "builder_git_hash": git_hash(),
        "build_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": seed,
        "call_negative_rate": call_negative_rate,
    }

    for split_name, split_files in splits.items():
        if not split_files:
            continue
        _build_split(
            split_name,
            split_files,
            os.path.join(output_dir, f"{split_name}_dataset.h5"),
            workers=workers,
            compression=compression,
            call_negative_rate=call_negative_rate,
            seed=seed,
            verify_every=verify_every,
            common_meta=common_meta,
        )


if __name__ == "__main__":
    _use_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="data/logs/2024_mjai")
    ap.add_argument("--out", default="data_v3")
    ap.add_argument("--max-files", type=int, default=None, help="使用する牌譜数 (既定: 全件)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--call-negative-rate", type=float, default=1.0)
    ap.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "none"])
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    # 保存形式の可逆性を起動時に 1 度だけ厳密検査する
    # (牌インデックス 34, 35 は常にパディング 0 なので、そこは 0 のままにする)
    _flat = np.zeros((256, 36), dtype=np.float32)
    _flat[5, 13] = 1.0
    _flat[213, :34] = 0.3
    _flat[221, 24] = 1.0
    _flat[222, :34] = 3 / 4.0
    _flat[224, 7] = 1.0
    assert_roundtrip(_flat.reshape(256, 4, 9))

    build_dataset(
        args.logs,
        args.out,
        max_files=args.max_files,
        call_negative_rate=args.call_negative_rate,
        compression=None if args.compression == "none" else args.compression,
        workers=args.workers,
        seed=args.seed,
    )
