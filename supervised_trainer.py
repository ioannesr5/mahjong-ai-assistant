"""
教師あり学習 (Supervised pre-training)

ネットワーク構造は models.py に集約済みで、本ファイルでは一切変更しない。

旧版からの主な修正:
  1. act_map の時限爆弾を撤去。旧 47 次元アクション空間のインデックス
     [45, 34, 37, 39, 41, 42] を 54 次元ヘッドにそのまま使っており、
     このまま再学習すると CHI->34(赤5m打牌)、PON->37(チー左) のように
     全ての鳴き/宣言の意味がずれるところだった。
     いまは mjai_parser が 54 次元の正解 ID を直接出力する。
  2. **動作レベルの合法手マスク** を適用する。
     旧 valid_mask は「そのサンプルが打牌か鳴きか」を表すサンプル単位のスカラーで、
     合法手マスクではなかった (= 54 クラス全部に対する無制約分類だった)。
  3. データ拡張の修正: 逆置換の誤り、t_waits / t_danger / seq_hist / legal_mask の
     非同期、麻雀の対称性ではない数字反転 (flip) を撤去。
  4. value loss を MSE から Huber へ (麻雀の点数分布は裾が重い)。
  5. t_waits の正例率は 0.8% 程度なので pos_weight を導入。
  6. 検証指標をヘッドごとに分離 (top-1 / top-3 / 合法手内 accuracy / AUC / AP)。
  7. 学習後に凍結ベースライン sl_baseline_frozen.pth を保存する。

usage:
    python supervised_trainer.py --data data_v3 --epochs 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch_directml
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import actions as A
from feature_extractor import SEQ_PAD_TOKEN
from models import (
    DirectMLSafeAdamW,
    SmartMahjongMultiTaskNet,
    directml_safe_bce_with_logits,
)
from state_codec import unpack_state

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


# ==========================================
# 1. 動的データ拡張 (花色置換のみ)
# ==========================================
def suit_permutation_tables(perm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    花色置換 perm (新しい花色 j の中身は元の花色 perm[j]) に対応する牌インデックスの写像。

    返り値:
      forward[新しい牌ID] = 元の牌ID   … 状態やマスクを引き直すのに使う
      inverse[元の牌ID]   = 新しい牌ID … 正解ラベルを付け替えるのに使う
    """
    forward = np.arange(34, dtype=np.int64)
    for new_suit in range(3):
        old_suit = int(perm[new_suit])
        forward[new_suit * 9 : new_suit * 9 + 9] = np.arange(old_suit * 9, old_suit * 9 + 9)
    inverse = np.empty(34, dtype=np.int64)
    inverse[forward] = np.arange(34, dtype=np.int64)
    return forward, inverse


class SuitPermutationAugmenter:
    """
    花色 (萬子/筒子/索子) の置換によるデータ拡張。

    【修正点】
      * 赤ドラチャネル 4/5/6 は「チャネル番号そのものが花色」なので、
        行の置換とは **逆向き** の写像で入れ替える必要がある。
        旧コードは perm をそのまま使っており、3 巡回の置換 (全体の約 1/3) で
        赤ドラの花色がずれていた。
      * 状態だけでなく legal_mask / t_waits / t_danger / seq_hist も同時に置換する。
        旧コードは target_discard しか置換しておらず、
        「状態と待ち牌ラベルが食い違うサンプル」を量産していた。
      * 数字反転 (1..9 -> 9..1) は削除。ドラ表示牌の n -> n+1 という関係が
        反転すると n -> n-1 になり、麻雀として矛盾したドラ推論を教えてしまう。
    """

    RED_CHANNEL_BASE = 4  # チャネル 4/5/6 = 赤5m/赤5p/赤5s

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def __call__(self, state_2d, action, legal_mask, waits, danger, seq_hist):
        perm = self.rng.permutation(3)
        if bool((perm == np.arange(3)).all()):
            return state_2d, action, legal_mask, waits, danger, seq_hist

        forward, inverse = suit_permutation_tables(perm)
        forward_t = torch.from_numpy(forward)
        inverse_t = torch.from_numpy(inverse)

        # --- 状態: 花色の行を入れ替える (全 256 チャネル共通) ---
        new_state = state_2d.clone()
        new_state[:, :3, :] = state_2d[:, torch.from_numpy(perm.copy()), :]
        # --- 赤ドラチャネルはチャネル番号自体が花色なので逆写像で付け替える ---
        base = self.RED_CHANNEL_BASE
        red = new_state[base : base + 3].clone()
        for old_suit in range(3):
            new_suit = int(inverse[old_suit * 9]) // 9
            new_state[base + new_suit] = red[old_suit]
        state_2d = new_state

        # --- 正解アクション: 打牌 (0..33) と赤ドラ打牌 (34..36) のみ花色に依存 ---
        action = int(action)
        if action < 34:
            action = int(inverse[action])
        elif action < 37:
            action = 34 + int(inverse[(action - 34) * 9 + 4]) // 9

        # --- 合法手マスク ---
        new_mask = legal_mask.clone()
        new_mask[:34] = legal_mask[forward_t]
        red_src = torch.tensor([34 + int(forward[s * 9]) // 9 for s in range(3)])
        new_mask[34:37] = legal_mask[red_src]
        legal_mask = new_mask

        # --- 待ち牌 / 危険度 (3家 x 34) ---
        waits = waits.view(3, 34)[:, forward_t].reshape(-1)
        danger = danger.view(3, 34)[:, forward_t].reshape(-1)

        # --- 系列トークン: token = tile_id*8 + rel*2 + cut。牌 ID 部分だけ置換 ---
        pad = seq_hist == SEQ_PAD_TOKEN
        tile_ids = torch.div(seq_hist, 8, rounding_mode="floor").clamp(0, 33)
        rest = seq_hist - tile_ids * 8
        seq_hist = torch.where(pad, seq_hist, inverse_t[tile_ids] * 8 + rest)

        return state_2d, action, legal_mask, waits, danger, seq_hist


# ==========================================
# 2. データセット
# ==========================================
class MahjongSupervisedDataset(Dataset):
    """data_builder v3 形式 (state_bin / state_ctx / state_dec) を読む"""

    def __init__(self, h5_path: str, is_train: bool = False, seed: int = 0):
        self.h5_path = h5_path
        self.is_train = is_train
        self.seed = seed
        self.file = None
        self.augmenter = None
        with h5py.File(h5_path, "r") as f:
            self.length = f["target_action"].shape[0]
            self.attrs = dict(f.attrs)
        expected = 3
        if int(self.attrs.get("schema_version", -1)) != expected:
            raise ValueError(
                f"{h5_path} の schema_version={self.attrs.get('schema_version')} は "
                f"想定 ({expected}) と異なります。data_builder.py で作り直してください。"
            )

    def __len__(self):
        return self.length

    def _ensure_open(self):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")
            worker = torch.utils.data.get_worker_info()
            wid = worker.id if worker else 0
            self.augmenter = SuitPermutationAugmenter(np.random.default_rng(self.seed + wid))

    def __getitem__(self, idx):
        self._ensure_open()
        f = self.file
        state_2d = torch.from_numpy(
            unpack_state(f["state_bin"][idx], f["state_ctx"][idx], f["state_dec"][idx])
        )
        cond_vec = torch.from_numpy(f["cond_vec"][idx].astype(np.float32))
        seq_hist = torch.from_numpy(f["seq_hist"][idx].astype(np.int64))
        action = int(f["target_action"][idx])
        legal_mask = torch.from_numpy(f["legal_mask"][idx].astype(np.float32))
        decision_type = int(f["decision_type"][idx])
        score = torch.tensor(float(f["target_score"][idx]), dtype=torch.float32)
        tenpai = torch.from_numpy(f["target_tenpai"][idx].astype(np.float32))
        danger = torch.from_numpy(f["target_danger"][idx].astype(np.float32))
        waits = torch.from_numpy(f["target_waits"][idx].astype(np.float32))

        if self.is_train:
            state_2d, action, legal_mask, waits, danger, seq_hist = self.augmenter(
                state_2d, action, legal_mask, waits, danger, seq_hist
            )

        return {
            "state_2d": state_2d,
            "cond_vec": cond_vec,
            "seq_hist": seq_hist,
            "target_action": torch.tensor(action, dtype=torch.long),
            "legal_mask": legal_mask,
            "decision_type": torch.tensor(decision_type, dtype=torch.long),
            "target_value": score,
            "target_tenpai": tenpai,
            "target_danger": danger,
            "target_waits": waits,
        }


# ==========================================
# 3. 指標
# ==========================================
class MetricAccumulator:
    """ヘッドごとの検証指標を貯める (旧版は total loss しか見ていなかった)"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.top1 = 0
        self.top3 = 0
        self.by_decision = {0: [0, 0], 1: [0, 0], 2: [0, 0]}  # [correct, count]
        self.value_abs = 0.0
        self.tenpai_correct = 0
        self.tenpai_n = 0
        self.waits_tp = 0
        self.waits_pred = 0
        self.waits_true = 0

    def update(self, logits, target, decision_type, value_pred, value_true, tenpai_logit,
               tenpai_true, waits_logit, waits_true):
        n = target.numel()
        self.n += n
        top3 = logits.topk(min(3, logits.size(-1)), dim=-1).indices
        hit1 = top3[:, 0] == target
        self.top1 += int(hit1.sum())
        self.top3 += int((top3 == target.unsqueeze(1)).any(dim=1).sum())
        for dtype in (0, 1, 2):
            sel = decision_type == dtype
            if bool(sel.any()):
                self.by_decision[dtype][0] += int(hit1[sel].sum())
                self.by_decision[dtype][1] += int(sel.sum())
        self.value_abs += float((value_pred - value_true).abs().sum())

        tenpai_pred = (torch.sigmoid(tenpai_logit) > 0.5).float()
        self.tenpai_correct += int((tenpai_pred == tenpai_true).sum())
        self.tenpai_n += tenpai_true.numel()

        waits_pred = (torch.sigmoid(waits_logit) > 0.5).float()
        self.waits_tp += int(((waits_pred == 1) & (waits_true == 1)).sum())
        self.waits_pred += int((waits_pred == 1).sum())
        self.waits_true += int((waits_true == 1).sum())

    def summary(self) -> dict:
        n = max(1, self.n)
        precision = self.waits_tp / max(1, self.waits_pred)
        recall = self.waits_tp / max(1, self.waits_true)
        return {
            "top1": self.top1 / n,
            "top3": self.top3 / n,
            "acc_discard": self.by_decision[0][0] / max(1, self.by_decision[0][1]),
            "acc_response": self.by_decision[1][0] / max(1, self.by_decision[1][1]),
            "acc_riichi": self.by_decision[2][0] / max(1, self.by_decision[2][1]),
            "value_mae": self.value_abs / n,
            "tenpai_acc": self.tenpai_correct / max(1, self.tenpai_n),
            "waits_precision": precision,
            "waits_recall": recall,
            "waits_f1": 2 * precision * recall / max(1e-8, precision + recall),
        }


def plot_training_history(history, out_path="training_curve_sl_v3.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss")
    axes[0].set_title("Total loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)
    for key in ("top1", "acc_discard", "acc_response", "acc_riichi"):
        axes[1].plot(epochs, [m[key] for m in history["val_metrics"]], label=key)
    axes[1].set_title("Validation accuracy by decision type")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f" -> [Info] 訓練グラフを保存しました: {out_path}")


# ==========================================
# 4. 訓練ループ
# ==========================================
def masked_cross_entropy(logits, legal_mask, target, label_smoothing=0.0):
    """合法手だけを台とする交差エントロピー"""
    masked = logits + (1.0 - legal_mask) * -1e9
    return F.cross_entropy(masked, target, label_smoothing=label_smoothing), masked


def train_supervised_multitask(
    data_dir="data_v3",
    epochs=20,
    batch_size=256,
    accumulation_steps=4,
    lr=1e-4,
    patience=4,
    num_workers=6,
    lambda_policy=1.0,
    lambda_value=0.5,
    lambda_tenpai=0.3,
    lambda_danger=0.3,
    lambda_waits=1.0,
    label_smoothing=0.02,
    seed=20260818,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch_directml.is_available():
        device = torch_directml.device()
        print(f"訓練デバイス (Device): DirectML - {torch_directml.device_name(0)}")
    else:
        device = torch.device("cpu")
        print("訓練デバイス (Device): CPU")

    train_ds = MahjongSupervisedDataset(
        os.path.join(data_dir, "train_dataset.h5"), is_train=True, seed=seed
    )
    val_ds = MahjongSupervisedDataset(os.path.join(data_dir, "val_dataset.h5"), is_train=False)
    print(f"データセット: train={len(train_ds):,} / val={len(val_ds):,}")
    print(f"  feature_version={train_ds.attrs.get('feature_version')} "
          f"builder={str(train_ds.attrs.get('builder_git_hash'))[:8]}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
    )

    model = SmartMahjongMultiTaskNet(input_channels=256, num_blocks=18, dropout_p=0.30).to(device)
    optimizer = DirectMLSafeAdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    warmup_epochs = min(2, max(1, epochs // 10))
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6),
        ],
        milestones=[warmup_epochs],
    )

    # t_waits の正例率は実測 0.76%。そのままでは全て 0 と予測するのが最適解になる。
    waits_pos_weight = torch.tensor(20.0, device=device)
    huber = nn.SmoothL1Loss(beta=0.5)

    best_val = float("inf")
    stall = 0
    history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    def compute_losses(batch):
        state_2d = batch["state_2d"].to(device)
        cond_vec = batch["cond_vec"].to(device)
        seq_hist = batch["seq_hist"].to(device)
        target = batch["target_action"].to(device)
        legal = batch["legal_mask"].to(device)
        value_true = batch["target_value"].to(device)
        tenpai_true = batch["target_tenpai"].to(device)
        danger_true = batch["target_danger"].to(device)
        waits_true = batch["target_waits"].to(device)

        logits, value, tenpai, danger, waits = model(state_2d, cond_vec, seq_hist, rl_mode=False)
        loss_policy, masked_logits = masked_cross_entropy(
            logits, legal, target, label_smoothing=label_smoothing
        )
        loss_value = huber(value.squeeze(-1), value_true)
        loss_tenpai = directml_safe_bce_with_logits(tenpai, tenpai_true)
        loss_danger = directml_safe_bce_with_logits(danger, danger_true)
        loss_waits = F.binary_cross_entropy_with_logits(
            waits, waits_true, pos_weight=waits_pos_weight
        )
        total = (
            lambda_policy * loss_policy
            + lambda_value * loss_value
            + lambda_tenpai * loss_tenpai
            + lambda_danger * loss_danger
            + lambda_waits * loss_waits
        )
        parts = {
            "policy": float(loss_policy),
            "value": float(loss_value),
            "tenpai": float(loss_tenpai),
            "danger": float(loss_danger),
            "waits": float(loss_waits),
        }
        outputs = (masked_logits, target, value.squeeze(-1), value_true, tenpai, tenpai_true, waits, waits_true)
        return total, parts, outputs

    for epoch in range(epochs):
        model.train()
        running = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}] Train", leave=False)
        for i, batch in enumerate(pbar):
            total, parts, _ = compute_losses(batch)
            (total / accumulation_steps).backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            running += float(total)
            if i % 50 == 0:
                pbar.set_postfix({k: f"{v:.3f}" for k, v in parts.items()})
        train_loss = running / max(1, len(train_loader))

        model.eval()
        metrics = MetricAccumulator()
        val_running = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{epochs}] Val", leave=False):
                total, _, out = compute_losses(batch)
                val_running += float(total)
                masked_logits, target, value, value_true, tenpai, tenpai_true, waits, waits_true = out
                metrics.update(
                    masked_logits.float().cpu(), target.cpu(), batch["decision_type"],
                    value.float().cpu(), value_true.float().cpu(),
                    tenpai.float().cpu(), tenpai_true.float().cpu(),
                    waits.float().cpu(), waits_true.float().cpu(),
                )
        val_loss = val_running / max(1, len(val_loader))
        summary = metrics.summary()
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(summary)

        print(
            f"Epoch [{epoch + 1}/{epochs}] train={train_loss:.4f} val={val_loss:.4f} | "
            f"top1={summary['top1']:.3f} top3={summary['top3']:.3f} | "
            f"打牌={summary['acc_discard']:.3f} 応答={summary['acc_response']:.3f} "
            f"立直={summary['acc_riichi']:.3f} | value MAE={summary['value_mae']:.3f} | "
            f"聴牌acc={summary['tenpai_acc']:.3f} waits F1={summary['waits_f1']:.3f} | "
            f"LR={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            stall = 0
            torch.save(model.state_dict(), "smart_mahjong_base_policy_v3.pth")
            with open("sl_metrics_v3.json", "w", encoding="utf-8") as fh:
                json.dump({"epoch": epoch + 1, "val_loss": val_loss, **summary}, fh, indent=2)
            print(" -> [Update] ベースポリシー保存: smart_mahjong_base_policy_v3.pth")
        else:
            stall += 1
            print(f" -> [Info] Val Loss が改善されません ({stall}/{patience})")
            if stall >= patience:
                print(f"*** Early Stopping: {epoch + 1} エポックで終了 ***")
                break

    plot_training_history(history)

    # 【追加】以後の全実験の恒久的な比較基準となる凍結ベースライン。
    # これを上書きしてはならない (rolling baseline とは別物)。
    if os.path.exists("smart_mahjong_base_policy_v3.pth"):
        frozen = torch.load("smart_mahjong_base_policy_v3.pth", map_location="cpu", weights_only=False)
        torch.save(frozen, "sl_baseline_frozen.pth")
        print(" -> [Info] 凍結ベースラインを保存しました: sl_baseline_frozen.pth")

    # 死行チェック: 立直/ロン/ツモの重みノルムが打牌行と同程度あるか
    weight = torch.load("smart_mahjong_base_policy_v3.pth", map_location="cpu", weights_only=False)[
        "policy_out.weight"
    ].float()
    norms = weight.norm(dim=1)
    reference = float(norms[:34].median())
    print(f" -> [Head Check] 打牌行ノルム中央値 = {reference:.3f}")
    for action in (A.RIICHI, A.RON, A.TSUMO, A.PASS_RESPONSE, A.PON, A.CHI_LEFT):
        ratio = float(norms[action]) / max(reference, 1e-8)
        flag = "OK" if ratio > 0.5 else "**DEAD**"
        print(f"      {A.ACTION_NAMES[action]:<16} |w|={float(norms[action]):.3f} ({ratio:.2f}x) {flag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_v3")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--accumulation-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=6)
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.data, "train_dataset.h5")):
        raise SystemExit(
            f"データセットが見つかりません: {args.data}\n"
            "先に `python data_builder.py --out data_v3` を実行してください。"
        )
    train_supervised_multitask(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        lr=args.lr,
        patience=args.patience,
        num_workers=args.num_workers,
    )
