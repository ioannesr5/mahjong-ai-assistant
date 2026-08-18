"""
状態テンソルのディスク表現コーデック (State tensor storage codec)

ネットワークの入力形状 (256, 4, 9) は一切変更しない。変えるのは **保存形式** だけである。

旧形式の問題:
  state_2d を float32 の (N, 256, 4, 9) でそのまま保存していたため 36 KB/サンプル。
  牌譜 5000 局分 (237 万サンプル) で 91 GB、手元の 184425 局を使い切るには約 3.3 TB 必要だった。
  しかも chunks=(128, 256, 4, 9) ≈ 4.7 MB 無圧縮なので、shuffle 読み出しでは
  1 サンプル読むたびに 4.7 MB のチャンクを丸ごと展開していた。

実測に基づく圧縮:
  * 値の 99.99% は 0/1、非二値なのは副露の枚数チャネル (最大 4) だけ  -> uint8 で十分
  * チャネル 211~220 は 4x9 全面に同じ値をブロードキャストしたスカラー -> 10 個の float で十分
  * チャネル 221~224 は one-hot と小さなスカラー                        -> 4 個の整数で十分
  * チャネル 225~255 は未使用 (常に 0)                                   -> 保存しない
  * 牌インデックス 34, 35 はパディング (常に 0)                          -> 保存しない

  36864 B -> 7174 + 40 + 8 = 7222 B (約 5.1 倍)、lzf 圧縮後はさらに 3~5 倍。
"""

from __future__ import annotations

import numpy as np

NUM_CHANNELS = 256
NUM_TILES = 34
BIN_CHANNELS = 211  # 0..210: 手牌 / 副露 / 捨て牌 / ドラ表示 / 守備行列
CTX_CHANNELS = 10  # 211..220: 場風 自風 本場 供託 残り牌 点差x4 オーラス
CTX_START = 211
DEC_START = 221  # 221..224: 決定コンテキスト
DEC_FIELDS = 4

STATE_SHAPE = (NUM_CHANNELS, 4, 9)


def pack_state(state_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(256, 4, 9) float32 -> (bin uint8, ctx float32, dec int16)"""
    flat = np.asarray(state_2d, dtype=np.float32).reshape(NUM_CHANNELS, 36)[:, :NUM_TILES]

    state_bin = np.rint(flat[:BIN_CHANNELS]).astype(np.uint8)
    ctx = flat[CTX_START : CTX_START + CTX_CHANNELS, 0].astype(np.float32)

    dec = np.full(DEC_FIELDS, -1, dtype=np.int16)
    last_tile = flat[DEC_START]
    if last_tile.any():
        dec[0] = int(np.argmax(last_tile))
    dec[1] = int(round(float(flat[DEC_START + 1, 0]) * 4.0)) - 1  # 0 -> -1 (該当なし)
    dec[2] = int(round(float(flat[DEC_START + 2, 0]) * 2.0))
    drawn = flat[DEC_START + 3]
    if drawn.any():
        dec[3] = int(np.argmax(drawn))
    return state_bin, ctx, dec


def unpack_state(state_bin: np.ndarray, ctx: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """(bin, ctx, dec) -> (256, 4, 9) float32"""
    flat = np.zeros((NUM_CHANNELS, 36), dtype=np.float32)
    flat[:BIN_CHANNELS, :NUM_TILES] = state_bin.astype(np.float32)
    flat[CTX_START : CTX_START + CTX_CHANNELS, :NUM_TILES] = ctx.astype(np.float32)[:, None]
    if dec[0] >= 0:
        flat[DEC_START, int(dec[0])] = 1.0
    if dec[1] >= 0:
        flat[DEC_START + 1, :NUM_TILES] = (float(dec[1]) + 1.0) / 4.0
    flat[DEC_START + 2, :NUM_TILES] = float(dec[2]) / 2.0
    if dec[3] >= 0:
        flat[DEC_START + 3, int(dec[3])] = 1.0
    return flat.reshape(STATE_SHAPE)


def unpack_state_batch(state_bin: np.ndarray, ctx: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """バッチ版。(B, 211, 34), (B, 10), (B, 4) -> (B, 256, 4, 9)"""
    batch = state_bin.shape[0]
    flat = np.zeros((batch, NUM_CHANNELS, 36), dtype=np.float32)
    flat[:, :BIN_CHANNELS, :NUM_TILES] = state_bin.astype(np.float32)
    flat[:, CTX_START : CTX_START + CTX_CHANNELS, :NUM_TILES] = ctx.astype(np.float32)[:, :, None]
    rows = np.arange(batch)
    has_last = dec[:, 0] >= 0
    flat[rows[has_last], DEC_START, dec[has_last, 0].astype(np.int64)] = 1.0
    has_actor = dec[:, 1] >= 0
    flat[rows[has_actor], DEC_START + 1, :NUM_TILES] = (
        dec[has_actor, 1].astype(np.float32)[:, None] + 1.0
    ) / 4.0
    flat[:, DEC_START + 2, :NUM_TILES] = dec[:, 2].astype(np.float32)[:, None] / 2.0
    has_drawn = dec[:, 3] >= 0
    flat[rows[has_drawn], DEC_START + 3, dec[has_drawn, 3].astype(np.int64)] = 1.0
    return flat.reshape(batch, *STATE_SHAPE)


def assert_roundtrip(state_2d: np.ndarray, atol: float = 1e-5) -> None:
    """パック→アンパックで元のテンソルに戻ることを検証する (build 時のサンプリング検査用)"""
    restored = unpack_state(*pack_state(state_2d))
    if not np.allclose(restored, state_2d, atol=atol):
        diff = np.argwhere(np.abs(restored - np.asarray(state_2d, dtype=np.float32)) > atol)
        raise AssertionError(
            f"状態のパック/アンパックが不可逆です。最初の不一致: {diff[:5].tolist()}"
        )
