"""
報酬関数 (Reward function)

【設計方針】
旧実装は環境ラッパーと worker の 2 箇所に報酬計算が散らばり、
局収支・順位ボーナス・立直ボーナス・向聴進速・受入れが 1 つのスカラーに
潰れて記録されていた。そのため「何が学習を動かしているのか」を切り分けられず、
報酬の設計ミス (下記) にも気付けなかった。

修正点:
  1. 立直の無条件ボーナス +0.1 を撤去。
     そもそも action_id==46 (= MINKAN) に付いており立直ではなかったうえ、
     満貫ツモ ≈ +0.8 に対して +0.1 は大きすぎ、濫立直を誘発する。
  2. 受入れ (ukeire) シェーピングに phase ゲートを追加。
     向聴シェーピングにはゲートがあったのに受入れには無く、
     「純粋 RL」を謳う Phase 3 でもヒューリスティックが効き続けていた。
  3. 全ての報酬成分を個別に記録し、CSV に出せるようにした。
  4. 半荘終了時の順位ボーナスと、局ごとの素点はここで一元的に計算する。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RewardConfig:
    """全ての報酬係数。実験のたびに書き換えるのではなく、設定として保存すること。"""

    # 局終了時の素点変動 (点) をこの倍率で報酬にする。満貫直撃 ≈ 8000 点 -> 0.8
    hand_payoff_scale: float = 1.0 / 10000.0
    # 半荘終了時の順位ボーナス (ウマ)。4 位の罰を重くして避 4 の本能を作る
    rank_bonus: tuple[float, float, float, float] = (1.2, 0.3, -0.1, -1.8)
    # 向聴数が 1 下がるごとの報酬 (phase 別)
    shanten_weight: dict[int, float] = field(default_factory=lambda: {1: 0.02, 2: 0.005, 3: 0.0})
    # 受入れ枚数 1 枚あたりの報酬 (phase 別)。Phase 3 は純粋 RL なので 0
    ukeire_weight: dict[int, float] = field(default_factory=lambda: {1: 0.0005, 2: 0.0002, 3: 0.0})
    # 非合法手のペナルティ。マスクが正しければ発生しないはずの値
    illegal_penalty: float = -1.0

    def shanten_scale(self, phase: int) -> float:
        return self.shanten_weight.get(phase, 0.0)

    def ukeire_scale(self, phase: int) -> float:
        return self.ukeire_weight.get(phase, 0.0)

    def as_dict(self) -> dict:
        return {
            "hand_payoff_scale": self.hand_payoff_scale,
            "rank_bonus": list(self.rank_bonus),
            "shanten_weight": dict(self.shanten_weight),
            "ukeire_weight": dict(self.ukeire_weight),
            "illegal_penalty": self.illegal_penalty,
        }


COMPONENTS = ("payoff", "rank", "shanten", "ukeire", "illegal")


class RewardDecomposition:
    """報酬成分ごとの累積値。worker からトレーナへ持ち帰って CSV に落とす。"""

    def __init__(self):
        self.totals: Counter = Counter()

    def add(self, component: str, value: float) -> float:
        if value:
            self.totals[component] += float(value)
        return float(value)

    def snapshot(self) -> dict[str, float]:
        return {name: float(self.totals.get(name, 0.0)) for name in COMPONENTS}

    def reset(self):
        self.totals.clear()


def rank_of(scores, seat: int = 0) -> int:
    """0 起点の順位 (同点は自分が上位)"""
    my_score = scores[seat]
    return sum(1 for i, s in enumerate(scores) if s > my_score or (s == my_score and i < seat))


def hanchan_rank_bonus(config: RewardConfig, final_scores) -> tuple[float, int]:
    rank = rank_of(final_scores, seat=0)
    return config.rank_bonus[min(rank, 3)], rank


def ukeire_count(shanten_calculator, tiles34: np.ndarray, discard_tile: int) -> int:
    """
    discard_tile を打った後の受入れ枚数を数える。

    注意: 自分の手牌しか見ていないので厳密な残り枚数ではない (山と他家の手は不明)。
    シェーピング用の近似指標であり、Phase 3 では 0 になるようゲートしてある。
    """
    tiles = tiles34.copy()
    if tiles[discard_tile] <= 0:
        return 0
    tiles[discard_tile] -= 1
    is_menzen = int(tiles.sum()) in (13, 14)
    try:
        base = shanten_calculator.calculate_shanten(
            tiles.tolist(), use_chiitoitsu=is_menzen, use_kokushi=is_menzen
        )
    except ValueError:
        return 0
    total = 0
    for tile_id in range(34):
        if tiles[tile_id] >= 4:
            continue
        tiles[tile_id] += 1
        try:
            if (
                shanten_calculator.calculate_shanten(
                    tiles.tolist(), use_chiitoitsu=is_menzen, use_kokushi=is_menzen
                )
                < base
            ):
                total += 4 - (int(tiles[tile_id]) - 1)
        except ValueError:
            pass
        tiles[tile_id] -= 1
    return total
