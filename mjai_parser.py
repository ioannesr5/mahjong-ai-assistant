"""
MJAI 牌譜パーサ (MJAI replay parser)

牌譜イベント列から局面状態を厳密に再現し、教師あり学習用のサンプルを生成する。

旧 data_builder.py からの主な修正:
  1. hora / ryukyoku の `deltas` を読んで **真の点数変動** をラベル化する
     (旧: score_changes = [0,0,0,0] 固定 → 価値ヘッドが定数 0 を学習していた)
  2. ankan / kakan を処理し、meld_types を正しく記録する
     (旧: chi/pon/daiminkan のみ。暗槓牌が手牌に残り続け、以降の全特徴が汚染)
  3. **鳴かなかった / 立直しなかった** 負例を生成する
     (旧: 鳴いた局面しか無く P(鳴く|鳴ける)=1 の退化方策になっていた)
  4. 立直宣言・ロン・ツモのサンプルを生成する
     (旧: 一件も無く、対応する出力行の重みが未学習のまま死んでいた)
  5. 全サンプルに 54 次元の合法手マスクを付与する
  6. self_wind / 各家の点数 / 残り牌数 / オーラス を実際に埋める
     (旧: 全て初期値のまま = モデルは巡目も点棒状況も見えていなかった)
  7. 聴牌ラベル・危険度ラベルをオラクル手牌から実際に計算する
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field

import numpy as np
from mahjong.shanten import Shanten

import actions as A
from feature_extractor import (
    DECISION_DISCARD,
    DECISION_RESPONSE,
    DECISION_RIICHI,
    MahjongFeatureExtractor256,
    MahjongGameState,
    PlayerState,
    UnknownTileError,
    parse_tile,
)

RIICHI_STICK = 1000
WALL_TILES = 70  # 王牌を除いたツモ可能枚数


class ReplayFormatError(ValueError):
    """牌譜がこちらの想定する MJAI 形式に合致しない場合に送出する"""


def load_events(file_path: str) -> list[dict]:
    """gzip / 平文どちらの .mjson も読み込む"""
    with open(file_path, "rb") as fh:
        magic = fh.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open
    with opener(file_path, "rt", encoding="utf-8") as fh:
        content = fh.read().strip()
    if not content:
        raise ReplayFormatError("空のファイルです")
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


# ==========================================================================
# 牌の集合演算ヘルパー
# ==========================================================================
def tiles_to_counts(tiles: list[str]) -> np.ndarray:
    counts = np.zeros(34, dtype=np.int32)
    for t in tiles:
        counts[parse_tile(t)[0]] += 1
    return counts


def meld_to_counts(meld_tiles: list[str]) -> np.ndarray:
    return tiles_to_counts(meld_tiles)


def shanten_with_melds(
    calculator: Shanten, closed_tiles: list[str], melds: list[list[str]], **kwargs
) -> int | None:
    """
    副露を含めた向聴数を計算する。

    【修正】mahjong ライブラリの calculate_shanten は「副露も含めた 13/14 枚相当」の
    34 次元配列を期待する。旧コードは門前部分 (ポン後なら 10 枚) だけを渡し
    sum % 3 == 1 で通していたため、副露手の向聴数と待ち牌が全て誤っていた。
    ここでは副露牌も配列に加えて枚数を揃える。
    """
    counts = tiles_to_counts(closed_tiles)
    for meld in melds:
        m = meld_to_counts(meld)
        if m.sum() == 4:  # カンは 3 枚として数える (面子は 3 枚で 1 面子)
            for tile_id in np.nonzero(m)[0]:
                m[tile_id] = min(int(m[tile_id]), 3)
        counts += m
    total = int(counts.sum())
    if total not in (13, 14):
        return None
    if (counts > 4).any():
        return None
    return calculator.calculate_shanten(counts.tolist(), **kwargs)


def waits_of_hand(
    calculator: Shanten, closed_tiles: list[str], melds: list[list[str]]
) -> np.ndarray:
    """13 枚 (相当) の手牌の待ち牌を 34 次元の 0/1 で返す。聴牌でなければ全 0。"""
    waits = np.zeros(34, dtype=np.float32)
    menzen = not melds
    shanten = shanten_with_melds(
        calculator, closed_tiles, melds, use_chiitoitsu=menzen, use_kokushi=menzen
    )
    if shanten != 0:
        return waits
    counts = tiles_to_counts(closed_tiles)
    for meld in melds:
        m = meld_to_counts(meld)
        if m.sum() == 4:
            for tile_id in np.nonzero(m)[0]:
                m[tile_id] = min(int(m[tile_id]), 3)
        counts += m
    for tile_id in range(34):
        if counts[tile_id] >= 4:
            continue
        counts[tile_id] += 1
        if calculator.calculate_shanten(counts.tolist(), use_chiitoitsu=menzen, use_kokushi=menzen) == -1:
            waits[tile_id] = 1.0
        counts[tile_id] -= 1
    return waits


# ==========================================================================
# 鳴き可能性の判定 (負例生成に必須)
# ==========================================================================
def available_chi_actions(hand_tiles: list[str], discarded: str) -> list[int]:
    """
    上家の打牌に対してチーできるかを判定し、可能なアクション ID を返す。
    赤ドラを使う変種は、対応する 5 の赤を持っている場合に追加される。
    """
    tile_id, _ = parse_tile(discarded)
    if tile_id >= 27:  # 字牌はチーできない
        return []
    suit, num = tile_id // 9, tile_id % 9
    counts = tiles_to_counts(hand_tiles)
    red_ids = {parse_tile(t)[0] for t in hand_tiles if parse_tile(t)[1]}

    def has(n: int) -> bool:
        return 0 <= n <= 8 and counts[suit * 9 + n] > 0

    result = []
    # 打たれた牌が面子の「最小」= 自分は上の 2 枚を持つ -> CHI_LEFT
    patterns = [
        (A.CHI_LEFT, A.CHI_LEFT_RED, (num + 1, num + 2)),
        (A.CHI_MIDDLE, A.CHI_MIDDLE_RED, (num - 1, num + 1)),
        (A.CHI_RIGHT, A.CHI_RIGHT_RED, (num - 2, num - 1)),
    ]
    for base_action, red_action, (n1, n2) in patterns:
        if has(n1) and has(n2):
            result.append(base_action)
            # 使用する 2 枚のいずれかが赤5なら赤入りの変種も選べる
            used_ids = {suit * 9 + n1, suit * 9 + n2}
            if used_ids & red_ids:
                result.append(red_action)
    return result


def available_pon_kan_actions(hand_tiles: list[str], discarded: str) -> list[int]:
    tile_id, _ = parse_tile(discarded)
    counts = tiles_to_counts(hand_tiles)
    red_ids = {parse_tile(t)[0] for t in hand_tiles if parse_tile(t)[1]}
    result = []
    if counts[tile_id] >= 2:
        result.append(A.PON)
        if tile_id in red_ids:
            result.append(A.PON_RED)
    if counts[tile_id] >= 3:
        result.append(A.MINKAN)
    return result


# ==========================================================================
# サンプル
# ==========================================================================
@dataclass
class Sample:
    state_2d: np.ndarray
    cond_vec: np.ndarray
    seq_hist: np.ndarray
    action: int  # 54 次元アクション空間の正解 ID
    legal_mask: np.ndarray  # 54 次元 0/1
    decision_type: int
    actor: int
    kyoku_index: int
    step_index: int
    waits: np.ndarray  # 102 = 3家 × 34
    tenpai: np.ndarray  # 3 = 他家3人の聴牌フラグ
    danger: np.ndarray  # 102 = 3家 × 34 の放銃確率
    score_hand: float = 0.0  # この局の自分の点数変動 / 10000
    score_final: float = 0.0  # 半荘終了時の自分の順位点 (未使用なら 0)


@dataclass
class _Player:
    seat: int
    hand: list[str] = field(default_factory=list)
    melds: list[list[str]] = field(default_factory=list)
    meld_types: list[str] = field(default_factory=list)
    discards: list[str] = field(default_factory=list)
    is_tsumogiri: list[bool] = field(default_factory=list)
    is_riichi: bool = False
    riichi_turn: int = -1
    score: int = 25000


class MjaiReplayParser:
    """1 つの牌譜ファイルを解析してサンプル列を返す"""

    WIND_MAP = {"E": 0, "S": 1, "W": 2, "N": 3}

    def __init__(
        self,
        extractor: MahjongFeatureExtractor256,
        emit_call_negatives: bool = True,
        emit_riichi_negatives: bool = True,
        call_negative_rate: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        self.extractor = extractor
        self.shanten = Shanten()
        self.emit_call_negatives = emit_call_negatives
        self.emit_riichi_negatives = emit_riichi_negatives
        self.call_negative_rate = call_negative_rate
        self.rng = rng or np.random.default_rng(0)

    # ---------------- 内部状態 ----------------
    def _reset_kyoku(self, event: dict) -> None:
        self.players = [_Player(seat=i) for i in range(4)]
        self.oya = event["oya"]
        self.bakaze = self.WIND_MAP.get(event.get("bakaze", "E"), 0)
        self.kyoku = event.get("kyoku", 1)
        self.honba = event.get("honba", 0)
        self.kyotaku = event.get("kyotaku", 0)
        self.dora_indicators = [event["dora_marker"]] if event.get("dora_marker") else []
        self.discard_events: list[tuple[int, str, bool]] = []
        self.tsumo_count = 0
        self.last_drawn: dict[int, str | None] = {i: None for i in range(4)}
        scores = event.get("scores", [25000] * 4)
        for i, p in enumerate(self.players):
            p.hand = list(event["tehais"][i])
            p.score = scores[i]
        self.kyoku_samples: list[Sample] = []
        self.pending_riichi: int | None = None

    def _game_state(
        self,
        seat: int,
        decision_type: int,
        last_tile: str | None = None,
        last_actor: int | None = None,
        drawn_tile: str | None = None,
        hand_override: list[str] | None = None,
    ) -> MahjongGameState:
        players = []
        for p in self.players:
            players.append(
                PlayerState(
                    seat=p.seat,
                    score=p.score,
                    discards=list(p.discards),
                    is_tsumogiri=list(p.is_tsumogiri),
                    melds=[list(m) for m in p.melds],
                    meld_types=list(p.meld_types),
                    is_riichi=p.is_riichi,
                    riichi_turn=p.riichi_turn,
                )
            )
        return MahjongGameState(
            self_seat=seat,
            players=players,
            closed_hand=list(self.players[seat].hand if hand_override is None else hand_override),
            dora_indicators=list(self.dora_indicators),
            round_wind=self.bakaze,
            self_wind=(seat - self.oya) % 4,
            honba=self.honba,
            kyotaku=self.kyotaku,
            tiles_left=max(0, WALL_TILES - self.tsumo_count),
            is_all_last=self.is_all_last,
            discard_events=list(self.discard_events),
            decision_type=decision_type,
            last_action_tile=last_tile,
            last_action_actor=last_actor,
            drawn_tile=drawn_tile,
        )

    # ---------------- オラクルラベル ----------------
    def _oracle_labels(self, actor: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """他家 3 人の待ち牌 (102) / 聴牌フラグ (3) / 危険度 (102) を計算する"""
        waits = np.zeros(102, dtype=np.float32)
        tenpai = np.zeros(3, dtype=np.float32)
        danger = np.zeros(102, dtype=np.float32)
        for rel in range(1, 4):
            target = (actor + rel) % 4
            p = self.players[target]
            w = waits_of_hand(self.shanten, p.hand, p.melds)
            offset = (rel - 1) * 34
            waits[offset : offset + 34] = w
            if w.any():
                tenpai[rel - 1] = 1.0
                # 危険度 = 「その牌を打ったら放銃する」= 聴牌かつ待ち牌
                # 立直者は降りないので危険度をそのまま、非立直者は打点/押し引きの
                # 不確実性があるため割り引く (単純な事前確率)
                weight = 1.0 if p.is_riichi else 0.6
                danger[offset : offset + 34] = w * weight
        return waits, tenpai, danger

    # ---------------- サンプル生成 ----------------
    def _emit(
        self,
        seat: int,
        action: int,
        legal_mask: np.ndarray,
        decision_type: int,
        last_tile: str | None = None,
        last_actor: int | None = None,
        drawn_tile: str | None = None,
        hand_override: list[str] | None = None,
    ) -> None:
        if legal_mask.sum() < 2:
            return  # 選択肢が 1 つしかない局面は学習価値が無い
        gs = self._game_state(
            seat, decision_type, last_tile, last_actor, drawn_tile, hand_override
        )
        feats = self.extractor.extract(gs)
        waits, tenpai, danger = self._oracle_labels(seat)
        self.kyoku_samples.append(
            Sample(
                state_2d=feats["state_2d"],
                cond_vec=feats["cond_vec"],
                seq_hist=feats["seq_hist"],
                action=action,
                legal_mask=legal_mask,
                decision_type=decision_type,
                actor=seat,
                kyoku_index=self.kyoku_counter,
                step_index=len(self.kyoku_samples),
                waits=waits,
                tenpai=tenpai,
                danger=danger,
            )
        )

    def _discard_legal_mask(self, seat: int) -> np.ndarray:
        """手牌から打てる牌の集合。立直後はツモ切りのみ。"""
        mask = np.zeros(A.N_ACTIONS, dtype=np.float32)
        p = self.players[seat]
        if p.is_riichi:
            drawn = self.last_drawn[seat]
            if drawn is None:
                return mask
            tile_id, is_red = parse_tile(drawn)
            mask[A.RED_DISCARD_OF_TILE[tile_id] if is_red else tile_id] = 1.0
            return mask
        for tile in p.hand:
            tile_id, is_red = parse_tile(tile)
            if is_red:
                mask[A.RED_DISCARD_OF_TILE[tile_id]] = 1.0
            else:
                mask[tile_id] = 1.0
        return mask

    def _handle_dahai(self, event: dict) -> None:
        actor = event["actor"]
        tile = event["pai"]
        tsumogiri = bool(event.get("tsumogiri", False))
        p = self.players[actor]

        # (a) 立直宣言サンプル: MJAI では reach イベントが打牌の直前に来る
        if self.pending_riichi == actor:
            mask = np.zeros(A.N_ACTIONS, dtype=np.float32)
            mask[A.RIICHI] = 1.0
            mask[A.PASS_RIICHI] = 1.0
            self._emit(actor, A.RIICHI, mask, DECISION_RIICHI, drawn_tile=self.last_drawn[actor])
        elif self.emit_riichi_negatives and self._could_declare_riichi(actor):
            mask = np.zeros(A.N_ACTIONS, dtype=np.float32)
            mask[A.RIICHI] = 1.0
            mask[A.PASS_RIICHI] = 1.0
            self._emit(actor, A.PASS_RIICHI, mask, DECISION_RIICHI, drawn_tile=self.last_drawn[actor])

        # (b) 打牌サンプル
        tile_id, is_red = parse_tile(tile)
        action = A.RED_DISCARD_OF_TILE[tile_id] if is_red else tile_id
        mask = self._discard_legal_mask(actor)
        if mask[action] == 0.0:
            raise ReplayFormatError(
                f"打牌 {tile!r} が手牌に存在しません (actor={actor}, hand={p.hand})"
            )
        self._emit(
            actor, action, mask, DECISION_DISCARD, drawn_tile=self.last_drawn[actor]
        )

        # (c) 状態更新
        if tile in p.hand:
            p.hand.remove(tile)
        else:
            raise ReplayFormatError(f"打牌 {tile!r} が手牌にありません (actor={actor})")
        p.discards.append(tile)
        p.is_tsumogiri.append(tsumogiri)
        self.discard_events.append((actor, tile, tsumogiri))
        self.last_drawn[actor] = None
        if self.pending_riichi == actor:
            p.is_riichi = True
            p.riichi_turn = len(p.discards) - 1
            p.score -= RIICHI_STICK
            self.kyotaku += 1
            self.pending_riichi = None

    def _could_declare_riichi(self, actor: int) -> bool:
        """門前・未立直・1000 点以上・ツモ後 14 枚で聴牌していれば立直可能"""
        p = self.players[actor]
        if p.melds or p.is_riichi or p.score < RIICHI_STICK:
            return False
        if self.last_drawn[actor] is None:
            return False
        if WALL_TILES - self.tsumo_count < 4:  # 残り 4 巡未満は立直不可
            return False
        shanten = shanten_with_melds(
            self.shanten, p.hand, p.melds, use_chiitoitsu=True, use_kokushi=True
        )
        return shanten == 0

    def _emit_call_decisions(self, discarder: int, tile: str, called_by: int | None,
                             called_action: int | None) -> None:
        """打牌に対する他家の鳴き応答サンプル (正例 + 負例) を生成する"""
        for rel in range(1, 4):
            responder = (discarder + rel) % 4
            p = self.players[responder]
            if p.is_riichi:
                continue  # 立直後は鳴けない
            available = available_pon_kan_actions(p.hand, tile)
            if rel == 1:  # 下家のみチー可能
                available += available_chi_actions(p.hand, tile)
            if not available:
                continue

            mask = np.zeros(A.N_ACTIONS, dtype=np.float32)
            for a in available:
                mask[a] = 1.0
            mask[A.PASS_RESPONSE] = 1.0

            if responder == called_by and called_action is not None:
                if mask[called_action] == 0.0:
                    # 牌譜の鳴きをこちらの合法手判定が再現できていない (赤変種の推定ずれ等)。
                    # ラベルを信じてマスクを広げる。
                    mask[called_action] = 1.0
                self._emit(
                    responder, called_action, mask, DECISION_RESPONSE,
                    last_tile=tile, last_actor=discarder,
                )
            elif self.emit_call_negatives and self.rng.random() < self.call_negative_rate:
                self._emit(
                    responder, A.PASS_RESPONSE, mask, DECISION_RESPONSE,
                    last_tile=tile, last_actor=discarder,
                )

    def _classify_chi(self, actor: int, taken: str, consumed: list[str]) -> int:
        """consumed から左/中/右チーと赤使用の有無を判定してアクション ID にする"""
        taken_id, _ = parse_tile(taken)
        consumed_ids = sorted(parse_tile(t)[0] for t in consumed)
        uses_red = any(parse_tile(t)[1] for t in consumed)
        if taken_id < min(consumed_ids):
            base, red = A.CHI_LEFT, A.CHI_LEFT_RED
        elif taken_id > max(consumed_ids):
            base, red = A.CHI_RIGHT, A.CHI_RIGHT_RED
        else:
            base, red = A.CHI_MIDDLE, A.CHI_MIDDLE_RED
        return red if uses_red else base

    def _handle_call(self, event: dict) -> None:
        kind = event["type"]
        actor = event["actor"]
        taken = event.get("pai", "")
        consumed = list(event.get("consumed", []))
        p = self.players[actor]

        if kind == "chi":
            action = self._classify_chi(actor, taken, consumed)
            meld_type = "chi"
        elif kind == "pon":
            uses_red = any(parse_tile(t)[1] for t in consumed)
            action = A.PON_RED if uses_red else A.PON
            meld_type = "pon"
        elif kind == "daiminkan":
            action, meld_type = A.MINKAN, "minkan"
        elif kind == "ankan":
            action, meld_type = A.ANKAN, "ankan"
        elif kind == "kakan":
            action, meld_type = A.KAKAN, "kakan"
        else:
            raise ReplayFormatError(f"未知の鳴きイベント: {kind}")

        # 暗槓・加槓は自分の手番で行う「自己アクション」なのでサンプルを別途生成する
        if kind in ("ankan", "kakan"):
            mask = self._discard_legal_mask(actor)
            mask[action] = 1.0
            self._emit(actor, action, mask, DECISION_DISCARD, drawn_tile=self.last_drawn[actor])

        for tile in consumed:
            if tile in p.hand:
                p.hand.remove(tile)
            elif kind != "kakan":
                raise ReplayFormatError(f"{kind} の consumed {tile!r} が手牌にありません")

        if kind == "kakan":
            # 既存のポン面子に 1 枚追加する
            target_id = parse_tile(taken)[0]
            for idx, meld in enumerate(p.melds):
                if p.meld_types[idx] == "pon" and parse_tile(meld[0])[0] == target_id:
                    meld.append(taken)
                    p.meld_types[idx] = "kakan"
                    break
            else:
                p.melds.append([*consumed, taken])
                p.meld_types.append("kakan")
            if taken in p.hand:
                p.hand.remove(taken)
        else:
            p.melds.append([*consumed, taken] if taken else list(consumed))
            p.meld_types.append(meld_type)
        self.last_drawn[actor] = None

    # ---------------- メインループ ----------------
    def parse_file(self, file_path: str) -> list[Sample]:
        events = load_events(file_path)

        bakaze_seen = {
            self.WIND_MAP.get(e.get("bakaze", "E"), 0)
            for e in events
            if e.get("type") == "start_kyoku"
        }
        final_bakaze = max(bakaze_seen) if bakaze_seen else 0

        samples: list[Sample] = []
        self.kyoku_counter = -1
        self.is_all_last = False
        started = False

        for idx, event in enumerate(events):
            etype = event.get("type")

            if etype == "start_kyoku":
                self.kyoku_counter += 1
                self.is_all_last = (
                    self.WIND_MAP.get(event.get("bakaze", "E"), 0) == final_bakaze
                    and event.get("kyoku", 1) == 4
                )
                self._reset_kyoku(event)
                started = True

            elif not started:
                continue

            elif etype == "dora":
                self.dora_indicators.append(event["dora_marker"])

            elif etype == "tsumo":
                actor = event["actor"]
                tile = event["pai"]
                if tile == "?":
                    raise ReplayFormatError("非公開のツモ牌 '?' が含まれています")
                self.players[actor].hand.append(tile)
                self.last_drawn[actor] = tile
                self.tsumo_count += 1

            elif etype == "dahai":
                self._handle_dahai(event)
                # 直後のイベントが鳴きなら、その打牌に対する応答である
                nxt = events[idx + 1] if idx + 1 < len(events) else {}
                called_by, called_action = None, None
                if nxt.get("type") in ("chi", "pon", "daiminkan"):
                    called_by = nxt["actor"]
                    if nxt["type"] == "chi":
                        called_action = self._classify_chi(
                            called_by, nxt.get("pai", ""), list(nxt.get("consumed", []))
                        )
                    elif nxt["type"] == "pon":
                        called_action = (
                            A.PON_RED
                            if any(parse_tile(t)[1] for t in nxt.get("consumed", []))
                            else A.PON
                        )
                    else:
                        called_action = A.MINKAN
                elif nxt.get("type") == "hora" and nxt.get("target") != nxt.get("actor"):
                    # ロン: 応答サンプルは (d) で別途生成するのでここでは鳴きのみ扱う
                    pass
                self._emit_call_decisions(event["actor"], event["pai"], called_by, called_action)

            elif etype in ("chi", "pon", "daiminkan", "ankan", "kakan"):
                self._handle_call(event)

            elif etype == "reach":
                self.pending_riichi = event["actor"]

            elif etype == "reach_accepted":
                pass

            elif etype == "hora":
                actor = event["actor"]
                target = event.get("target", actor)
                is_tsumo = target == actor
                mask = np.zeros(A.N_ACTIONS, dtype=np.float32)
                if is_tsumo:
                    mask[A.TSUMO] = 1.0
                    for tile in self.players[actor].hand:
                        tile_id, is_red = parse_tile(tile)
                        mask[A.RED_DISCARD_OF_TILE[tile_id] if is_red else tile_id] = 1.0
                    self._emit(
                        actor, A.TSUMO, mask, DECISION_DISCARD,
                        drawn_tile=self.last_drawn[actor],
                    )
                else:
                    winning_tile = event.get("pai") or (
                        self.discard_events[-1][1] if self.discard_events else None
                    )
                    mask[A.RON] = 1.0
                    mask[A.PASS_RESPONSE] = 1.0
                    hand_with_win = list(self.players[actor].hand)
                    if winning_tile:
                        hand_with_win.append(winning_tile)
                    self._emit(
                        actor, A.RON, mask, DECISION_RESPONSE,
                        last_tile=winning_tile, last_actor=target,
                        hand_override=hand_with_win,
                    )
                samples.extend(self._settle(event.get("deltas", [0, 0, 0, 0])))

            elif etype == "ryukyoku":
                samples.extend(self._settle(event.get("deltas", [0, 0, 0, 0])))

        return samples

    def _settle(self, deltas: list[int]) -> list[Sample]:
        """局終了: 実際の点数変動をサンプルに書き込んで確定させる"""
        if len(deltas) != 4:
            raise ReplayFormatError(f"deltas の長さが 4 ではありません: {deltas}")
        for s in self.kyoku_samples:
            s.score_hand = float(deltas[s.actor]) / 10000.0
        for i, p in enumerate(self.players):
            p.score += int(deltas[i])
        out = self.kyoku_samples
        self.kyoku_samples = []
        return out


__all__ = [
    "MjaiReplayParser",
    "ReplayFormatError",
    "Sample",
    "UnknownTileError",
    "available_chi_actions",
    "available_pon_kan_actions",
    "load_events",
    "shanten_with_melds",
    "waits_of_hand",
]
