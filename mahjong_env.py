"""
自己対局環境ラッパー (Self-play environment wrapper)

【設計変更の核心】
従来は `decode_obs_93_to_256()` が obs_93 から 256 チャネルテンソルを独自に組み立てており、
SL 側の `MahjongFeatureExtractor256` とは **別実装** だった。その結果、同じ局面が
2 つの異なるテンソルに符号化され、少なくとも以下の食い違いが生じていた:

  * 点差チャネル 216-219 が絶対座席 (SL は相対座席)
  * 本場 / 供託 / 残り牌 / オーラス (213/214/215/220) が常に 0
  * 副露チャネルにポン・カンをチーの枠へ書き込み
  * 「立直後の捨て牌」枠に「全捨て牌 - 立直宣言牌」を書き込み
  * カベ判定が obs の「1枚以上」フラグの総和を「見えている枚数」として == 4 と比較
    (obs_93 は全て 0/1 なので、この判定は原理的に成立しない)
  * 系列トークンが別仕様、しかも SL は最古 72 手・RL は直近 72 手

本ファイルではラッパー自身が公開情報を厳密に追跡し、**SL とまったく同じ抽出器**を呼ぶ。
これにより二重実装そのものを無くし、食い違いが構造的に発生しないようにする。

さらに半荘の連続性 (親の移動・点数の持ち越し・本場・供託・南場) を実装する。
従来は毎局 `env.reset()` を引数なしで呼んでいたため、点数が毎局 25000 に戻り、
供託 (立直棒) が消滅し、場風が常に東のままだった。
"""

from __future__ import annotations

import numpy as np
import pymahjong  # type: ignore
from mahjong.shanten import Shanten

import actions as A
from feature_extractor import (
    DECISION_DISCARD,
    DECISION_RESPONSE,
    DECISION_RIICHI,
    MahjongFeatureExtractor256,
    MahjongGameState,
    PlayerState,
)

TILE_ID_TO_STR = (
    [f"{n}m" for n in range(1, 10)]
    + [f"{n}p" for n in range(1, 10)]
    + [f"{n}s" for n in range(1, 10)]
    + ["E", "S", "W", "N", "P", "F", "C"]
)
RED_TILE_STR = {4: "5mr", 13: "5pr", 22: "5sr"}
WALL_TILES = 70
RIICHI_STICK = 1000

# obs_93 のチャネル定義 (observation_action_explanation.pdf Table 4)
OBS_HAND_COUNTS = slice(0, 4)  # 手牌が >=1,>=2,>=3,4 枚
OBS_HAND_RED = 5  # 手牌に赤ドラ
OBS_DORA_INDICATOR = slice(70, 74)  # ドラ表示牌 (>=1..4 回)
OBS_ROUND_WIND = 78
OBS_SELF_WIND = 79
OBS_LATEST_ACTION_TILE = 80  # 直前のアクションに対応する牌 (one-hot)


def tiles_from_obs(obs_93: np.ndarray) -> list[str]:
    """観測から現在の行動プレイヤーの手牌 (牌文字列のリスト) を厳密に復元する"""
    counts = obs_93[OBS_HAND_COUNTS].sum(axis=0).astype(int)
    reds = obs_93[OBS_HAND_RED].astype(int)
    hand: list[str] = []
    for tile_id in range(34):
        n = int(counts[tile_id])
        if n <= 0:
            continue
        has_red = bool(reds[tile_id]) and tile_id in RED_TILE_STR
        for k in range(n):
            if has_red and k == 0:
                hand.append(RED_TILE_STR[tile_id])
            else:
                hand.append(TILE_ID_TO_STR[tile_id])
    return hand


class _Seat:
    __slots__ = (
        "discards",
        "is_tsumogiri",
        "melds",
        "meld_types",
        "is_riichi",
        "riichi_turn",
        "hand_counts",
    )

    def __init__(self):
        self.reset()

    def reset(self):
        self.discards: list[str] = []
        self.is_tsumogiri: list[bool] = []
        self.melds: list[list[str]] = []
        self.meld_types: list[str] = []
        self.is_riichi = False
        self.riichi_turn = -1
        self.hand_counts = np.zeros(34, dtype=np.int32)


class MultiAgentMahjongEnvWrapper:
    """半荘累計制ラッパー。公開情報を自前で追跡し、SL と同一の抽出器で特徴を作る。"""

    def __init__(self, extractor: MahjongFeatureExtractor256 | None = None):
        self.env = pymahjong.MahjongEnv()
        self.extractor = extractor or MahjongFeatureExtractor256()
        self.shanten_calc = Shanten()
        self.reset_hanchan()

    # ------------------------------------------------------------------ reset
    def reset(self):
        return self.reset_hanchan()

    def reset_hanchan(self):
        self.scores = np.array([25000] * 4, dtype=np.int64)
        self.oya = 0
        self.bakaze = 0  # 0=東, 1=南
        self.kyoku = 1
        self.honba = 0
        self.kyotaku = 0
        self.kyoku_count = 0
        self.is_hanchan_done = False
        # 半荘を通じた突き合わせ統計 (追跡ロジックの健全性を監視するため)
        self.reconcile_added = 0  # 観測から補完した (自動実行で見えなかった) 捨て牌
        self.reconcile_removed = 0  # 鳴かれて捨て牌の山から消えた牌
        self._reset_hand_internal()
        return self._get_state_dict(), self._get_mask(), self.current_player

    def _reset_hand_internal(self):
        """
        1 局分の状態をリセットする。

        【修正】従来は引数なしの env.reset() だったため、点数が毎局 25000 に戻り、
        本場・供託が引き継がれず、場風が常に東だった。立直棒は毎局消滅していたので
        4 家の合計点が保存されず、「平均順位 1.94 なのに平均素点 -355」という
        矛盾した評価値の原因になっていた。
        """
        self.env.reset(
            oya=self.oya,
            game_wind=("east", "south", "west", "north")[self.bakaze],
            scores=[int(s) for s in self.scores],
            honba=self.honba,
            kyoutaku=self.kyotaku,
        )
        self.seats = [_Seat() for _ in range(4)]
        self.discard_events: list[tuple[int, str, bool]] = []
        self.dora_indicators: list[str] = []
        self.tsumo_count = 0
        self.riichi_declared_this_hand = 0
        self.last_discard: tuple[int, str] | None = None
        # 鳴かれて捨て牌の山から取られた牌 (観測との突き合わせ用)。
        # 捨て牌の履歴自体は SL 側と揃えるため保持し続ける。
        self.called_away = np.zeros((4, 34), dtype=int)
        self.pending_riichi_seat: int | None = None
        self.last_winner: int | None = None
        self.current_player = self.env.get_curr_player_id()

        # メトリクス / 報酬シェーピング用
        self.last_discarder = -1
        self.p0_min_shanten = None
        self._pending_shanten_reduction = 0
        self._drawn_tile: dict[int, str | None] = {i: None for i in range(4)}

        self._sync_public_state()

    # -------------------------------------------------------------- tracking
    def _sync_public_state(self):
        """行動プレイヤーの観測から、追跡だけでは得られない情報 (ドラ表示牌) を更新する"""
        if self.env.is_over():
            return
        obs = self.env.get_obs(self.current_player)
        indicators = obs[OBS_DORA_INDICATOR].sum(axis=0).astype(int)
        revealed: list[str] = []
        for tile_id in range(34):
            revealed.extend([TILE_ID_TO_STR[tile_id]] * int(indicators[tile_id]))
        if len(revealed) != len(self.dora_indicators):
            self.dora_indicators = revealed

        # ツモ牌の推定: 直前に記録した手牌枚数との差分
        seat = self.seats[self.current_player]
        counts = obs[OBS_HAND_COUNTS].sum(axis=0).astype(np.int32)
        delta = counts - seat.hand_counts
        drawn_ids = np.nonzero(delta > 0)[0]
        if int(counts.sum()) % 3 == 2 and len(drawn_ids) == 1 and int(delta[drawn_ids[0]]) == 1:
            self._drawn_tile[self.current_player] = TILE_ID_TO_STR[int(drawn_ids[0])]
        elif int(counts.sum()) % 3 != 2:
            self._drawn_tile[self.current_player] = None
        seat.hand_counts = counts
        self._reconcile_melds(obs, self.current_player)
        self._reconcile_discards()

    def _tracked_discard_counts(self, seat_id: int) -> np.ndarray:
        counts = np.zeros(34, dtype=int)
        for tile in self.seats[seat_id].discards:
            base = tile[:2] if tile.endswith("r") else tile
            counts[TILE_ID_TO_STR.index(base)] += 1
        return counts

    def _pond_counts(self, seat_id: int) -> np.ndarray:
        """観測が見せている「捨て牌の山」= 記録した捨て牌 - 鳴かれて取られた牌"""
        return np.maximum(self._tracked_discard_counts(seat_id) - self.called_away[seat_id], 0)

    def _reconcile_discards(self):
        """
        観測に現れているのに追跡していない捨て牌を補完する。

        【重要】pymahjong の _proceed() は「選択肢が 1 つしかない手番」を
        こちらに一切見せずに自動実行する (env_pymahjong.py:111-114)。
        立直後のプレイヤーはツモ切りしか選べないため、**立直後の捨て牌は
        step() を通らず、行動の記録だけでは永久に取りこぼす**。
        現物・筋・危険度の判断で最も重要な捨て牌が丸ごと欠落するので、
        毎ステップ観測と突き合わせて補完する。

        観測は「>=1 / >=2 / >=3 枚捨てた」の 3 枚までしか表現できないため、
        同じ牌を 4 枚捨てた場合のみ補完できない (実戦上ほぼ起こらない)。
        """
        if self.env.is_over() or self._in_riichi_stage2():
            return
        p = self.current_player
        obs = self.env.get_obs(p)
        # 補完は手番順 (直前の打牌者の次から) に走査して、順序をできるだけ実際に近づける
        start = (self.last_discarder + 1) % 4 if self.last_discarder >= 0 else p
        for offset in range(4):
            seat_id = (start + offset) % 4
            rel = (seat_id - p) % 4
            observed = obs[30 + rel * 10 : 30 + rel * 10 + 3].sum(axis=0).astype(int)
            diff = observed - np.minimum(self._pond_counts(seat_id), 3)

            # 正の差分 = 自動実行された立直後のツモ切りを取りこぼしている -> 補完する
            for tile_id in np.nonzero(diff > 0)[0]:
                for _ in range(int(diff[tile_id])):
                    tile_str = TILE_ID_TO_STR[int(tile_id)]
                    self.seats[seat_id].discards.append(tile_str)
                    self.seats[seat_id].is_tsumogiri.append(True)
                    self.discard_events.append((seat_id, tile_str, True))
                    self.last_discarder = seat_id
                    self.reconcile_added += 1

            # 負の差分 = 捨て牌の山から牌が消えた = 鳴かれた。
            # どの家の何を鳴いたかを推測せず、観測との差分で自己修復する。
            # (捨て牌の履歴自体は SL 側と揃えるため保持し続ける)
            for tile_id in np.nonzero(diff < 0)[0]:
                take = int(-diff[tile_id])
                self.called_away[seat_id][tile_id] += take
                self.reconcile_removed += take
        self.tsumo_count = sum(len(seat.discards) for seat in self.seats)
        self._resolve_call_target(obs, p)

    def _resolve_call_target(self, obs: np.ndarray, p: int) -> None:
        """
        鳴き応答を求められている場合、対象牌と打牌者を観測から確定させる。

        対象牌は obs チャネル 80 (「直前のアクションに対応する牌」) が権威。
        自前の last_discard に頼ると、自動実行された立直後のツモ切りを
        補完した順序のぶれで取り違える可能性がある。
        """
        if A.PASS_RESPONSE not in set(self.env.get_valid_actions()):
            return
        latest = obs[OBS_LATEST_ACTION_TILE]
        if not latest.any():
            return
        tile_id = int(np.argmax(latest))
        tile_str = TILE_ID_TO_STR[tile_id]
        candidates = [
            seat_id
            for seat_id in range(4)
            if seat_id != p
            and self.seats[seat_id].discards
            and self.seats[seat_id].discards[-1][:2].rstrip("r") == tile_str[:2].rstrip("r")
        ]
        if not candidates:
            candidates = [
                seat_id
                for seat_id in range(4)
                if seat_id != p and self.seats[seat_id].discards
                and any(d.startswith(tile_str[:2]) for d in self.seats[seat_id].discards)
            ]
        if not candidates:
            return
        discarder = self.last_discarder if self.last_discarder in candidates else candidates[0]
        actual = self.seats[discarder].discards[-1]
        self.last_discard = (discarder, actual)
        self.last_discarder = discarder

    def _in_riichi_stage2(self) -> bool:
        """
        立直の第 2 段階か (打牌を選んだ直後に「立直するか否か」を問われている状態)。

        この間、選んだ牌はまだ捨て牌の山に置かれていないため、観測と自前の記録が
        一時的に 1 枚ずれる。ここで突き合わせをすると「鳴かれた」と誤検出して
        以後ずっと 1 枚ずれ続けるので、この状態では突き合わせを行わない。
        """
        valid = set(self.env.get_valid_actions())
        return A.RIICHI in valid and A.PASS_RIICHI in valid

    def _decision_type(self) -> int:
        """いまモデルに求めている判断の種類 (合法手から推定する)"""
        valid = set(self.env.get_valid_actions())
        if A.RIICHI in valid or A.PASS_RIICHI in valid:
            return DECISION_RIICHI
        if A.PASS_RESPONSE in valid:
            return DECISION_RESPONSE
        return DECISION_DISCARD

    def _record_discard(self, seat_id: int, action_id: int):
        seat = self.seats[seat_id]
        if action_id >= 34:
            tile_id = {34: 4, 35: 13, 36: 22}[action_id]
            tile_str = RED_TILE_STR[tile_id]
        else:
            tile_id = action_id
            tile_str = TILE_ID_TO_STR[tile_id]
        drawn = self._drawn_tile.get(seat_id)
        tsumogiri = drawn is not None and drawn == TILE_ID_TO_STR[tile_id]
        seat.discards.append(tile_str)
        seat.is_tsumogiri.append(tsumogiri)
        self.discard_events.append((seat_id, tile_str, tsumogiri))
        self.last_discard = (seat_id, tile_str)
        self.last_discarder = seat_id
        self._drawn_tile[seat_id] = None
        if self.pending_riichi_seat == seat_id:
            seat.riichi_turn = len(seat.discards) - 1

    def _meld_counts(self, seat_id: int) -> np.ndarray:
        counts = np.zeros(34, dtype=int)
        for meld in self.seats[seat_id].melds:
            for tile in meld:
                base = tile[:2] if tile.endswith("r") else tile
                counts[TILE_ID_TO_STR.index(base)] += 1
        return counts

    def _reconcile_melds(self, obs: np.ndarray, viewer: int) -> None:
        """
        副露を観測から復元する (捨て牌と同じく突き合わせ方式)。

        行動の記録から面子を組み立てるのは、以下の理由で成立しない:
          * 同じ打牌に複数人が鳴き / ロンを主張した場合、pymahjong は優先順位で
            1 つだけ成立させるので、step() を呼べた鳴きが成立するとは限らない
          * 暗槓・加槓は対象牌がアクション ID に含まれない
          * 選択肢が 1 つの手番は _proceed() が黙って自動実行する

        観測の副露チャネル (6+rel*6 .. +5) は「副露に >=1..4 枚」「他家の捨て牌由来か」
        「赤ドラを含むか」を与えるので、増分から面子を一意に近く再構成できる。
        """
        for rel in range(4):
            seat_id = (viewer + rel) % 4
            seat = self.seats[seat_id]
            base = 6 + rel * 6
            observed = obs[base : base + 4].sum(axis=0).astype(int)
            diff = observed - self._meld_counts(seat_id)
            if not (diff > 0).any():
                continue

            from_others = obs[base + 4].astype(int)
            red_flags = obs[base + 5].astype(int)

            added: list[str] = []
            for tile_id in np.nonzero(diff > 0)[0]:
                for k in range(int(diff[tile_id])):
                    tid = int(tile_id)
                    if k == 0 and tid in RED_TILE_STR and red_flags[tid]:
                        added.append(RED_TILE_STR[tid])
                    else:
                        added.append(TILE_ID_TO_STR[tid])

            ids = [TILE_ID_TO_STR.index(t[:2] if t.endswith("r") else t) for t in added]
            unique = set(ids)

            if len(added) == 1:
                # 加槓: 既存のポン面子を 4 枚に昇格させる
                target = added[0][:2] if added[0].endswith("r") else added[0]
                for idx, meld in enumerate(seat.melds):
                    head = meld[0][:2] if meld[0].endswith("r") else meld[0]
                    if seat.meld_types[idx] in ("pon",) and head == target:
                        meld.append(added[0])
                        seat.meld_types[idx] = "kakan"
                        break
                else:
                    seat.melds.append(added)
                    seat.meld_types.append("kakan")
            elif len(added) == 4 and len(unique) == 1:
                seat.melds.append(added)
                seat.meld_types.append("minkan" if from_others[ids[0]] else "ankan")
            elif len(unique) == 1:
                seat.melds.append(added)
                seat.meld_types.append("pon")
            else:
                seat.melds.append(added)
                seat.meld_types.append("chi")

    # ----------------------------------------------------------------- state
    def _game_state(self) -> MahjongGameState:
        p = self.current_player
        obs = self.env.get_obs(p)
        players = [
            PlayerState(
                seat=i,
                score=int(self.scores[i]),
                discards=list(s.discards),
                is_tsumogiri=list(s.is_tsumogiri),
                melds=[list(m) for m in s.melds],
                meld_types=list(s.meld_types),
                is_riichi=s.is_riichi,
                riichi_turn=s.riichi_turn,
            )
            for i, s in enumerate(self.seats)
        ]
        decision = self._decision_type()
        last_tile, last_actor = (None, None)
        if decision == DECISION_RESPONSE and self.last_discard is not None:
            last_actor, last_tile = self.last_discard

        return MahjongGameState(
            self_seat=p,
            players=players,
            closed_hand=tiles_from_obs(obs),
            dora_indicators=list(self.dora_indicators),
            round_wind=self.bakaze,
            self_wind=(p - self.oya) % 4,
            honba=self.honba,
            kyotaku=self.kyotaku,
            tiles_left=max(0, WALL_TILES - self.tsumo_count),
            is_all_last=(self.bakaze == 1 and self.kyoku == 4),
            discard_events=list(self.discard_events),
            decision_type=decision,
            last_action_tile=last_tile,
            last_action_actor=last_actor,
            drawn_tile=self._drawn_tile.get(p),
        )

    def _get_state_dict(self):
        if self.env.is_over():
            return {
                "state_2d": np.zeros((256, 4, 9), dtype=np.float32),
                "cond_vec": np.zeros(16, dtype=np.float32),
                "seq_hist": np.full(72, 272, dtype=np.int64),
            }
        self._update_shanten_metric()
        return self.extractor.extract(self._game_state())

    def _update_shanten_metric(self):
        """p0 のシャンテン進速 (報酬シェーピング用メトリクス) を更新する"""
        if self.current_player != 0:
            return
        obs = self.env.get_obs(0)
        tiles34 = obs[OBS_HAND_COUNTS].sum(axis=0).astype(np.int32)
        is_menzen = not self.seats[0].melds
        try:
            current = self.shanten_calc.calculate_shanten(
                tiles34.tolist(), use_chiitoitsu=is_menzen, use_kokushi=is_menzen
            )
        except ValueError:
            return
        if self.p0_min_shanten is None:
            self.p0_min_shanten = current
        elif current < self.p0_min_shanten:
            self._pending_shanten_reduction += self.p0_min_shanten - current
            self.p0_min_shanten = current

    def _get_mask(self):
        mask = np.zeros(A.N_ACTIONS, dtype=np.float32)
        if not self.env.is_over():
            for act in self.env.get_valid_actions():
                if 0 <= act < A.N_ACTIONS:
                    mask[act] = 1.0
        return mask

    # ------------------------------------------------------------------ step
    def step(self, action_id, strict=False):
        p = self.current_player
        info = {"hand_done": False, "p0_win": False, "p0_deal_in": False, "p0_riichi": False}

        if action_id not in self.env.get_valid_actions():
            # 【修正】従来はここで done=True を返し、軌跡を偽の終端で切っていた。
            # マスクが正しければ到達しないので、デバッグ時は即座に落とす。
            if strict:
                raise AssertionError(
                    f"非合法アクション {action_id} ({A.ACTION_NAMES.get(action_id)}) が選択されました。"
                    f" 合法手={sorted(self.env.get_valid_actions())}"
                )
            return self._get_state_dict(), self._get_mask(), -1.0, False, p, info

        # --- 追跡の更新 ---
        if action_id in A.DISCARD_ACTIONS:
            self._record_discard(p, action_id)
        elif action_id == A.RIICHI:
            self.pending_riichi_seat = p
            self.seats[p].is_riichi = True
            self.seats[p].riichi_turn = max(0, len(self.seats[p].discards) - 1)
            self.riichi_declared_this_hand += 1
            if p == 0:
                info["p0_riichi"] = True
        elif action_id == A.RON:
            self.last_winner = p
            if p == 0:
                info["p0_win"] = True
            elif self.last_discarder == 0:
                info["p0_deal_in"] = True
        elif action_id == A.TSUMO:
            self.last_winner = p
            if p == 0:
                info["p0_win"] = True

        self.env.step(p, action_id)
        hand_done = self.env.is_over()
        info["hand_done"] = hand_done
        reward = 0.0

        if hand_done:
            # get_payoffs() は「局終了時の絶対点 - 25000」を返す (INIT_POINTS 基準の固定オフセット)。
            # 開始点をこちらから渡していても基準は 25000 のままなので、絶対点に戻して持ち越す。
            payoffs = self.env.get_payoffs()
            before = self.scores.copy()
            new_scores = np.array(
                [int(round(25000 + float(x))) for x in payoffs], dtype=np.int64
            )
            deltas = new_scores - before
            self.scores = new_scores
            self.kyoku_count += 1
            info["payoffs"] = [int(d) for d in deltas]
            info["hand_payoff_p0"] = float(deltas[0])
            info["scores"] = [int(x) for x in new_scores]
            self._advance_kyoku(deltas)

            if self.is_hanchan_done:
                self.current_player = p
                info["final_scores"] = [int(s) for s in self.scores]
            else:
                self._reset_hand_internal()
        else:
            self.current_player = self.env.get_curr_player_id()
            self._sync_public_state()

        return (
            self._get_state_dict(),
            self._get_mask(),
            reward,
            self.is_hanchan_done,
            self.current_player,
            info,
        )

    def _advance_kyoku(self, deltas):
        """
        親の移動・本場・供託・場風を進める。deltas はこの局の点数変動。
        """
        renchan = self.last_winner == self.oya
        if self.last_winner is None:
            # 流局: 親が聴牌なら連荘。点数変動の符号で近似する
            # (全員聴牌 / 全員不聴で変動 0 の場合も慣例どおり連荘扱いとする)
            renchan = int(deltas[self.oya]) >= 0

        if self.last_winner is None or renchan:
            self.honba += 1
        else:
            self.honba = 0

        # 供託: 誰かが和了すれば回収され、流局なら次局へ持ち越す
        if self.last_winner is not None:
            self.kyotaku = 0
        else:
            self.kyotaku += self.riichi_declared_this_hand

        if not renchan:
            self.oya = (self.oya + 1) % 4
            self.kyoku += 1
            if self.kyoku > 4:
                self.kyoku = 1
                self.bakaze += 1

        # 半荘終了判定: 南 4 局を打ち終えた / 飛び / 安全弁
        busted = bool((self.scores < 0).any())
        finished_south = self.bakaze >= 2
        self.is_hanchan_done = busted or finished_south or self.kyoku_count >= 16

    # ------------------------------------------------------- 整合性チェック
    def assert_tracking_matches_observation(self):
        """
        追跡している公開情報が pymahjong の観測と矛盾していないか検証する。
        テストとデバッグ専用 (毎ステップ呼ぶには重い)。
        """
        if self.env.is_over() or self._in_riichi_stage2():
            return
        p = self.current_player
        obs = self.env.get_obs(p)
        for rel in range(4):
            seat_id = (p + rel) % 4
            seat = self.seats[seat_id]

            # チャネル 30/31/32 = 「その牌を >=1 / >=2 / >=3 枚捨てた」。
            # チャネル 33 は資料上「4 枚を副露で持っている」と意味が違うので使わない。
            # 鳴かれた牌は捨て牌の山から消えるため、比較前に差し引く。
            observed = obs[30 + rel * 10 : 30 + rel * 10 + 3].sum(axis=0).astype(int)
            tracked = np.minimum(self._pond_counts(seat_id), 3)
            if not np.array_equal(observed, tracked):
                diff = tracked - observed
                detail = [
                    (TILE_ID_TO_STR[i], int(diff[i])) for i in np.nonzero(diff)[0]
                ]
                raise AssertionError(
                    f"捨て牌の追跡が観測と一致しません (seat={seat_id}, rel={rel}): {detail}"
                )

            # 副露の枚数もクロスチェックする (チャネル 6+rel*6 .. +3 = 副露に >=1..4 枚)
            observed_meld = int(obs[6 + rel * 6 : 6 + rel * 6 + 4].sum())
            tracked_meld = sum(len(meld) for meld in seat.melds)
            if tracked_meld != observed_meld:
                raise AssertionError(
                    f"副露の追跡が観測と一致しません (seat={seat_id}, rel={rel}): "
                    f"observed={observed_meld} tracked={tracked_meld}"
                )

        hand = tiles_from_obs(obs)
        expected = int(obs[OBS_HAND_COUNTS].sum())
        if len(hand) != expected:
            raise AssertionError(f"手牌復元の枚数が一致しません: {len(hand)} != {expected}")
