from dataclasses import dataclass, field

import numpy as np

# ==============================================================================
# 牌インデックス定義（Tile Index Definition）
# 0..8: マンズ(1m-9m), 9..17: ピンズ(1p-9p), 18..26: ソーズ(1s-9s), 27..33: 字牌(東南西北白發中)
# 空間テンソル(4x9)への変換を見据え、字牌の末尾(34, 35)はパディングとして扱います。
# ==============================================================================
# MJAI 牌表記 -> 牌ID (0..33)
#   数牌: "1m".."9m" / "1p".."9p" / "1s".."9s"
#   字牌: "E"(東) "S"(南) "W"(西) "N"(北) "P"(白) "F"(發) "C"(中)
#   赤ドラ: "5mr" / "5pr" / "5sr" (天鳳形式の "0m"/"0p"/"0s" も受け付ける)
#
# 【重大な修正】以前はこの表が "1z".."7z" しか持っておらず、実際の牌譜が使う
# "E"/"S"/"W"/"N"/"P"/"F"/"C" は dict.get(..., 0) の既定値に落ちて **全て 1m (ID 0)**
# として解析されていた。字牌は全打牌の約 3 割を占めるため、
# 打牌ラベル・可視枚数・現物/筋/壁・待ち牌ラベル・系列トークンの全てが汚染されていた。
TILE_STR_TO_ID = {
    **{f"{n}m": n - 1 for n in range(1, 10)},
    **{f"{n}p": 8 + n for n in range(1, 10)},
    **{f"{n}s": 17 + n for n in range(1, 10)},
    "E": 27,  # 東
    "S": 28,  # 南
    "W": 29,  # 西
    "N": 30,  # 北
    "P": 31,  # 白
    "F": 32,  # 發
    "C": 33,  # 中
    # 別表記の互換 (一部のコンバータは "1z".."7z" を出力する)
    **{f"{n}z": 26 + n for n in range(1, 8)},
}

# 赤ドラ表記 -> (牌ID, 赤フラグ)
RED_TILE_STR_TO_ID = {
    "0m": 4,
    "5mr": 4,
    "0p": 13,
    "5pr": 13,
    "0s": 22,
    "5sr": 22,
}


class UnknownTileError(ValueError):
    """牌譜に未知の牌表記が現れた場合に送出する (黙って 1m に落とさないため)"""


def parse_tile(tile_str: str, strict: bool = True) -> tuple[int, bool]:
    """
    牌文字列を (牌ID 0..33, 赤ドラフラグ) に変換する。

    strict=True の場合、未知の表記は UnknownTileError を送出する。
    以前は不明な表記を黙って 0 (=1m) にフォールバックしており、
    字牌が全て 1m として学習される原因になっていた。
    """
    if tile_str in RED_TILE_STR_TO_ID:
        return RED_TILE_STR_TO_ID[tile_str], True
    tile_id = TILE_STR_TO_ID.get(tile_str)
    if tile_id is None:
        if strict:
            raise UnknownTileError(f"未知の牌表記です: {tile_str!r}")
        return 0, False
    return tile_id, False


# ==============================================================================
# 捨て牌系列トークンの唯一の仕様 (Single spec for the discard-sequence tokens)
#   token = tile_id * 8 + rel_actor * 2 + cut_type
#     tile_id  : 0..33
#     rel_actor: (打牌者 - 自分) % 4        0..3
#     cut_type : 1 = 手切り, 0 = ツモ切り
#   最大値 = 33*8 + 3*2 + 1 = 271、PAD = 272 (Embedding の padding_idx)
#
# 【重大な修正】従来は 3 箇所で別々の符号化が行われていた:
#   - SL 打牌サンプル: rel_actor が常に 0、cut_type が「現在の一手」の値を系列全体に適用
#   - SL 鳴きサンプル: tile_id*8 のみ (別の符号化)
#   - RL 側        : cut_type が常に 1、しかも直近 72 手 (SL は最古の 72 手)
# 3 つの符号化が同じ Embedding テーブルを共有していたため、系列表現は無意味だった。
SEQ_PAD_TOKEN = 272
SEQ_VOCAB_SIZE = 273


def encode_discard_token(tile_id: int, rel_actor: int, is_tsumogiri: bool) -> int:
    return int(tile_id) * 8 + int(rel_actor) * 2 + (0 if is_tsumogiri else 1)


def encode_discard_sequence(discard_events, self_seat: int, max_len: int = 72) -> np.ndarray:
    """
    捨て牌イベント列 [(actor, tile_str, is_tsumogiri), ...] を系列トークンに変換する。
    直近 max_len 手を、古い順に左詰めで格納する。
    """
    seq = np.full(max_len, SEQ_PAD_TOKEN, dtype=np.int64)
    recent = discard_events[-max_len:]
    for idx, (actor, tile_str, is_tsumogiri) in enumerate(recent):
        tile_id, _ = parse_tile(tile_str)
        rel_actor = (actor - self_seat) % 4
        seq[idx] = encode_discard_token(tile_id, rel_actor, is_tsumogiri)
    return seq


@dataclass
class PlayerState:
    seat: int
    score: int = 25000
    discards: list[str] = field(default_factory=list)
    is_tsumogiri: list[bool] = field(default_factory=list)
    melds: list[list[str]] = field(default_factory=list)
    meld_types: list[str] = field(default_factory=list)  # 追加: "chi", "pon", "ankan", "minkan"
    is_riichi: bool = False
    riichi_turn: int = -1


# 決定タイプ (チャネル 223 / seq とは別に、どの種類の判断を求められているかを明示する)
DECISION_DISCARD = 0  # 打牌選択
DECISION_RESPONSE = 1  # 他家の打牌に対する鳴き/ロン応答
DECISION_RIICHI = 2  # 立直宣言するか否か


@dataclass
class MahjongGameState:
    self_seat: int
    players: list[PlayerState]
    closed_hand: list[str]
    dora_indicators: list[str]
    round_wind: int = 0
    self_wind: int = 0
    honba: int = 0
    kyotaku: int = 0
    tiles_left: int = 70
    is_all_last: bool = False

    # 捨て牌イベント列 (actor, 牌文字列, ツモ切りか)。seq_hist トークンの生成に使う。
    discard_events: list[tuple[int, str, bool]] = field(default_factory=list)

    # --- 決定コンテキスト (チャネル 221~224) ---
    # 従来これらの情報はテンソルに一切入っていなかったため、
    # 「どの牌を鳴くか」を問われても対象牌が分からず、鳴き判断が原理的に学習不能だった。
    decision_type: int = DECISION_DISCARD
    last_action_tile: str | None = None  # 直前に打たれた/鳴き対象の牌
    last_action_actor: int | None = None  # その牌を打ったプレイヤー (絶対座席)
    drawn_tile: str | None = None  # 自分がいまツモった牌


class MahjongFeatureExtractor256:
    """
    V2: 256チャネル完全情報テンソル抽出器（End-to-Endモデル用）
    （256通道全信息张量提取器）
    """

    def __init__(self):
        self.num_channels = 256
        self.num_tiles = 34
        self.seq_max_len = 72
        self.pad_id = 34

    def extract(self, game_state: MahjongGameState) -> dict:
        tensor_1d = np.zeros((self.num_channels, self.num_tiles), dtype=np.float32)

        # ---------------------------------------------------------
        # 1. 見えている全ての牌のカウント (Global visible counts calculation)
        # ---------------------------------------------------------
        visible_counts = np.zeros(34, dtype=int)
        for p in game_state.players:
            for tile_str in p.discards:
                t_id, _ = parse_tile(tile_str)
                visible_counts[t_id] += 1
            for meld in p.melds:
                for tile_str in meld:
                    t_id, _ = parse_tile(tile_str)
                    visible_counts[t_id] += 1
        for ind_str in game_state.dora_indicators:
            t_id, _ = parse_tile(ind_str)
            visible_counts[t_id] += 1

        # 自家手牌分を可視牌に加算
        hand_counts = np.zeros(34, dtype=int)
        has_red_5 = np.zeros(34, dtype=bool)
        for tile_str in game_state.closed_hand:
            t_id, is_red = parse_tile(tile_str)
            hand_counts[t_id] += 1
            visible_counts[t_id] += 1
            if is_red:
                has_red_5[t_id] = True

        # ドラ(宝牌)のマッピング
        is_dora = np.zeros(34, dtype=bool)
        for ind_str in game_state.dora_indicators:
            ind_id, _ = parse_tile(ind_str)
            if ind_id < 27:
                dora_id = ind_id + 1 if (ind_id % 9 != 8) else ind_id - 8
            elif ind_id < 31:
                dora_id = ind_id + 1 if ind_id < 30 else 27
            else:
                dora_id = ind_id + 1 if ind_id < 33 else 31
            is_dora[dora_id] = True

        # =========================================================
        # チャンネル 0~6: [自家手牌 (手牌) & 赤ドラ]
        # =========================================================
        for count_idx in range(4):
            tensor_1d[count_idx] = (hand_counts >= (count_idx + 1)).astype(np.float32)

        tensor_1d[4, 4] = 1.0 if has_red_5[4] else 0.0  # 赤5m
        tensor_1d[5, 13] = 1.0 if has_red_5[13] else 0.0  # 赤5p
        tensor_1d[6, 22] = 1.0 if has_red_5[22] else 0.0  # 赤5s

        # =========================================================
        # チャンネル 7~54: [四家副露 (副露)] (4人 × 12 = 48通道)
        # =========================================================
        for p_idx in range(4):
            rel_seat = (p_idx - game_state.self_seat) % 4
            base_meld = 7 + rel_seat * 12
            player = game_state.players[p_idx]

            for m_idx, meld in enumerate(player.melds):
                m_type = player.meld_types[m_idx] if m_idx < len(player.meld_types) else "pon"

                # 0~3: チー (Chi), 4~7: ポン (Pon), 8: 暗槓 (Ankan), 9: 明槓 (Minkan)
                for tile_str in meld:
                    t_id, is_red = parse_tile(tile_str)

                    if m_type == "chi":
                        tensor_1d[base_meld + 0, t_id] += 1.0
                    elif m_type == "pon":
                        tensor_1d[base_meld + 4, t_id] += 1.0
                    elif m_type == "ankan":
                        tensor_1d[base_meld + 8, t_id] += 1.0
                    elif m_type in ["minkan", "daiminkan", "kakan"]:
                        tensor_1d[base_meld + 9, t_id] += 1.0

                    # 10: 副露内のドラ (Dora in meld)
                    if is_dora[t_id]:
                        tensor_1d[base_meld + 10, t_id] += 1.0
                    # 11: 副露内の赤ドラ (Red Dora in meld)
                    if is_red:
                        tensor_1d[base_meld + 11, t_id] += 1.0

        # =========================================================
        # チャンネル 55~150: [四家舍牌与手/模切 (捨て牌)] (4人 × 24 = 96通道)
        # =========================================================
        for p_idx in range(4):
            rel_seat = (p_idx - game_state.self_seat) % 4
            base_disc = 55 + rel_seat * 24
            player = game_state.players[p_idx]

            tegiri_count = np.zeros(34, dtype=int)
            tsumo_count = np.zeros(34, dtype=int)
            post_riichi_count = np.zeros(34, dtype=int)

            for turn, tile_str in enumerate(player.discards):
                t_id, _ = parse_tile(tile_str)
                is_tsumo = player.is_tsumogiri[turn] if turn < len(player.is_tsumogiri) else False

                # 手切り(0~3) vs ツモ切り(4~7)
                if is_tsumo:
                    tensor_1d[base_disc + min(4 + tsumo_count[t_id], 7), t_id] = 1.0
                    tsumo_count[t_id] += 1
                else:
                    tensor_1d[base_disc + min(tegiri_count[t_id], 3), t_id] = 1.0
                    tegiri_count[t_id] += 1

                # 8: 立直宣言牌 (Riichi Declaration Tile)
                if player.is_riichi and turn == player.riichi_turn:
                    tensor_1d[base_disc + 8, t_id] = 1.0

                # 9~12: 立直後の切牌 (Post-Riichi Discards)
                if player.is_riichi and turn > player.riichi_turn:
                    tensor_1d[base_disc + min(9 + post_riichi_count[t_id], 12), t_id] = 1.0
                    post_riichi_count[t_id] += 1

            # 13~23 は将来的な鳴き飛ばし・リーチ後の他家ツモ切り記録などのための予約(Padding)

        # =========================================================
        # チャンネル 151~170: [宝牌指示牌 (ドラ表示牌)] (20通道)
        # =========================================================
        # 0~3: 表ドラ指示牌, 4~7: 槓ドラ指示牌
        for idx, ind_str in enumerate(game_state.dora_indicators):
            t_id, _ = parse_tile(ind_str)
            if idx < 5:  # 表
                tensor_1d[151 + min(idx, 3), t_id] = 1.0
            else:  # 槓
                tensor_1d[155 + min(idx - 5, 3), t_id] = 1.0

        # =========================================================
        # チャンネル 171~210: [防守与读牌矩阵 (防守マトリックス)] (4人 × 10 = 40通道)
        # 筋(スジ)、壁(カベ/ノーチャンス)、現物など厳密な防守物理状態
        # =========================================================
        for p_idx in range(4):
            rel_seat = (p_idx - game_state.self_seat) % 4
            base_def = 171 + rel_seat * 10
            player = game_state.players[p_idx]

            genbutsu = np.zeros(34, dtype=bool)
            for tile_str in player.discards:
                t_id, _ = parse_tile(tile_str)
                genbutsu[t_id] = True

            for t_id in range(34):
                if genbutsu[t_id]:
                    # 0: 現物 (Genbutsu) - 絶対安全
                    tensor_1d[base_def + 0, t_id] = 1.0

            # 筋牌(Suji)と壁牌(Kabe)の推論 (数牌 0~26 のみ)
            for suit in range(3):
                offset = suit * 9
                for n in range(9):
                    t_id = offset + n

                    # 1: 筋牌 (Suji) 推論
                    is_suji = False
                    if n == 0 and genbutsu[offset + 3]:
                        is_suji = True  # 1 (4現物)
                    if n == 8 and genbutsu[offset + 5]:
                        is_suji = True  # 9 (6現物)
                    if n == 1 and genbutsu[offset + 4]:
                        is_suji = True  # 2 (5現物)
                    if n == 7 and genbutsu[offset + 4]:
                        is_suji = True  # 8 (5現物)
                    if n == 2 and genbutsu[offset + 5]:
                        is_suji = True  # 3 (6現物)
                    if n == 6 and genbutsu[offset + 3]:
                        is_suji = True  # 7 (4現物)
                    if n == 3 and genbutsu[offset + 0] and genbutsu[offset + 6]:
                        is_suji = True  # 4 (1,7現物)
                    if n == 4 and genbutsu[offset + 1] and genbutsu[offset + 7]:
                        is_suji = True  # 5 (2,8現物)
                    if n == 5 and genbutsu[offset + 2] and genbutsu[offset + 8]:
                        is_suji = True  # 6 (3,9現物)

                    if is_suji:
                        tensor_1d[base_def + 1, t_id] = 1.0

                    # 2: 壁牌 (No Chance / ノーチャンス) - 4枚見えによる物理ブロック
                    is_no_chance = False
                    if n == 0 and visible_counts[offset + 1] == 4:
                        is_no_chance = True  # 1 (2が4枚見え)
                    if n == 1 and visible_counts[offset + 2] == 4:
                        is_no_chance = True  # 2 (3が4枚見え)
                    if n == 7 and visible_counts[offset + 6] == 4:
                        is_no_chance = True  # 8 (7が4枚見え)
                    if n == 8 and visible_counts[offset + 7] == 4:
                        is_no_chance = True  # 9 (8が4枚見え)
                    if n == 2 and (visible_counts[offset + 1] == 4 or visible_counts[offset + 3] == 4):
                        is_no_chance = True  # 3 (2か4が4枚見え)
                    if n == 6 and (visible_counts[offset + 5] == 4 or visible_counts[offset + 7] == 4):
                        is_no_chance = True  # 7 (6か8が4枚見え)

                    if is_no_chance:
                        tensor_1d[base_def + 2, t_id] = 1.0

            # 3: 無筋危険牌マスク (Musuji Danger Mask) - 現物でも筋でも壁でもない牌
            for t_id in range(27):
                if not genbutsu[t_id] and tensor_1d[base_def + 1, t_id] == 0.0 and tensor_1d[base_def + 2, t_id] == 0.0:
                    tensor_1d[base_def + 3, t_id] = 1.0

            # 4: 生牌（ションパイ）の字牌マスク
            for t_id in range(27, 34):
                if visible_counts[t_id] == 0:
                    tensor_1d[base_def + 4, t_id] = 1.0

        # =========================================================
        # チャンネル 211~255: [局势全局信息 (局勢コンテキスト)] (45通道)
        # 空間全体(4x9)に同一の数値をブロードキャスト填充
        # =========================================================
        tensor_1d[211, :] = game_state.round_wind / 3.0
        tensor_1d[212, :] = game_state.self_wind / 3.0
        tensor_1d[213, :] = game_state.honba / 10.0
        tensor_1d[214, :] = game_state.kyotaku / 10.0
        tensor_1d[215, :] = game_state.tiles_left / 70.0

        self_score = game_state.players[game_state.self_seat].score
        for p_idx in range(4):
            rel_seat = (p_idx - game_state.self_seat) % 4
            diff = (game_state.players[p_idx].score - self_score) / 100000.0
            tensor_1d[216 + rel_seat, :] = diff

        tensor_1d[220, :] = 1.0 if game_state.is_all_last else 0.0

        # =========================================================
        # チャンネル 221~224: [決定コンテキスト (Decision Context)]
        # 従来 221~255 は未使用の予約領域だった。ネットワークの入力形状は変えずに、
        # 「いま何を判断しているのか」「対象牌は何か」を明示する。
        # 221: 対象牌 (直前の打牌 / 鳴き対象牌) の one-hot
        # 222: その打牌者の相対座席 ((rel+1)/4 をブロードキャスト、0 は該当なし)
        # 223: 決定タイプ (0=打牌 / 1=応答 / 2=立直宣言 を /2 でブロードキャスト)
        # 224: 自分のツモ牌 one-hot
        # =========================================================
        if game_state.last_action_tile is not None:
            last_id, _ = parse_tile(game_state.last_action_tile)
            tensor_1d[221, last_id] = 1.0
        if game_state.last_action_actor is not None:
            rel_actor = (game_state.last_action_actor - game_state.self_seat) % 4
            # 0 は「該当なし」を表すため 1 起点で符号化する ((rel+1)/4 ∈ {0.25, 0.5, 0.75, 1.0})
            tensor_1d[222, :] = (rel_actor + 1) / 4.0
        tensor_1d[223, :] = game_state.decision_type / 2.0
        if game_state.drawn_tile is not None:
            drawn_id, _ = parse_tile(game_state.drawn_tile)
            tensor_1d[224, drawn_id] = 1.0

        # --- 2D空間へのトランスフォーム (Transform to 256x4x9 Spatial Tensor) ---
        padded_tensor = np.pad(tensor_1d, ((0, 0), (0, 2)), mode="constant", constant_values=0)
        state_2d = padded_tensor.reshape((self.num_channels, 4, 9))

        # ======================================================================
        # FiLM条件ベクトル (16次元) & 時系列シーケンス (72次元) は変更なし
        # ======================================================================
        cond_vec = np.zeros(16, dtype=np.float32)
        for i in range(4):
            rel_idx = (game_state.self_seat + i) % 4
            cond_vec[i] = (game_state.players[rel_idx].score - 25000) / 10000.0
        cond_vec[4 + game_state.round_wind] = 1.0
        cond_vec[8 + game_state.self_wind] = 1.0
        cond_vec[12] = min(game_state.honba / 10.0, 1.0)
        cond_vec[13] = min(game_state.kyotaku / 5.0, 1.0)
        cond_vec[14] = game_state.tiles_left / 70.0
        cond_vec[15] = 1.0 if game_state.is_all_last else 0.0

        seq_hist = encode_discard_sequence(
            game_state.discard_events, game_state.self_seat, self.seq_max_len
        )

        return {"state_2d": state_2d, "cond_vec": cond_vec, "seq_hist": seq_hist}
