from dataclasses import dataclass, field

import numpy as np

# ==============================================================================
# 牌インデックス定義（Tile Index Definition）
# 0..8: マンズ(1m-9m), 9..17: ピンズ(1p-9p), 18..26: ソーズ(1s-9s), 27..33: 字牌(東南西北白發中)
# ==============================================================================

TILE_STR_TO_ID = {
    "1m": 0,
    "2m": 1,
    "3m": 2,
    "4m": 3,
    "5m": 4,
    "6m": 5,
    "7m": 6,
    "8m": 7,
    "9m": 8,
    "1p": 9,
    "2p": 10,
    "3p": 11,
    "4p": 12,
    "5p": 13,
    "6p": 14,
    "7p": 15,
    "8p": 16,
    "9p": 17,
    "1s": 18,
    "2s": 19,
    "3s": 20,
    "4s": 21,
    "5s": 22,
    "6s": 23,
    "7s": 24,
    "8s": 25,
    "9s": 26,
    "1z": 27,
    "2z": 28,
    "3z": 29,
    "4z": 30,
    "5z": 31,
    "6z": 32,
    "7z": 33,
}


def parse_tile(tile_str: str) -> tuple[int, bool]:
    """
    牌文字列（例: "5m", "5mr" / 赤ドラ）を牌ID(0..33)と赤ドラフラグに変換するヘルパー関数。
    :param tile_str: 牌の文字列表現（赤5は "5mr", "5pr", "5sr" または "0m", "0p", "0s"）
    :return: (tile_id: int, is_red: bool)
    """
    is_red = False
    if tile_str in ["0m", "5mr"]:
        return 4, True
    elif tile_str in ["0p", "5pr"]:
        return 13, True
    elif tile_str in ["0s", "5sr"]:
        return 22, True

    clean_str = tile_str.replace("r", "")
    tile_id = TILE_STR_TO_ID.get(clean_str, 0)
    return tile_id, is_red


@dataclass
class PlayerState:
    """
    各プレイヤーの対局状態を保持するデータクラス。
    """

    seat: int  # 0: 自家, 1: 下家, 2: 対家, 3: 上家
    score: int = 25000
    discards: list[str] = field(
        default_factory=list
    )  # 打牌履歴（例: ["1m", "5pr", ...]）
    is_tsumogiri: list[bool] = field(default_factory=list)  # ツモ切りフラグのリスト
    melds: list[list[str]] = field(
        default_factory=list
    )  # 副露履歴（例: [["1m", "2m", "3m"]]）
    is_riichi: bool = False
    riichi_turn: int = -1  # 立直宣言巡目（未立直は -1）


@dataclass
class MahjongGameState:
    """
    麻雀の1局面における全観測可能データを保持する構造体。
    """

    self_seat: int  # 自家の席順 (0~3)
    players: list[PlayerState]  # 4人分の状態 (自家から時計回り)
    closed_hand: list[str]  # 自家の門前手牌
    dora_indicators: list[str]  # ドラ指示牌のリスト
    round_wind: int = 0  # 場風 (0:東, 1:南, 2:西, 3:北)
    self_wind: int = 0  # 自風 (0:東, 1:南, 2:西, 3:北)
    honba: int = 0  # 本場数
    kyotaku: int = 0  # 供託立直棒数
    tiles_left: int = 70  # 牌山残り枚数
    is_all_last: bool = False  # オーラスフラグ


class MahjongFeatureExtractor128:
    """
    麻雀のゲーム状態（MahjongGameState）から、
    自研AIモデル用の 128チャネル × 34牌 特徴量テンソル（Feature Tensor）を抽出するクラス。

    テンソル形状: (128, 34)
    データ型: np.float32
    """

    def __init__(self):
        # 128チャネルのゼロ行列を初期化用テンプレートとして保持
        self.num_channels = 128
        self.num_tiles = 34

    def extract(self, game_state: MahjongGameState) -> np.ndarray:
        """
        1局の状態オブジェクトから128チャネルの特徴量テンソルを生成して返します。

        :param game_state: 観測可能な麻雀の1局面データ
        :return: np.ndarray 形状 (128, 34) の float32 テンソル
        """
        tensor = np.zeros((self.num_channels, self.num_tiles), dtype=np.float32)

        # ----------------------------------------------------------------------
        # グループ 1: 自家の手牌および副露状態 (Ch 0 ~ 15)
        # ----------------------------------------------------------------------
        # Ch 0~3: 門前手牌の枚数（1~4枚のバイナリマスク）
        hand_counts = np.zeros(34, dtype=int)
        red_dora_counts = np.zeros(34, dtype=int)

        for tile_str in game_state.closed_hand:
            t_id, is_red = parse_tile(tile_str)
            hand_counts[t_id] += 1
            if is_red:
                red_dora_counts[t_id] += 1

        for count_idx in range(4):
            tensor[count_idx] = (hand_counts >= (count_idx + 1)).astype(np.float32)

        # Ch 4~7: 自家の副露枚数（1~4枚）
        self_player = game_state.players[game_state.self_seat]
        meld_counts = np.zeros(34, dtype=int)
        for meld in self_player.melds:
            for tile_str in meld:
                t_id, is_red = parse_tile(tile_str)
                meld_counts[t_id] += 1
                if is_red:
                    red_dora_counts[t_id] += 1

        for count_idx in range(4):
            tensor[4 + count_idx] = (meld_counts >= (count_idx + 1)).astype(np.float32)

        # Ch 8~11: ドラ指示牌およびドラ枚数
        dora_counts = np.zeros(34, dtype=int)
        for ind_str in game_state.dora_indicators:
            ind_id, _ = parse_tile(ind_str)
            # ドラ指示牌から表ドラIDを算出 (数牌は+1、字牌は東南西北/白發中トイツ)
            if ind_id < 27:
                dora_id = ind_id + 1 if (ind_id % 9 != 8) else ind_id - 8
            elif ind_id < 31:
                dora_id = ind_id + 1 if ind_id < 30 else 27
            else:
                dora_id = ind_id + 1 if ind_id < 33 else 31
            dora_counts[dora_id] += 1

        for count_idx in range(4):
            tensor[8 + count_idx] = (dora_counts >= (count_idx + 1)).astype(np.float32)

        # Ch 12~15: 赤ドラ（赤5m, 赤5p, 赤5s）フラグ
        for count_idx in range(4):
            tensor[12 + count_idx] = (red_dora_counts >= (count_idx + 1)).astype(
                np.float32
            )

        # ----------------------------------------------------------------------
        # グループ 2: 四家の打牌河および副露履歴 (Ch 16 ~ 47)
        # ----------------------------------------------------------------------
        for p_idx in range(4):
            rel_seat = (p_idx - game_state.self_seat) % 4  # 自家基準の相対席順
            player = game_state.players[p_idx]
            base_ch = 16 + rel_seat * 8

            # 河の打牌枚数 (Ch 0~3 relative)
            p_discard_counts = np.zeros(34, dtype=int)
            for tile_str in player.discards:
                t_id, _ = parse_tile(tile_str)
                p_discard_counts[t_id] += 1

            for c in range(4):
                tensor[base_ch + c] = (p_discard_counts >= (c + 1)).astype(np.float32)

            # 副露牌枚数 (Ch 4~7 relative)
            p_meld_counts = np.zeros(34, dtype=int)
            for meld in player.melds:
                for tile_str in meld:
                    t_id, _ = parse_tile(tile_str)
                    p_meld_counts[t_id] += 1

            for c in range(4):
                tensor[base_ch + 4 + c] = (p_meld_counts >= (c + 1)).astype(np.float32)

        # ----------------------------------------------------------------------
        # グループ 3: ツモ切り・手切りおよび立直動向 (Ch 48 ~ 63)
        # ----------------------------------------------------------------------
        for p_idx in range(4):
            rel_seat = (p_idx - game_state.self_seat) % 4
            player = game_state.players[p_idx]

            # Ch 48~51: 直近の打牌がツモ切りか
            if len(player.is_tsumogiri) > 0 and player.is_tsumogiri[-1]:
                last_tile_id, _ = parse_tile(player.discards[-1])
                tensor[48 + rel_seat, last_tile_id] = 1.0

            # Ch 52~55: 直近の打牌的手切りか
            if len(player.is_tsumogiri) > 0 and not player.is_tsumogiri[-1]:
                last_tile_id, _ = parse_tile(player.discards[-1])
                tensor[52 + rel_seat, last_tile_id] = 1.0

            # Ch 56~59: 立直宣言牌のマスク
            if (
                player.is_riichi
                and player.riichi_turn >= 0
                and player.riichi_turn < len(player.discards)
            ):
                r_tile_id, _ = parse_tile(player.discards[player.riichi_turn])
                tensor[56 + rel_seat, r_tile_id] = 1.0

            # Ch 60~63: 立直以降に捨てられた安全牌（立直後通過牌）
            if player.is_riichi and player.riichi_turn >= 0:
                for turn in range(player.riichi_turn, len(player.discards)):
                    t_id, _ = parse_tile(player.discards[turn])
                    tensor[60 + rel_seat, t_id] = 1.0

        # ----------------------------------------------------------------------
        # グループ 4: 動的安全度・危険度マトリクス (Ch 64 ~ 95)
        # ----------------------------------------------------------------------
        # 全家の河牌の論理和（見え牌総数算出用）
        visible_counts = hand_counts + meld_counts
        for p in game_state.players:
            for t_str in p.discards:
                t_id, _ = parse_tile(t_str)
                visible_counts[t_id] += 1

        for p_idx in range(4):
            rel_seat = (p_idx - game_state.self_seat) % 4
            player = game_state.players[p_idx]

            # Ch 64~67: 現物（絶対安全牌）マスク
            if player.is_riichi:
                for t_str in player.discards:
                    t_id, _ = parse_tile(t_str)
                    tensor[64 + rel_seat, t_id] = 1.0

        # Ch 68~71: スジ安全度（1~4段階の簡易スジ判定）
        for suit_offset in [0, 9, 18]:  # マンズ, ピンズ, ソーズ
            for i in range(3):  # 1-4, 2-5, 3-6 スジ
                # 両端が河に見えている場合のスジ判定 logic
                pass1 = visible_counts[suit_offset + i + 3] > 0
                pass2 = visible_counts[suit_offset + i] > 0
                if pass1:
                    tensor[68, suit_offset + i] = 1.0
                if pass2:
                    tensor[69, suit_offset + i + 3] = 1.0

        # Ch 72~75: カベ（No-Chance: 4枚見えている数牌の隣接牌）
        for t_id in range(27):
            if visible_counts[t_id] >= 4:
                # 4枚見えの牌に隣接するスジ牌をカベ安全としてフラグ立て
                num = t_id % 9
                suit = (t_id // 9) * 9
                if num > 0:
                    tensor[72, suit + num - 1] = 1.0
                if num < 8:
                    tensor[73, suit + num + 1] = 1.0

        # Ch 76~79: ワンチャンス (3枚見え)
        for t_id in range(27):
            if visible_counts[t_id] == 3:
                tensor[76, t_id] = 1.0

        # Ch 80~83: 字牌の見え枚数（1~4枚）
        for honor_id in range(27, 34):
            v_cnt = visible_counts[honor_id]
            for c in range(4):
                if v_cnt >= (c + 1):
                    tensor[80 + c, honor_id] = 1.0

        # Ch 84~95: 他家3人への危険度評価（ダミー初期化。実際は推測確率をロード）
        tensor[84:96] = 0.1  # デフォルト危険度

        # ----------------------------------------------------------------------
        # グループ 5: 壁枚数推定量および受け入れ推定 (Ch 96 ~ 111)
        # ----------------------------------------------------------------------
        # Ch 96~99: 全場見え枚数（1~4枚）
        for c in range(4):
            tensor[96 + c] = (visible_counts >= (c + 1)).astype(np.float32)

        # Ch 100~103: 牌山に残っている推定枚数 (0枚, 1枚, 2枚, 3枚以上)
        left_in_wall = 4 - visible_counts
        tensor[100] = (left_in_wall == 0).astype(np.float32)
        tensor[101] = (left_in_wall == 1).astype(np.float32)
        tensor[102] = (left_in_wall == 2).astype(np.float32)
        tensor[103] = (left_in_wall >= 3).astype(np.float32)

        # Ch 104~107: 向聴数（シャンテン数）＆受け入れ枚数変化（プレースホルダー）
        tensor[104:108] = 0.0

        # Ch 108~111: フリテン（振聴）フラグおよび同巡フリテンマスク
        tensor[108:112] = 0.0

        # ----------------------------------------------------------------------
        # グループ 6: 局況・点差・場況コンテキスト (Ch 112 ~ 127)
        # ----------------------------------------------------------------------
        # Ch 112~115: 場風（東南西北） - 全34スロットに同じスカラー値を埋め込み
        tensor[112 + game_state.round_wind, :] = 1.0

        # Ch 116~119: 自風（東南西北）
        tensor[116 + game_state.self_wind, :] = 1.0

        # Ch 120~123: 順位ランク（1位~4位のワンホット）
        scores = [p.score for p in game_state.players]
        self_score = scores[game_state.self_seat]
        rank = sum(1 for s in scores if s > self_score)  # 0: 1位, 1: 2位, ...
        tensor[120 + min(rank, 3), :] = 1.0

        # Ch 124~127: 本場, 供託, 残り牌数ステップ, オーラス
        tensor[124, :] = min(game_state.honba / 10.0, 1.0)
        tensor[125, :] = min(game_state.kyotaku / 5.0, 1.0)
        tensor[126, :] = game_state.tiles_left / 70.0
        tensor[127, :] = 1.0 if game_state.is_all_last else 0.0

        return tensor
