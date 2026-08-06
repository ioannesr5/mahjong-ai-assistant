from dataclasses import dataclass, field
import numpy as np

# ==============================================================================
# 牌インデックス定義（Tile Index Definition）
# 0..8: マンズ(1m-9m), 9..17: ピンズ(1p-9p), 18..26: ソーズ(1s-9s), 27..33: 字牌(東南西北白發中)
# 空間テンソル(4x9)への変換を見据え、字牌の末尾(34, 35)はパディングとして扱います。
# ==============================================================================

TILE_STR_TO_ID = {
    "1m": 0, "2m": 1, "3m": 2, "4m": 3, "5m": 4, "6m": 5, "7m": 6, "8m": 7, "9m": 8,
    "1p": 9, "2p": 10, "3p": 11, "4p": 12, "5p": 13, "6p": 14, "7p": 15, "8p": 16, "9p": 17,
    "1s": 18, "2s": 19, "3s": 20, "4s": 21, "5s": 22, "6s": 23, "7s": 24, "8s": 25, "9s": 26,
    "1z": 27, "2z": 28, "3z": 29, "4z": 30, "5z": 31, "6z": 32, "7z": 33,
}

def parse_tile(tile_str: str) -> tuple[int, bool]:
    """
    牌文字列（例: "5m", "5mr" / 赤ドラ）を牌ID(0..33)と赤ドラフラグに変換するヘルパー関数。
    （将牌字符串解析为牌 ID 与赤宝牌标志）
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
    （保存各玩家对局状态的数据类）
    """
    seat: int  # 0: 自家, 1: 下家, 2: 対家, 3: 上家
    score: int = 25000
    discards: list[str] = field(default_factory=list)  # 打牌履歴
    is_tsumogiri: list[bool] = field(default_factory=list)  # ツモ切りフラグ
    melds: list[list[str]] = field(default_factory=list)  # 副露履歴
    is_riichi: bool = False
    riichi_turn: int = -1  # 立直宣言巡目

@dataclass
class MahjongGameState:
    """
    麻雀の1局面における全観測可能データを保持する構造体。
    （麻将单局全观测数据）
    """
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
    
    # 新規追加: 時系列エンコーダ(Transformer)用の全家グローバル打牌履歴
    # (新增：用于时序编码器的全家全局打牌历史)
    global_discards: list[str] = field(default_factory=list)

class MahjongFeatureExtractor128:
    """
    自研AIモデル用に、以下の3つの特徴量を抽出するクラス：
    1. state_2d: (128, 4, 9) の2D空間テンソル（空間特徴 / Spatial Tensor）
    2. cond_vec: (16,) のFiLM用条件ベクトル（条件ベクトル / Condition Vector）
    3. seq_hist: (72,) の時系列打牌シーケンス（時系列シーケンス / Sequence History）
    """
    def __init__(self):
        self.num_channels = 128
        self.num_tiles = 34
        self.seq_max_len = 72
        self.pad_id = 34  # ボキャブラリサイズ35のパディングID (Padding ID)

    def extract(self, game_state: MahjongGameState) -> dict:
        """
        1局面データからマルチタスクモデル用の特徴量辞書を抽出します。
        （从单局数据中提取多任务模型所需的三组特征字典）
        """
        # ======================================================================
        # 1. 2D空間テンソル (Spatial Tensor) の構築
        # ======================================================================
        tensor_1d = np.zeros((self.num_channels, self.num_tiles), dtype=np.float32)

        # (中略) 以前の実装と全く同じ論理で 128x34 のテンソルを構築します。
        # 欠落を防ぐため、元のロジックを完全再現（完整保留原特征提取逻辑，确保特征无损）
        hand_counts = np.zeros(34, dtype=int)
        red_dora_counts = np.zeros(34, dtype=int)

        for tile_str in game_state.closed_hand:
            t_id, is_red = parse_tile(tile_str)
            hand_counts[t_id] += 1
            if is_red:
                red_dora_counts[t_id] += 1

        for count_idx in range(4):
            tensor_1d[count_idx] = (hand_counts >= (count_idx + 1)).astype(np.float32)

        self_player = game_state.players[game_state.self_seat]
        meld_counts = np.zeros(34, dtype=int)
        for meld in self_player.melds:
            for tile_str in meld:
                t_id, is_red = parse_tile(tile_str)
                meld_counts[t_id] += 1
                if is_red:
                    red_dora_counts[t_id] += 1

        for count_idx in range(4):
            tensor_1d[4 + count_idx] = (meld_counts >= (count_idx + 1)).astype(np.float32)

        dora_counts = np.zeros(34, dtype=int)
        for ind_str in game_state.dora_indicators:
            ind_id, _ = parse_tile(ind_str)
            if ind_id < 27:
                dora_id = ind_id + 1 if (ind_id % 9 != 8) else ind_id - 8
            elif ind_id < 31:
                dora_id = ind_id + 1 if ind_id < 30 else 27
            else:
                dora_id = ind_id + 1 if ind_id < 33 else 31
            dora_counts[dora_id] += 1

        for count_idx in range(4):
            tensor_1d[8 + count_idx] = (dora_counts >= (count_idx + 1)).astype(np.float32)

        for count_idx in range(4):
            tensor_1d[12 + count_idx] = (red_dora_counts >= (count_idx + 1)).astype(np.float32)

        for p_idx in range(4):
            rel_seat = (p_idx - game_state.self_seat) % 4
            player = game_state.players[p_idx]
            base_ch = 16 + rel_seat * 8

            p_discard_counts = np.zeros(34, dtype=int)
            for tile_str in player.discards:
                t_id, _ = parse_tile(tile_str)
                p_discard_counts[t_id] += 1
            for c in range(4):
                tensor_1d[base_ch + c] = (p_discard_counts >= (c + 1)).astype(np.float32)

            p_meld_counts = np.zeros(34, dtype=int)
            for meld in player.melds:
                for tile_str in meld:
                    t_id, _ = parse_tile(tile_str)
                    p_meld_counts[t_id] += 1
            for c in range(4):
                tensor_1d[base_ch + 4 + c] = (p_meld_counts >= (c + 1)).astype(np.float32)

            if len(player.is_tsumogiri) > 0 and player.is_tsumogiri[-1]:
                last_tile_id, _ = parse_tile(player.discards[-1])
                tensor_1d[48 + rel_seat, last_tile_id] = 1.0

            if len(player.is_tsumogiri) > 0 and not player.is_tsumogiri[-1]:
                last_tile_id, _ = parse_tile(player.discards[-1])
                tensor_1d[52 + rel_seat, last_tile_id] = 1.0

            if player.is_riichi and player.riichi_turn >= 0 and player.riichi_turn < len(player.discards):
                r_tile_id, _ = parse_tile(player.discards[player.riichi_turn])
                tensor_1d[56 + rel_seat, r_tile_id] = 1.0

            if player.is_riichi and player.riichi_turn >= 0:
                for turn in range(player.riichi_turn, len(player.discards)):
                    t_id, _ = parse_tile(player.discards[turn])
                    tensor_1d[60 + rel_seat, t_id] = 1.0

        visible_counts = hand_counts + meld_counts
        for p in game_state.players:
            for t_str in p.discards:
                t_id, _ = parse_tile(t_str)
                visible_counts[t_id] += 1

        for p_idx in range(4):
            rel_seat = (p_idx - game_state.self_seat) % 4
            player = game_state.players[p_idx]
            if player.is_riichi:
                for t_str in player.discards:
                    t_id, _ = parse_tile(t_str)
                    tensor_1d[64 + rel_seat, t_id] = 1.0

        for suit_offset in [0, 9, 18]:
            for i in range(3):
                if visible_counts[suit_offset + i + 3] > 0:
                    tensor_1d[68, suit_offset + i] = 1.0
                if visible_counts[suit_offset + i] > 0:
                    tensor_1d[69, suit_offset + i + 3] = 1.0

        for t_id in range(27):
            if visible_counts[t_id] >= 4:
                num = t_id % 9
                suit = (t_id // 9) * 9
                if num > 0:
                    tensor_1d[72, suit + num - 1] = 1.0
                if num < 8:
                    tensor_1d[73, suit + num + 1] = 1.0
            if visible_counts[t_id] == 3:
                tensor_1d[76, t_id] = 1.0

        for honor_id in range(27, 34):
            v_cnt = visible_counts[honor_id]
            for c in range(4):
                if v_cnt >= (c + 1):
                    tensor_1d[80 + c, honor_id] = 1.0

        tensor_1d[84:96] = 0.1

        for c in range(4):
            tensor_1d[96 + c] = (visible_counts >= (c + 1)).astype(np.float32)

        left_in_wall = 4 - visible_counts
        tensor_1d[100] = (left_in_wall == 0).astype(np.float32)
        tensor_1d[101] = (left_in_wall == 1).astype(np.float32)
        tensor_1d[102] = (left_in_wall == 2).astype(np.float32)
        tensor_1d[103] = (left_in_wall >= 3).astype(np.float32)

        tensor_1d[104:112] = 0.0
        tensor_1d[112 + game_state.round_wind, :] = 1.0
        tensor_1d[116 + game_state.self_wind, :] = 1.0

        scores = [p.score for p in game_state.players]
        self_score = scores[game_state.self_seat]
        rank = sum(1 for s in scores if s > self_score)
        tensor_1d[120 + min(rank, 3), :] = 1.0

        tensor_1d[124, :] = min(game_state.honba / 10.0, 1.0)
        tensor_1d[125, :] = min(game_state.kyotaku / 5.0, 1.0)
        tensor_1d[126, :] = game_state.tiles_left / 70.0
        tensor_1d[127, :] = 1.0 if game_state.is_all_last else 0.0

        # --- 2D空間へのトランスフォーム (Transform to 2D Spatial Tensor) ---
        # 128x34 -> 128x36 (パディング) -> 128x4x9 へのリシェイプ
        padded_tensor = np.pad(tensor_1d, ((0, 0), (0, 2)), mode='constant', constant_values=0)
        state_2d = padded_tensor.reshape((self.num_channels, 4, 9))

        # ======================================================================
        # 2. FiLM用 条件ベクトル (Condition Vector) の構築 (16次元)
        # ======================================================================
        cond_vec = np.zeros(16, dtype=np.float32)
        
        # [0:4] 相対スコア (Relative Scores normalized)
        for i in range(4):
            rel_idx = (game_state.self_seat + i) % 4
            cond_vec[i] = (game_state.players[rel_idx].score - 25000) / 10000.0
            
        # [4:8] 場風 (Round Wind One-Hot)
        cond_vec[4 + game_state.round_wind] = 1.0
        
        # [8:12] 自風 (Self Wind One-Hot)
        cond_vec[8 + game_state.self_wind] = 1.0
        
        # [12:16] その他のマクロ状態 (Other Macro States)
        cond_vec[12] = min(game_state.honba / 10.0, 1.0)
        cond_vec[13] = min(game_state.kyotaku / 5.0, 1.0)
        cond_vec[14] = game_state.tiles_left / 70.0
        cond_vec[15] = 1.0 if game_state.is_all_last else 0.0

        # ======================================================================
        # 3. 時系列打牌シーケンス (Sequence History) の構築 (72次元)
        # ======================================================================
        seq_hist = np.full(self.seq_max_len, self.pad_id, dtype=np.int64)
        
        # グローバル打牌履歴からシーケンスを抽出
        actual_len = min(len(game_state.global_discards), self.seq_max_len)
        for i in range(actual_len):
            tile_id, _ = parse_tile(game_state.global_discards[i])
            seq_hist[i] = tile_id

        # ======================================================================
        # 抽出結果の返却 (Return as Dictionary)
        # ======================================================================
        return {
            "state_2d": state_2d,
            "cond_vec": cond_vec,
            "seq_hist": seq_hist
        }

if __name__ == "__main__":
    # テストと出力検証
    extractor = MahjongFeatureExtractor128()
    test_state = MahjongGameState(
        self_seat=0,
        players=[PlayerState(seat=i) for i in range(4)],
        closed_hand=["1m", "9p", "5sr"],
        dora_indicators=["4m"],
        global_discards=["1m", "2p", "3s", "1z"]
    )
    
    features = extractor.extract(test_state)
    print(f"2D空間テンソル形状 (state_2d shape): {features['state_2d'].shape}") # (128, 4, 9)
    print(f"条件ベクトル形状 (cond_vec shape): {features['cond_vec'].shape}")   # (16,)
    print(f"時系列シーケンス形状 (seq_hist shape): {features['seq_hist'].shape}") # (72,)