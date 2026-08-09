import os
import json
import glob
import h5py
import numpy as np
import traceback
import gzip
from tqdm import tqdm

# 注意: 向聴数（Shanten）と待ち牌（Waits）を正確に計算するため、
# 標準の mahjong ライブラリを使用します。(pip install mahjong)
from mahjong.shanten import Shanten
from feature_extractor import MahjongFeatureExtractor256, MahjongGameState, PlayerState, parse_tile

# ==========================================
# 1. アクションID定義 (Action ID Definitions)
# ==========================================
ACTION_PASS = 0
ACTION_CHI = 1
ACTION_PON = 2
ACTION_KAN = 3
ACTION_RIICHI = 4
ACTION_HORA = 5

def create_initial_game_state() -> MahjongGameState:
    """
    初期状態の MahjongGameState オブジェクトを生成する。
    （生成初始状态的 MahjongGameState 对象）
    """
    players = [PlayerState(seat=i) for i in range(4)]
    return MahjongGameState(
        self_seat=0,
        players=players,
        closed_hand=[],
        dora_indicators=[],
        global_discards=[]
    )

def calculate_waits_matrix(act_seat: int, current_hands: dict, shanten_calculator: Shanten) -> np.ndarray:
    """
    他家3人の手牌からテンパイ状態（聴牌状態）を判定し、
    102次元（3家 × 34牌）の待ち牌マトリックス（待牌マトリックス）を生成する。
    （计算除行动者外其余三家的听牌待牌矩阵）
    """
    waits_matrix = np.zeros(102, dtype=np.float32)
    
    for rel_p in range(1, 4):
        target_id = (act_seat + rel_p) % 4
        target_hand = current_hands[target_id]
        
        # 34次元の牌カウント配列に変換 (转换为34维计数组)
        tiles_34 = [0] * 34
        for tile_str in target_hand:
            if tile_str != "?":  # 未知の牌は無視
                t_id, _ = parse_tile(tile_str)
                tiles_34[t_id] += 1
                
        # 手牌が13枚（またはそれ以下で副露済み）の場合、向聴数を計算
        if sum(tiles_34) % 3 == 1:
            shanten = shanten_calculator.calculate_shanten(tiles_34)
            # 向聴数 == 0 ならテンパイ（聴牌）
            if shanten == 0:
                # 34種の牌を順番に加えて和了（アガリ / Shanten == -1）になるかテスト
                for i in range(34):
                    if tiles_34[i] < 4:
                        tiles_34[i] += 1
                        if shanten_calculator.calculate_shanten(tiles_34) == -1:
                            # 和了できる牌なら待ち牌として 1.0 を立てる
                            waits_matrix[(rel_p - 1) * 34 + i] = 1.0
                        tiles_34[i] -= 1
                        
    return waits_matrix

class MjaiLogParser:
    """
    MJAI形式のログを解析し、マルチタスク学習用の特徴量とラベルを抽出する。
    V2: 256チャネル対応および aux_waits のラベル抽出を追加。
    """
    def __init__(self, extractor: MahjongFeatureExtractor256):
        self.extractor = extractor
        self.shanten_calculator = Shanten() # 向聴数計算機を初期化

    def parse_file(self, file_path: str):
        main_buffer = {
            'state_2d': [], 'cond_vec': [], 'seq_hist': [],
            't_disc': [], 't_act': [], 'm_disc': [], 'm_act': [],
            't_score': [], 't_tenpai': [], 't_danger': [],
            't_waits': [] # V2: 敵方待牌分布ラベル (Enemy waits distribution) [102次元]
        }

        kyoku_buffer = {
            'actor': [], 'state_2d': [], 'cond_vec': [], 'seq_hist': [],
            't_disc': [], 't_act': [], 'm_disc': [], 'm_act': [],
            'current_hands_snapshot': [] # 毎ステップの他家手牌状態を保存し、後で待ち牌を計算
        }

        game_state = create_initial_game_state()
        player_hands = {0: [], 1: [], 2: [], 3: []}

        try:
            with open(file_path, 'rb') as f_test:
                magic_bytes = f_test.read(2)
            
            if magic_bytes == b'\x1f\x8b':
                f_open = gzip.open(file_path, 'rt', encoding='utf-8')
            else:
                f_open = open(file_path, 'r', encoding='utf-8')

            with f_open as f:
                content = f.read().strip()
                if not content:
                    return None
                
                if content.startswith('['):
                    events = json.loads(content)
                else:
                    events = [json.loads(line) for line in content.split('\n') if line.strip()]
                    
        except Exception as e:
            print(f"\n[デコードエラー] {file_path}: {e}")
            return None

        def flush_kyoku_buffer(score_changes):
            for i in range(len(kyoku_buffer['actor'])):
                act = kyoku_buffer['actor'][i]
                main_buffer['state_2d'].append(kyoku_buffer['state_2d'][i])
                main_buffer['cond_vec'].append(kyoku_buffer['cond_vec'][i])
                main_buffer['seq_hist'].append(kyoku_buffer['seq_hist'][i])
                main_buffer['t_disc'].append(kyoku_buffer['t_disc'][i])
                main_buffer['t_act'].append(kyoku_buffer['t_act'][i])
                main_buffer['m_disc'].append(kyoku_buffer['m_disc'][i])
                main_buffer['m_act'].append(kyoku_buffer['m_act'][i])
                
                delta_score = score_changes[act] / 10000.0 if act < len(score_changes) else 0.0
                main_buffer['t_score'].append(delta_score)
                
                # プレースホルダーとしてゼロ初期化（後続タスクで拡充可能）
                main_buffer['t_tenpai'].append(np.zeros(3, dtype=np.float32))
                main_buffer['t_danger'].append(np.zeros(102, dtype=np.float32))
                
                # V2: 毎ステップ記録した手牌スナップショットから待牌マトリックスを計算
                hands_snapshot = kyoku_buffer['current_hands_snapshot'][i]
                waits_matrix = calculate_waits_matrix(act, hands_snapshot, self.shanten_calculator)
                main_buffer['t_waits'].append(waits_matrix)
                
            for key in kyoku_buffer:
                kyoku_buffer[key].clear()

        try:
            for event in events:
                type_str = event.get("type")

                if type_str == "start_kyoku":
                    game_state = create_initial_game_state()
                    wind_map = {"E": 0, "S": 1, "W": 2, "N": 3}
                    game_state.round_wind = wind_map.get(event.get("bakaze", "E"), 0)
                    game_state.honba = event.get("honba", 0)
                    game_state.kyotaku = event.get("kyotaku", 0)
                    
                    if event.get("dora_marker"):
                        game_state.dora_indicators = [event.get("dora_marker")]

                    for p_idx, tehai in enumerate(event.get("tehais", [[], [], [], []])):
                        player_hands[p_idx] = list(tehai)

                elif type_str == "dora":
                    game_state.dora_indicators.append(event.get("dora_marker"))

                elif type_str == "tsumo":
                    actor = event.get("actor")
                    tile = event.get("pai")
                    if actor is not None and tile != "?":
                        player_hands[actor].append(tile)

                elif type_str == "dahai":
                    actor = event.get("actor")
                    tile = event.get("pai")
                    is_tsumogiri = event.get("tsumogiri", False)

                    if actor is not None and actor in player_hands:
                        game_state.self_seat = actor
                        game_state.closed_hand = list(player_hands[actor])
                        
                        feats = self.extractor.extract(game_state)
                        tile_id, _ = parse_tile(tile)

                        # V2: seq_hist を 0~272 の複合 Token に変換
                        raw_seq = feats.get("seq_hist", np.full(72, 34, dtype=np.int64))
                        relative_player_id = (actor - actor) & 3 
                        cut_type = 0 if is_tsumogiri else 1

                        encoded_seq = []
                        for t in raw_seq:
                            if t >= 34: 
                                encoded_seq.append(272) 
                            else:
                                token = int(t) * 8 + 0 * 2 + cut_type
                                encoded_seq.append(min(token, 272))

                        encoded_seq = np.array(encoded_seq, dtype=np.int64)

                        kyoku_buffer['actor'].append(actor)
                        kyoku_buffer['state_2d'].append(feats["state_2d"])
                        kyoku_buffer['cond_vec'].append(feats["cond_vec"])
                        kyoku_buffer['seq_hist'].append(encoded_seq) 
                        kyoku_buffer['t_disc'].append(tile_id)
                        kyoku_buffer['t_act'].append(ACTION_PASS)
                        kyoku_buffer['m_disc'].append(1.0)
                        kyoku_buffer['m_act'].append(0.0)
                        
                        # 手牌スナップショットをディープコピーして保存 (ディープコピーによる状態の保存)
                        snapshot = {k: list(v) for k, v in player_hands.items()}
                        kyoku_buffer['current_hands_snapshot'].append(snapshot)

                    if tile in player_hands[actor]:
                        player_hands[actor].remove(tile)
                    game_state.players[actor].discards.append(tile)
                    game_state.players[actor].is_tsumogiri.append(is_tsumogiri)
                    game_state.global_discards.append(tile)

                elif type_str in ["chi", "pon", "daiminkan"]:
                    actor = event.get("actor")
                    consumed = event.get("consumed", [])
                    
                    if actor is not None and actor in player_hands:
                        game_state.self_seat = actor
                        game_state.closed_hand = list(player_hands[actor])
                        
                        feats = self.extractor.extract(game_state)
                        raw_seq = feats.get("seq_hist", np.full(72, 34, dtype=np.int64))
                        encoded_seq = np.where(raw_seq >= 34, 272, raw_seq * 8) 
                        
                        act_id = ACTION_CHI if type_str == "chi" else (ACTION_PON if type_str == "pon" else ACTION_KAN)

                        kyoku_buffer['actor'].append(actor)
                        kyoku_buffer['state_2d'].append(feats["state_2d"])
                        kyoku_buffer['cond_vec'].append(feats["cond_vec"])
                        kyoku_buffer['seq_hist'].append(encoded_seq)
                        kyoku_buffer['t_disc'].append(0)
                        kyoku_buffer['t_act'].append(act_id)
                        kyoku_buffer['m_disc'].append(0.0)
                        kyoku_buffer['m_act'].append(1.0)
                        
                        snapshot = {k: list(v) for k, v in player_hands.items()}
                        kyoku_buffer['current_hands_snapshot'].append(snapshot)

                    for c_tile in consumed:
                        if c_tile in player_hands[actor]:
                            player_hands[actor].remove(c_tile)
                    
                    meld_tiles = consumed + [event.get("pai", "")]
                    game_state.players[actor].melds.append(meld_tiles)

                elif type_str == "reach":
                    actor = event.get("actor")
                    if actor is not None:
                        game_state.players[actor].is_riichi = True
                        game_state.players[actor].riichi_turn = len(game_state.players[actor].discards)
                
                elif type_str in ["hora", "ryukyoku"]:
                    scores = event.get("scores", [0, 0, 0, 0])
                    score_changes = [0, 0, 0, 0] 
                    if type_str == "hora":
                        pass
                    flush_kyoku_buffer(score_changes)

        except Exception as e:
            print(f"\n[ロジックエラー] {file_path} の処理中に例外が発生しました:")
            traceback.print_exc()
            return None

        if not main_buffer['state_2d']:
            return None

        return (
            np.array(main_buffer['state_2d'], dtype=np.float32),
            np.array(main_buffer['cond_vec'], dtype=np.float32),
            np.array(main_buffer['seq_hist'], dtype=np.int64), 
            np.array(main_buffer['t_disc'], dtype=np.int64),
            np.array(main_buffer['t_act'], dtype=np.int64),
            np.array(main_buffer['m_disc'], dtype=np.float32),
            np.array(main_buffer['m_act'], dtype=np.float32),
            np.array(main_buffer['t_score'], dtype=np.float32),
            np.array(main_buffer['t_tenpai'], dtype=np.float32),
            np.array(main_buffer['t_danger'], dtype=np.float32),
            np.array(main_buffer['t_waits'], dtype=np.float32) # V2 追加
        )

# ==========================================
# 2. HDF5 データセット構築 (Dataset Builder)
# ==========================================

def build_dataset(log_dir: str, output_dir: str, max_files: int = 5000):
    os.makedirs(output_dir, exist_ok=True)
    mjson_files = glob.glob(os.path.join(log_dir, "**", "*.mjson"), recursive=True)
    if not mjson_files:
        mjson_files = glob.glob(os.path.join(log_dir, "**", "*.json"), recursive=True)

    print(f"[検索完了] 検出された牌譜ファイル数: {len(mjson_files)}")
    if len(mjson_files) == 0:
        return

    selected_files = mjson_files[:max_files]
    split_idx = int(len(selected_files) * 0.9)
    train_files = selected_files[:split_idx]
    val_files = selected_files[split_idx:]

    extractor = MahjongFeatureExtractor256() # V2: 256チャネルExtractorを使用
    parser = MjaiLogParser(extractor)

    def process_and_save(file_list, h5_path):
        print(f"\n--- HDF5 作成開始: {h5_path} ---")
        with h5py.File(h5_path, 'w') as h5f:
            # V2: state_2d のチャネル数を 128 から 256 に変更
            ds_state = h5f.create_dataset('state_2d', shape=(0, 256, 4, 9), maxshape=(None, 256, 4, 9), dtype='float32', chunks=(128, 256, 4, 9))
            ds_cond = h5f.create_dataset('cond_vec', shape=(0, 16), maxshape=(None, 16), dtype='float32')
            ds_seq = h5f.create_dataset('seq_hist', shape=(0, 72), maxshape=(None, 72), dtype='int64')
            
            ds_t_disc = h5f.create_dataset('target_discards', shape=(0,), maxshape=(None,), dtype='int64')
            ds_t_act = h5f.create_dataset('target_actions', shape=(0,), maxshape=(None,), dtype='int64')
            ds_m_disc = h5f.create_dataset('mask_discards', shape=(0,), maxshape=(None,), dtype='float32')
            ds_m_act = h5f.create_dataset('mask_actions', shape=(0,), maxshape=(None,), dtype='float32')
            
            ds_t_score = h5f.create_dataset('target_score', shape=(0,), maxshape=(None,), dtype='float32')
            ds_t_tenpai = h5f.create_dataset('target_tenpai', shape=(0, 3), maxshape=(None, 3), dtype='float32')
            ds_t_danger = h5f.create_dataset('target_danger', shape=(0, 102), maxshape=(None, 102), dtype='float32')
            
            # V2: aux_waits 用のデータセットを追加
            ds_t_waits = h5f.create_dataset('target_waits', shape=(0, 102), maxshape=(None, 102), dtype='float32')

            total_samples = 0
            for fpath in tqdm(file_list, desc="Processing Logs"):
                res = parser.parse_file(fpath)
                if res is None:
                    continue

                state, cond, seq, t_disc, t_act, m_disc, m_act, t_score, t_tenpai, t_danger, t_waits = res
                n_samples = state.shape[0]

                for ds in [ds_state, ds_cond, ds_seq, ds_t_disc, ds_t_act, ds_m_disc, ds_m_act, ds_t_score, ds_t_tenpai, ds_t_danger, ds_t_waits]:
                    ds.resize(total_samples + n_samples, axis=0)

                ds_state[total_samples:total_samples + n_samples] = state
                ds_cond[total_samples:total_samples + n_samples] = cond
                ds_seq[total_samples:total_samples + n_samples] = seq 
                ds_t_disc[total_samples:total_samples + n_samples] = t_disc
                ds_t_act[total_samples:total_samples + n_samples] = t_act
                ds_m_disc[total_samples:total_samples + n_samples] = m_disc
                ds_m_act[total_samples:total_samples + n_samples] = m_act
                ds_t_score[total_samples:total_samples + n_samples] = t_score
                ds_t_tenpai[total_samples:total_samples + n_samples] = t_tenpai
                ds_t_danger[total_samples:total_samples + n_samples] = t_danger
                ds_t_waits[total_samples:total_samples + n_samples] = t_waits # V2: 追加

                total_samples += n_samples

            print(f"[完了] 総サンプル数 (Total Samples): {total_samples}")

    process_and_save(train_files, os.path.join(output_dir, "train_dataset.h5"))
    process_and_save(val_files, os.path.join(output_dir, "val_dataset.h5"))

if __name__ == "__main__":
    LOGS_PATH = "data/logs/2024_mjai"
    OUTPUT_PATH = "data"
    build_dataset(LOGS_PATH, OUTPUT_PATH, max_files=5000)