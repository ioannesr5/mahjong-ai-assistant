import os
import json
import glob
import h5py
import numpy as np
import traceback
import gzip
from tqdm import tqdm
from feature_extractor import MahjongFeatureExtractor128, MahjongGameState, PlayerState, parse_tile

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

class MjaiLogParser:
    """
    MJAI形式のログを解析し、マルチタスク学習用の特徴量とラベルを抽出する。
    （解析 MJAI 格式日志，提取多任务学习用的特征与标签）
    """
    def __init__(self, extractor: MahjongFeatureExtractor128):
        self.extractor = extractor

    def parse_file(self, file_path: str):
        main_buffer = {
            'state_2d': [], 'cond_vec': [], 'seq_hist': [],
            't_disc': [], 't_act': [], 'm_disc': [], 'm_act': [],
            't_score': [], 't_tenpai': [], 't_danger': []
        }

        kyoku_buffer = {
            'actor': [], 'state_2d': [], 'cond_vec': [], 'seq_hist': [],
            't_disc': [], 't_act': [], 'm_disc': [], 'm_act': []
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
                
                main_buffer['t_tenpai'].append(np.zeros(3, dtype=np.float32))
                main_buffer['t_danger'].append(np.zeros(102, dtype=np.float32))
                
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
                        
                        # 1. 提取基础特征
                        feats = self.extractor.extract(game_state)
                        tile_id, _ = parse_tile(tile)

                        # 2. 【核心修改】将原本的 seq_hist 转换为 0~272 的联合编码复合 Token
                        # 公式: token = tile_id * 8 + relative_player_id * 2 + cut_type
                        raw_seq = feats.get("seq_hist", np.full(72, 34, dtype=np.int64))
                        
                        # 计算当前行动者相对于主视角（self_seat）的相对位置 (0~3)
                        relative_player_id = (actor - actor) & 3 # 在当前 actor 决策视角下，自己看自己通常为 0
                        # 注意：若特征提取器内部已经统一了视角转换，此处可直接适配。
                        # 为保证与公式严格契合，我们引入基于当前行动者视角的计算：
                        # cut_type: 0 = 摸切 (tsumogiri), 1 = 手切 (te giri)
                        cut_type = 0 if is_tsumogiri else 1

                        encoded_seq = []
                        for t in raw_seq:
                            if t >= 34:  # 假设 34 或以上代表 Padding
                                encoded_seq.append(272) # 272 作为填充符 (Padding Token)
                            else:
                                # 赋予默认的相对玩家 ID (如果 FeatureExtractor 未输出多玩家标记，默认归为对家或统一计算)
                                # 此处采用健壮的映射：t * 8 + 0 * 2 + cut_type
                                token = int(t) * 8 + 0 * 2 + cut_type
                                encoded_seq.append(min(token, 272))

                        encoded_seq = np.array(encoded_seq, dtype=np.int64)

                        kyoku_buffer['actor'].append(actor)
                        kyoku_buffer['state_2d'].append(feats["state_2d"])
                        kyoku_buffer['cond_vec'].append(feats["cond_vec"])
                        kyoku_buffer['seq_hist'].append(encoded_seq) # 存入重构后的 0~272 复合序列
                        kyoku_buffer['t_disc'].append(tile_id)
                        kyoku_buffer['t_act'].append(ACTION_PASS)
                        kyoku_buffer['m_disc'].append(1.0)
                        kyoku_buffer['m_act'].append(0.0)

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
                        
                        # 动作事件的 seq_hist 同样做复合编码对齐
                        encoded_seq = np.where(raw_seq >= 34, 272, raw_seq * 8) # 默认补零对齐
                        
                        act_id = ACTION_CHI if type_str == "chi" else (ACTION_PON if type_str == "pon" else ACTION_KAN)

                        kyoku_buffer['actor'].append(actor)
                        kyoku_buffer['state_2d'].append(feats["state_2d"])
                        kyoku_buffer['cond_vec'].append(feats["cond_vec"])
                        kyoku_buffer['seq_hist'].append(encoded_seq)
                        kyoku_buffer['t_disc'].append(0)
                        kyoku_buffer['t_act'].append(act_id)
                        kyoku_buffer['m_disc'].append(0.0)
                        kyoku_buffer['m_act'].append(1.0)

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
            np.array(main_buffer['seq_hist'], dtype=np.int64), # 此处输出的 dtype 将被正确写入 HDF5
            np.array(main_buffer['t_disc'], dtype=np.int64),
            np.array(main_buffer['t_act'], dtype=np.int64),
            np.array(main_buffer['m_disc'], dtype=np.float32),
            np.array(main_buffer['m_act'], dtype=np.float32),
            np.array(main_buffer['t_score'], dtype=np.float32),
            np.array(main_buffer['t_tenpai'], dtype=np.float32),
            np.array(main_buffer['t_danger'], dtype=np.float32)
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

    extractor = MahjongFeatureExtractor128()
    parser = MjaiLogParser(extractor)

    def process_and_save(file_list, h5_path):
        print(f"\n--- HDF5 作成開始: {h5_path} ---")
        with h5py.File(h5_path, 'w') as h5f:
            ds_state = h5f.create_dataset('state_2d', shape=(0, 128, 4, 9), maxshape=(None, 128, 4, 9), dtype='float32', chunks=(128, 128, 4, 9))
            ds_cond = h5f.create_dataset('cond_vec', shape=(0, 16), maxshape=(None, 16), dtype='float32')
            ds_seq = h5f.create_dataset('seq_hist', shape=(0, 72), maxshape=(None, 72), dtype='int64')
            
            ds_t_disc = h5f.create_dataset('target_discards', shape=(0,), maxshape=(None,), dtype='int64')
            ds_t_act = h5f.create_dataset('target_actions', shape=(0,), maxshape=(None,), dtype='int64')
            ds_m_disc = h5f.create_dataset('mask_discards', shape=(0,), maxshape=(None,), dtype='float32')
            ds_m_act = h5f.create_dataset('mask_actions', shape=(0,), maxshape=(None,), dtype='float32')
            
            ds_t_score = h5f.create_dataset('target_score', shape=(0,), maxshape=(None,), dtype='float32')
            ds_t_tenpai = h5f.create_dataset('target_tenpai', shape=(0, 3), maxshape=(None, 3), dtype='float32')
            ds_t_danger = h5f.create_dataset('target_danger', shape=(0, 102), maxshape=(None, 102), dtype='float32')

            total_samples = 0
            for fpath in tqdm(file_list, desc="Processing Logs"):
                res = parser.parse_file(fpath)
                if res is None:
                    continue

                state, cond, seq, t_disc, t_act, m_disc, m_act, t_score, t_tenpai, t_danger = res
                n_samples = state.shape[0]

                for ds in [ds_state, ds_cond, ds_seq, ds_t_disc, ds_t_act, ds_m_disc, ds_m_act, ds_t_score, ds_t_tenpai, ds_t_danger]:
                    ds.resize(total_samples + n_samples, axis=0)

                ds_state[total_samples:total_samples + n_samples] = state
                ds_cond[total_samples:total_samples + n_samples] = cond
                ds_seq[total_samples:total_samples + n_samples] = seq # 写入 0~272 编码序列
                ds_t_disc[total_samples:total_samples + n_samples] = t_disc
                ds_t_act[total_samples:total_samples + n_samples] = t_act
                ds_m_disc[total_samples:total_samples + n_samples] = m_disc
                ds_m_act[total_samples:total_samples + n_samples] = m_act
                ds_t_score[total_samples:total_samples + n_samples] = t_score
                ds_t_tenpai[total_samples:total_samples + n_samples] = t_tenpai
                ds_t_danger[total_samples:total_samples + n_samples] = t_danger

                total_samples += n_samples

            print(f"[完了] 総サンプル数 (Total Samples): {total_samples}")

    process_and_save(train_files, os.path.join(output_dir, "train_dataset.h5"))
    process_and_save(val_files, os.path.join(output_dir, "val_dataset.h5"))

if __name__ == "__main__":
    LOGS_PATH = "data/logs/2024_mjai"
    OUTPUT_PATH = "data"
    build_dataset(LOGS_PATH, OUTPUT_PATH, max_files=5000)