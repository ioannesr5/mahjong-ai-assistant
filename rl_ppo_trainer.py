import csv
import gc
import glob
import os
import random
import re
import sys
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch_directml
from torch import multiprocessing as mp
from torch import nn
from tqdm import tqdm

import actions as A
from mahjong_env import MultiAgentMahjongEnvWrapper
from models import (
    DirectMLSafeAdamW,
    SmartMahjongMultiTaskNet,
    directml_safe_bce_with_logits,
    directml_safe_masked_sample,
    masked_log_probs,
)
from reward import (
    COMPONENTS,
    RewardConfig,
    RewardDecomposition,
    hanchan_rank_bonus,
    ukeire_count,
)


def _use_utf8_stdout() -> None:
    """Windows の既定コンソール (cp932/gbk) で日本語ログが落ちないようにする"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


_use_utf8_stdout()


# 報酬成分を個別に記録する (旧: 1 つのスカラーしか無く、学習を動かしている要因を切り分けられなかった)
TRAIN_LOG_HEADER = [
    "Iteration", "Phase", "Loss", "Reward", "Entropy", "WinRate", "DealInRate", "MeanShantenRed",
    *[f"R_{name}" for name in COMPONENTS],
]


class TrainingLogger:
    """
    CSVベースの永続化ロガー（断点続行時のデータ重複・不整合を防止する自動切り詰め・同期機能付き）
    """

    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.train_log_path = os.path.join(log_dir, "rl_train_log.csv")
        self.eval_log_path = os.path.join(log_dir, "rl_eval_log.csv")

        # 訓練ログのヘッダー初期化（監視メトリクスを追加）
        if not os.path.exists(self.train_log_path):
            with open(self.train_log_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    TRAIN_LOG_HEADER
                )

        # 評価ログのヘッダー初期化
        if not os.path.exists(self.eval_log_path):
            with open(self.eval_log_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["Iteration", "Phase", "AvgRank", "AvgNet", "WinRate", "DealInRate"])

    def truncate_after(self, start_it: int):
        """
        断点続行（レジューム）時、再開イテレーション (start_it) より後の古い重複レコードを自動削除し、ログの単調性を維持する
        """
        if start_it < 0:
            return

        for path, header in [
            (
                self.train_log_path,
                TRAIN_LOG_HEADER,
            ),
            (
                self.eval_log_path,
                ["Iteration", "Phase", "AvgRank", "AvgNet", "WinRate", "DealInRate"],
            ),
        ]:
            if not os.path.exists(path):
                continue
            try:
                valid_rows = []
                with open(path, newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    file_header = next(reader, None)
                    if file_header:
                        for row in reader:
                            if not row:
                                continue
                            try:
                                row_it = int(row[0])
                                if row_it <= start_it:
                                    valid_rows.append(row)
                            except (ValueError, IndexError):
                                pass

                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(valid_rows)
            except Exception as e:
                print(f"⚠️ [Logger Warning] {path} のログ整理中にエラーが発生しました: {e}")

    def log_train(self, it, phase, loss, reward, entropy, win_r, deal_r, shanten_red, parts=None):
        parts = parts or {}
        with open(self.train_log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    it,
                    phase,
                    f"{loss:.4f}",
                    f"{reward:.4f}",
                    f"{entropy:.4f}",
                    f"{win_r:.4f}",
                    f"{deal_r:.4f}",
                    f"{shanten_red:.4f}",
                    *[f"{parts.get(name, 0.0):.4f}" for name in COMPONENTS],
                ]
            )

    def log_eval(self, it, phase, rank, net, win_r, deal_r):
        with open(self.eval_log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([it, phase, f"{rank:.3f}", f"{net:.1f}", f"{win_r:.4f}", f"{deal_r:.4f}"])


CHECKPOINT_FORMAT = 2  # 1 = 生の state_dict、2 = 学習状態を含む辞書


def build_checkpoint(model, trainer, phase, iteration, reward_config, extra=None):
    """
    【修正】旧実装は model.state_dict() だけを保存していたため
      * optimizer / scheduler / phase / kl_beta が失われ、再開のたびに学習が不連続になる
      * TRUE_BEST (ファイル名に _iter が無い) から再開すると it が 0 に戻り、
        直後の logger.truncate_after(0) が CSV 全体を消し飛ばす
      * どの報酬設定・どの seed で得た重みか後から辿れない
    という問題があった。学習状態を丸ごと保存する。
    """
    return {
        "format": CHECKPOINT_FORMAT,
        "model": model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "phase": phase,
        "iteration": iteration,
        "kl_beta": trainer.kl_beta,
        "gamma": trainer.gamma,
        "reward_config": reward_config.as_dict(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        **(extra or {}),
    }


def extract_model_state(payload):
    """新旧どちらの形式のチェックポイントからでもモデル重みを取り出す"""
    if isinstance(payload, dict) and "model" in payload and "format" in payload:
        return payload["model"]
    return payload


def restore_training_state(payload, trainer):
    """辞書形式なら optimizer / kl_beta / RNG も復元し、再開イテレーションを返す"""
    if not (isinstance(payload, dict) and payload.get("format") == CHECKPOINT_FORMAT):
        return None
    try:
        trainer.optimizer.load_state_dict(payload["optimizer"])
    except (ValueError, KeyError) as exc:
        print(f" -> [Warn] optimizer 状態の復元に失敗しました ({exc})。初期状態から続行します")
    trainer.kl_beta = payload.get("kl_beta", trainer.kl_beta)
    if "torch_rng" in payload:
        torch.set_rng_state(payload["torch_rng"])
    if "numpy_rng" in payload:
        np.random.set_state(payload["numpy_rng"])
    if "python_rng" in payload:
        random.setstate(payload["python_rng"])
    return payload.get("iteration")


class CheckpointManager:
    """
    チェックポイントおよびファイルローテーション管理モジュール
    """

    def __init__(self, max_keep=3):
        self.max_keep = max_keep

    def safe_save(self, state_dict, filepath):
        temp_path = filepath + ".tmp"
        torch.save(state_dict, temp_path)
        os.replace(temp_path, filepath)

    def save_with_rotation(self, state_dict, prefix, current_phase, iteration):
        filename = f"{prefix}_phase{current_phase}_iter{iteration}.pth"
        self.safe_save(state_dict, filename)

        pattern = f"{prefix}_phase{current_phase}_iter*.pth"
        files = glob.glob(pattern)
        files.sort(key=os.path.getmtime, reverse=True)

        for f in files[self.max_keep :]:
            try:
                os.remove(f)
            except OSError:
                pass


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ==========================================
# 1. カスタムオプティマイザ (DirectML Safe AdamW)
# ==========================================
# ==========================================
# 2. ネットワーク構成要素 (SmartMahjongMultiTaskNet V2)
# ==========================================
def adapt_policy_state_dict(sd, target_actions=A.N_ACTIONS):
    """
    旧47次元チェックポイントの重みを54次元アクション空間へ移行する。

    写像は actions.LEGACY47_TO_54 (= observation_action_explanation.pdf Table 5) が唯一の根拠。
    旧空間に対応物が無い新アクション (赤ドラ打牌・チー/ポン/カンの赤変種) は、
    意味的に最も近い旧アクションの重みで初期化する。
    """
    if "policy_out.weight" not in sd:
        return sd
    w = sd["policy_out.weight"]
    b = sd["policy_out.bias"]
    if w.shape[0] == target_actions:
        return sd
    if w.shape[0] != A.Legacy47.N_ACTIONS or target_actions != A.N_ACTIONS:
        raise ValueError(
            f"未知のアクション次元です: checkpoint={w.shape[0]}, target={target_actions}"
        )

    L = A.Legacy47
    new_w = torch.zeros((A.N_ACTIONS, w.shape[1]), dtype=w.dtype)
    new_b = torch.zeros(A.N_ACTIONS, dtype=b.dtype)

    # 1) 旧空間に厳密な対応があるもの (打牌 0..33 / チー3種 / ポン / カン3種 / 立直 / ロン / ツモ / 九種 / パス2種)
    for legacy_id, new_id in A.LEGACY47_TO_54.items():
        new_w[new_id] = w[legacy_id]
        new_b[new_id] = b[legacy_id]

    # 2) 赤ドラ打牌 (34..36) <- 対応する通常5の打牌
    for tile_id, red_action in A.RED_DISCARD_OF_TILE.items():
        new_w[red_action] = w[tile_id]
        new_b[red_action] = b[tile_id]

    # 3) 赤を使う鳴き変種 <- 赤を使わない同種の鳴き
    for red_action, base_action in (
        (A.CHI_LEFT_RED, L.CHI_SMALL),
        (A.CHI_MIDDLE_RED, L.CHI_MIDDLE),
        (A.CHI_RIGHT_RED, L.CHI_LARGE),
        (A.PON_RED, L.PON),
    ):
        new_w[red_action] = w[base_action]
        new_b[red_action] = b[base_action]

    sd_copy = dict(sd)
    sd_copy["policy_out.weight"] = new_w
    sd_copy["policy_out.bias"] = new_b
    return sd_copy


# --- 死んだアクションヘッドの復活 (Dead Action Head Revival) ---------------
# 旧 SL データセットには reach / hora / pass サンプルが一切含まれていなかったため、
# 立直・ロン・ツモ・九種・パスに対応する出力行は「初期化されたまま」学習されていない。
# その結果「和了できる場面でパスする」「鳴ける場面で必ず鳴く」退化方策になっていた。
# ここでそれらの行を再初期化し、麻雀の常識に沿った事前分布を与える。
# 阶段2 で正しいラベルを持つデータセットを再構築し SL を再学習したら、この処置は不要になる。

# 「和了できるなら和了する」は大半の局面でほぼ最適なので、正のバイアスで温かく起動する。
AGARI_WARM_START_BIAS = 2.0
# 「門前聴牌なら立直する」も既定としては妥当 (ダマは例外)。
RIICHI_WARM_START_BIAS = 1.0
# 鳴きロジットへの追加補正。
#
# 実測メモ: 旧 SL は「鳴いた局面」しか学習しておらず、鳴き/パスの選択に有効な信号を
# 一切持たない。そのため鳴きロジットとパスロジットの差 Δ を掃引しても応答は
# ナイフエッジ (Δ=1.5 で 6%、Δ=1.94 で 60~79%) で、人間並みの鳴き率 (25~40%) に
# 調整することは原理的に不可能である。
# ここでは「鳴きすぎて門前率 0 になり和了が消える」退化だけを防げばよいので、
# 追加補正は 0 とし、パスロジットの復活 (bias 0) による自然な抑制に任せる。
# 本来の鳴き判断は 阶段2 の PASS 負例を含むデータセットで SL を再学習して獲得する。
DEFAULT_CALL_LOGIT_PENALTY = 0.0


def diagnose_policy_head(sd, dead_ratio=0.5):
    """
    policy_out の各行の重みノルムを見て「一度も学習されていない行」を検出する。
    打牌行 (0..33) の中央値の dead_ratio 倍を下回る行を死行とみなす。
    """
    w = sd["policy_out.weight"].float()
    norms = w.norm(dim=1)
    reference = float(norms[A.DISCARD_START : A.DISCARD_END].median())
    threshold = reference * dead_ratio
    dead = [i for i in range(w.shape[0]) if float(norms[i]) < threshold]
    return dead, reference, threshold


def repair_policy_head(sd, call_logit_penalty=DEFAULT_CALL_LOGIT_PENALTY, verbose=True):
    """
    死んだ出力行を復活させ、行動事前分布を注入した state_dict を返す (in-place ではない)。
    ネットワーク構造は一切変更しない。変わるのは policy_out の重みの初期値のみ。

    復活の方針:
      * 意味的に対応する「生きている兄弟行」があればそれを複製する
        (チー中/右 <- チー左、暗槓/加槓 <- 明槓、字牌打牌 <- 么九牌打牌の平均)
      * 兄弟が無い宣言系 (立直/ロン/ツモ/九種/パス) は再初期化し、事前分布バイアスを与える

    これは阶段2 でラベルを修正したデータセットを作り、阶段4 で SL を再学習するまでの
    足場 (scaffolding) である。再学習後は MJ_REPAIR_HEAD=0 で無効化すること。
    """
    sd = dict(sd)
    w = sd["policy_out.weight"].clone().float()
    b = sd["policy_out.bias"].clone().float()

    dead, reference, threshold = diagnose_policy_head(sd)
    dead_set = set(dead)
    if verbose:
        print(f" -> [Head Repair] 打牌行の重みノルム中央値={reference:.3f} (死行判定閾値={threshold:.3f})")
        print(f" -> [Head Repair] 検出された死行 ({len(dead)}): {[A.ACTION_NAMES[i] for i in dead]}")
    if not dead_set:
        if verbose:
            print(" -> [Head Repair] 死行なし。修復をスキップします")
        return sd

    repaired = []

    def clone_from(target, donors, note):
        """donors のうち生きている行の平均を target にコピーする"""
        alive = [d for d in donors if d not in dead_set]
        if not alive:
            return False
        stacked = torch.stack([w[d] for d in alive])
        averaged = stacked.mean(dim=0)
        # 平均は方向が打ち消し合ってノルムが縮むので、ドナーの平均ノルムに揃え直す
        # (揃えないとロジットのスケールが小さくなり、当該アクションが選ばれにくくなる)
        target_norm = float(stacked.norm(dim=1).mean())
        averaged = averaged * (target_norm / max(float(averaged.norm()), 1e-8))
        w[target] = averaged
        b[target] = torch.stack([b[d] for d in alive]).mean()
        repaired.append(f"{A.ACTION_NAMES[target]}<-{note}")
        return True

    # 1) 鳴きの変種: 生きている同種の鳴きから複製する
    #    (旧データセットは全てのチーを 1 つのラベルに潰していたため、中/右チーが死んでいる)
    for group in (A.CHI_ACTIONS, A.PON_ACTIONS, A.KAN_ACTIONS):
        for target in group:
            if target in dead_set:
                clone_from(target, list(group), "同種の鳴き")

    # 2) 字牌の打牌 (27..33): 么九牌の打牌行の平均から復活させる
    #    (parse_tile の字牌バグにより、旧データセットでは字牌打牌ラベルが 1 件も存在しなかった)
    TERMINALS = [0, 8, 9, 17, 18, 26]
    for target in range(27, 34):
        if target in dead_set:
            clone_from(target, TERMINALS, "么九牌打牌の平均")

    # 3) 宣言系: 対応する兄弟が存在しないので再初期化 + 事前分布バイアス
    dead_declarations = [i for i in A.DECLARATION_ACTIONS if i in dead_set]
    if dead_declarations:
        fan_in = w.shape[1]
        std = float(w[A.DISCARD_START : A.DISCARD_END].std())
        generator = torch.Generator().manual_seed(20260818)
        for i in dead_declarations:
            w[i] = torch.randn(fan_in, generator=generator) * std
            b[i] = 0.0
            repaired.append(f"{A.ACTION_NAMES[i]}<-再初期化")
        # 「和了できるなら和了する」は大半の局面でほぼ最適なので温かく起動する
        b[A.RON] = AGARI_WARM_START_BIAS
        b[A.TSUMO] = AGARI_WARM_START_BIAS
        b[A.RIICHI] = RIICHI_WARM_START_BIAS
        b[A.KYUSHUKYUHAI] = -1.0  # 九種九牌が正解の局面は稀

    if call_logit_penalty:
        # 鳴きロジットを一律に押し下げる。バイアス項に畳み込むため
        # サンプリング・更新・評価のすべてで一貫し、PPO は学習でこれを打ち消せる。
        for i in A.CALL_ACTIONS:
            b[i] -= call_logit_penalty

    if verbose:
        print(f" -> [Head Repair] 復活: {', '.join(repaired)}")
        if dead_declarations:
            print(
                f" -> [Head Repair] RON/TSUMO に +{AGARI_WARM_START_BIAS}、"
                f"RIICHI に +{RIICHI_WARM_START_BIAS} のウォームスタートバイアス"
            )
        if call_logit_penalty:
            print(f" -> [Head Repair] 鳴き {len(A.CALL_ACTIONS)} 件に -{call_logit_penalty} のロジット補正")

    sd["policy_out.weight"] = w.to(sd["policy_out.weight"].dtype)
    sd["policy_out.bias"] = b.to(sd["policy_out.bias"].dtype)
    return sd


class PolicyInferenceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, state_2d, cond_vec, seq_hist):
        p_out, v_head, _, _, _ = self.model(state_2d, cond_vec, seq_hist, True)
        return p_out, v_head


# ==========================================
# 4. 独立SL凍結評価モジュール (Independent SL Eval)
# ==========================================
def evaluation_worker(worker_id, rl_sd, sl_sd, num_hanchan, result_queue):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    env = MultiAgentMahjongEnvWrapper()
    device = torch.device("cpu")

    rl_model = SmartMahjongMultiTaskNet(input_channels=256, num_blocks=18).to(device)
    sl_model = SmartMahjongMultiTaskNet(input_channels=256, num_blocks=18).to(device)
    rl_model.load_state_dict(rl_sd)
    sl_model.load_state_dict(sl_sd)
    rl_model.eval()
    sl_model.eval()

    rl_wrapper = PolicyInferenceWrapper(rl_model)
    sl_wrapper = PolicyInferenceWrapper(sl_model)

    dummy_s2d = torch.zeros(1, 256, 4, 9, dtype=torch.float32, device=device)
    dummy_c = torch.zeros(1, 16, dtype=torch.float32, device=device)
    dummy_seq = torch.zeros(1, 72, dtype=torch.int64, device=device)

    traced_rl = torch.jit.trace(rl_wrapper, (dummy_s2d, dummy_c, dummy_seq), check_trace=False)
    traced_sl = torch.jit.trace(sl_wrapper, (dummy_s2d, dummy_c, dummy_seq), check_trace=False)

    opp_models = [traced_sl for _ in range(3)]

    ranks, net_points = [], []
    win_count, deal_in_count, kyoku_count = 0, 0, 0

    state_dict, mask, current_player = env.reset_hanchan()
    completed_hanchan = 0

    while completed_hanchan < num_hanchan:
        s_2d = torch.tensor(state_dict["state_2d"], dtype=torch.float32, device=device).unsqueeze(0)
        c_vec = torch.tensor(state_dict["cond_vec"], dtype=torch.float32, device=device).unsqueeze(0)
        seq_h = torch.tensor(state_dict["seq_hist"], dtype=torch.int64, device=device).unsqueeze(0)
        t_mask = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            if current_player == 0:
                p_out, _ = traced_rl(s_2d, c_vec, seq_h)
            else:
                p_out, _ = opp_models[current_player - 1](s_2d, c_vec, seq_h)

        masked_logits = p_out + (1.0 - t_mask) * -1e9
        action_val = torch.argmax(masked_logits, dim=-1).item()

        next_state_dict, next_mask, _step_reward, done, next_player, info = env.step(action_val)

        if info["p0_win"]:
            win_count += 1
        if info["p0_deal_in"]:
            deal_in_count += 1
        if info["hand_done"]:
            kyoku_count += 1

        if done:
            my_score = env.scores[0]
            rank = sum(1 for x in env.scores if x > my_score) + 1
            ranks.append(rank)
            net_points.append(my_score - 25000)

            completed_hanchan += 1
            next_state_dict, next_mask, next_player = env.reset_hanchan()

        state_dict, mask, current_player = next_state_dict, next_mask, next_player

    result_queue.put(
        {
            "ranks": ranks,
            "nets": net_points,
            "win_count": win_count,
            "deal_in_count": deal_in_count,
            "kyoku_count": kyoku_count,
        }
    )


def parallel_evaluate_against_sl(rl_model, sl_model, total_hanchan=2500, num_eval_workers=10):
    print(f"\n[Eval] 並列評価モジュールを起動: 総目標 {total_hanchan} 半荘 ({num_eval_workers} プロセスで分散実行)")

    rl_sd = {k: v.cpu() for k, v in rl_model.state_dict().items()}
    sl_sd = {k: v.cpu() for k, v in sl_model.state_dict().items()}

    hanchan_per_worker = total_hanchan // num_eval_workers
    result_queue = mp.Queue()
    workers = []

    for i in range(num_eval_workers):
        p = mp.Process(target=evaluation_worker, args=(i, rl_sd, sl_sd, hanchan_per_worker, result_queue))
        p.start()
        workers.append(p)

    eval_pbar = tqdm(total=num_eval_workers, desc="Aggregating Parallel Evals", leave=False)

    all_ranks, all_nets = [], []
    total_wins, total_deals, total_kyoku = 0, 0, 0

    for _ in range(num_eval_workers):
        res = result_queue.get()
        all_ranks.extend(res["ranks"])
        all_nets.extend(res["nets"])
        total_wins += res["win_count"]
        total_deals += res["deal_in_count"]
        total_kyoku += res["kyoku_count"]
        eval_pbar.update(1)

    eval_pbar.close()

    for p in workers:
        p.join()

    avg_rank = np.mean(all_ranks)
    avg_net = np.mean(all_nets)
    win_rate = total_wins / max(1, total_kyoku)
    deal_in_rate = total_deals / max(1, total_kyoku)

    return avg_rank, avg_net, win_rate, deal_in_rate


# ==========================================
# 5. マルチプロセス・ワーカー定義 (Worker Process)
# ==========================================
def sync_params(src_model, dst_model):
    for src_p, dst_p in zip(src_model.parameters(), dst_model.parameters()):
        dst_p.data.copy_(src_p.data)


def async_environment_worker(
    worker_id, request_queue, response_pipe, trajectory_queue, steps_to_collect, shared_phase,
    reward_config=None,
):
    env = MultiAgentMahjongEnvWrapper()
    reward_config = reward_config or RewardConfig()
    decomposition = RewardDecomposition()

    state_dict, mask, current_player = env.reset()
    pending_transition = None
    accumulated_reward = 0.0

    # ローカルメトリクスの初期化
    local_metrics = {"hand_count": 0, "win_count": 0, "deal_in_count": 0, "shanten_reduction": 0}

    while True:
        request_queue.put({
            "worker_id": worker_id,
            # 【修正】以前は astype(np.int8) で量子化していたが、state_2d の
            # チャネル 211/212 (風) と 216-219 (点差) は |値| < 1 の小数のため
            # 全て 0 に潰れていた。サンプリング時と更新時で状態が食い違い、
            # PPO の重要度比が第0エポックから壊れる原因になっていたので float16 に変更。
            # (36KB -> 18KB。値域は [-1, 4] 程度なので float16 の精度で十分)
            "state_2d": state_dict["state_2d"].astype(np.float16),
            "cond_vec": state_dict["cond_vec"].astype(np.float16),
            "seq_hist": state_dict["seq_hist"].astype(np.int16),
            "mask": mask.astype(np.int8)
        })

        response = response_pipe.recv()
        action_val = response["action"]

        if current_player == 0:
            log_prob_val = response["log_prob"]
            value_val = response["value"]

            # [修正] アクション実行「前」に、前回の打牌以降に発生した向聴進速を抽出し、
            # 前回の pending_transition に還元する (Credit Assignment Fix)
            current_phase = shared_phase.value
            shaping_reward = decomposition.add(
                "shanten", env._pending_shanten_reduction * reward_config.shanten_scale(current_phase)
            )

            if pending_transition is not None:
                accumulated_reward += shaping_reward
                local_metrics["shanten_reduction"] += env._pending_shanten_reduction

                pending_transition["reward"] = accumulated_reward
                pending_transition["done"] = False
                # 【修正】GAE のブートストラップ用に、この遷移の「次状態」の価値を厳密に添付する。
                # 以前は軌跡末端で values[-1] (= V(s_t) 自身) を V(s_{t+1}) として流用していた。
                # ここでの value_val はまさに次の p0 決定局面 = s_{t+1} の価値なので厳密。
                pending_transition["next_value"] = value_val
                trajectory_queue.put(pending_transition)

            accumulated_reward = 0.0
            env._pending_shanten_reduction = 0

            # [修正] 受入れ (Ukeire) シェーピング。旧実装は phase ゲートが無く、
            # 「純粋 RL」であるはずの Phase 3 でも効き続けていた。
            ukeire_scale = reward_config.ukeire_scale(current_phase)
            if ukeire_scale and action_val < 34:
                obs_93 = env.env.get_obs(0)
                tiles34 = obs_93[0:4].sum(axis=0).astype(np.int32)
                count = ukeire_count(env.shanten_calc, tiles34, action_val)
                accumulated_reward += decomposition.add("ukeire", count * ukeire_scale)

            pending_transition = {
                "worker_id": worker_id,
                "state_2d": state_dict["state_2d"],
                "cond_vec": state_dict["cond_vec"],
                "seq_hist": state_dict["seq_hist"],
                "action": action_val,
                "mask": mask,
                "log_prob": log_prob_val,
                "value": value_val,
            }

        next_state_dict, next_mask, _env_reward, done, next_player, info = env.step(action_val)

        # 【変更】報酬の計算は環境から reward.py へ移した。
        # 環境は「何が起きたか」だけを返し、それをどう評価するかはここで一元管理する。
        step_reward = 0.0
        if info["hand_done"]:
            step_reward += decomposition.add(
                "payoff", info["hand_payoff_p0"] * reward_config.hand_payoff_scale
            )
        if done and "final_scores" in info:
            bonus, _rank = hanchan_rank_bonus(reward_config, info["final_scores"])
            step_reward += decomposition.add("rank", bonus)

        if info["p0_win"]:
            local_metrics["win_count"] += 1
        if info["p0_deal_in"]:
            local_metrics["deal_in_count"] += 1
        if info["hand_done"]:
            local_metrics["hand_count"] += 1

        if pending_transition is not None:
            accumulated_reward += float(step_reward)

        if done:
            if pending_transition is not None:
                pending_transition["reward"] = accumulated_reward
                pending_transition["done"] = True
                pending_transition["next_value"] = 0.0  # 終端状態の価値は 0
                pending_transition["metrics"] = local_metrics.copy()  # 半荘完了時にメトリクスを送信
                pending_transition["reward_parts"] = decomposition.snapshot()
                trajectory_queue.put(pending_transition)

            accumulated_reward = 0.0
            pending_transition = None
            local_metrics = {"hand_count": 0, "win_count": 0, "deal_in_count": 0, "shanten_reduction": 0}
            decomposition.reset()
            next_state_dict, next_mask, next_player = env.reset()

        state_dict, mask, current_player = next_state_dict, next_mask, next_player


# ==========================================
# 6. PPO エンジン (動的学習率 & 分離バッファ対応)
# ==========================================
class HeroReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def add(self, s_2d, c_vec, seq_h, action, mask):
        self.buffer.append((s_2d, c_vec, seq_h, action, mask))

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)

        s_2d = [item[0] for item in batch]
        c_vec = [item[1] for item in batch]
        seq_h = [item[2] for item in batch]
        actions = [item[3] for item in batch]
        masks = [item[4] for item in batch]

        return s_2d, c_vec, seq_h, actions, masks


class PPOBuffer:
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.clear()

    def clear(self):
        self.trajectories = {
            i: {
                "states_2d": [],
                "cond_vecs": [],
                "seq_hists": [],
                "actions": [],
                "masks": [],
                "log_probs": [],
                "rewards": [],
                "state_values": [],
                "next_values": [],
                "dones": [],
            }
            for i in range(self.num_workers)
        }

    def add(self, worker_id, step_data):
        traj = self.trajectories[worker_id]
        traj["states_2d"].append(step_data["state_2d"])
        traj["cond_vecs"].append(step_data["cond_vec"])
        traj["seq_hists"].append(step_data["seq_hist"])
        traj["actions"].append(step_data["action"])
        traj["masks"].append(step_data["mask"])
        traj["log_probs"].append(step_data["log_prob"])
        traj["rewards"].append(step_data["reward"])
        traj["state_values"].append(step_data["value"])
        traj["next_values"].append(step_data["next_value"])
        traj["dones"].append(step_data["done"])


class PPOKLPenaltyTrainer:
    def __init__(self, model, sl_model, device, num_workers, lr=1.5e-4, kl_beta=0.01, ppo_epochs=4):
        self.device = device
        self.model = model.to(self.device)
        self.sl_model = sl_model.to(self.device)
        self.sl_model.eval()
        self.optimizer = DirectMLSafeAdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)
        self.kl_beta = kl_beta
        self.clip_eps = 0.2
        self.ppo_epochs = ppo_epochs
        # 【修正】0.99 では p0 の 1 半荘 150+ ステップに対して 0.99^150 ≈ 0.22 まで減衰し、
        # 順位ボーナスが序盤にほとんど伝わらなかった。終局の順位が目的関数なので 1.0 に近づける。
        self.gamma = 1.0
        self.gae_lambda = 0.95
        self.buffer = PPOBuffer(num_workers)
        self.hero_buffer = HeroReplayBuffer(max_size=10000)

        # 【追加】KL 錨定を信頼できる次元だけに限定するマスク。
        # 旧 SL データセットには reach / hora / pass サンプルが無いため、
        # 48..53 の SL 事前分布はゼロ同然であり、KL はそこへ策略を引き戻してしまう
        # (= 和了アクションを永久に殺す)。該当次元を除いた条件付き KL を用いる。
        self.kl_trust_mask = torch.ones(A.N_ACTIONS, device=self.device)
        for i in A.DECLARATION_ACTIONS:
            self.kl_trust_mask[i] = 0.0

        # 診断用: 直近の更新における第0エポックの重要度比 (1.0 から離れていたら
        # サンプリング側と更新側で状態/モードが食い違っている)
        self.last_first_epoch_ratio = float("nan")

    def set_learning_rate(self, new_lr):
        for g in self.optimizer.param_groups:
            g["lr"] = new_lr

    def update_from_buffer(self, current_phase, mini_batch_size=256):
        all_s_2d, all_c_vec, all_seq_h = [], [], []
        all_actions, all_masks, all_old_log_probs = [], [], []
        all_advantages, all_returns, all_old_values = [], [], []

        total_steps = 0

        for traj in self.buffer.trajectories.values():
            rewards = traj["rewards"]
            if len(rewards) == 0:
                continue

            values = traj["state_values"]
            next_values = traj["next_values"]
            dones = traj["dones"]
            T = len(rewards)
            total_steps += T

            # 【修正】各遷移が自前で厳密な V(s_{t+1}) を持つ (worker 側で添付)。
            # 以前は軌跡末端で V(s_t) を V(s_{t+1}) として流用していた。
            w_advantages = np.zeros(T, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(T)):
                delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                w_advantages[t] = gae

            all_advantages.extend(w_advantages)
            all_returns.extend(w_advantages + np.array(values))

            all_s_2d.extend(traj["states_2d"])
            all_c_vec.extend(traj["cond_vecs"])
            all_seq_h.extend(traj["seq_hists"])
            all_actions.extend(traj["actions"])
            all_masks.extend(traj["masks"])
            all_old_log_probs.extend(traj["log_probs"])
            all_old_values.extend(values)

        if total_steps == 0:
            return 0.0, 0.0

        # 【新增】将高质量操作存入 Hero Buffer (SIL)
        # 仅在 Phase 1/2 生效。避免在 Phase 3 陷入局部最优（见逃等高级战术需要自由探索）。
        if current_phase < 3:
            for i in range(len(all_returns)):
                if all_returns[i] > 0.1:
                    self.hero_buffer.add(
                        all_s_2d[i], all_c_vec[i], all_seq_h[i], all_actions[i], all_masks[i]
                    )

        # 【修正】更新も eval モードで行う。
        # このネットワークは 18 個の Dropout2d(p=0.30) を持つため、更新時だけ train() にすると
        # サンプリング時の方策と更新時の方策が別物になり、PPO の重要度比が
        # 第0エポックから 1.0 にならない (実測 ratio ≈ 100)。
        # eval() は Dropout と BatchNorm の挙動を変えるだけで勾配は普通に流れる。
        # BatchNorm の移動統計も SL 由来のものを保ち、自己対局のミニバッチで汚さない。
        self.model.eval()

        s_2d = torch.tensor(np.array(all_s_2d), dtype=torch.float32, device=self.device)
        c_vec = torch.tensor(np.array(all_c_vec), dtype=torch.float32, device=self.device)
        seq_h = torch.tensor(np.array(all_seq_h), dtype=torch.int64, device=self.device)
        actions = torch.tensor(all_actions, dtype=torch.int64, device=self.device)
        masks = torch.tensor(np.array(all_masks), dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(all_old_log_probs, dtype=torch.float32, device=self.device)

        advantages = torch.tensor(all_advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(all_returns, dtype=torch.float32, device=self.device)
        old_values_tensor = torch.tensor(all_old_values, dtype=torch.float32, device=self.device)

        adv_mean = advantages.mean()
        adv_std = torch.sqrt(torch.mean((advantages - adv_mean) ** 2) + 1e-8)
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        classes = torch.arange(54, device=self.device).unsqueeze(0)
        one_hot_actions = (actions.unsqueeze(1) == classes).to(torch.float32)

        total_ppo_loss = 0.0
        total_entropy_val = 0.0
        batch_size = total_steps
        indices = np.arange(batch_size)
        num_updates = 0

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, batch_size, mini_batch_size):
                end_idx = start_idx + mini_batch_size
                mb_indices = indices[start_idx:end_idx]

                mb_s_2d = s_2d[mb_indices]
                mb_c_vec = c_vec[mb_indices]
                mb_seq_h = seq_h[mb_indices]
                mb_one_hot_actions = one_hot_actions[mb_indices]
                mb_masks = masks[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_values = old_values_tensor[mb_indices]

                p_out, v_score, aux_t, aux_d, aux_w = self.model(mb_s_2d, mb_c_vec, mb_seq_h, rl_mode=True)
                new_values = v_score.squeeze(-1)
                log_probs_all, p_out_masked = masked_log_probs(p_out, mb_masks)
                new_probs = F.softmax(p_out_masked, dim=-1)

                with torch.no_grad():
                    sl_out, _, sl_t, sl_d, sl_w = self.sl_model(mb_s_2d, mb_c_vec, mb_seq_h, rl_mode=False)

                    if mb_masks.size(-1) == 34:
                        full_mask = torch.zeros(mb_masks.size(0), 54, device=self.device)
                        full_mask[:, :34] = mb_masks
                        full_mask[:, 34:] = 1.0
                    else:
                        full_mask = mb_masks

                    sl_probs = F.softmax(sl_out + (1.0 - full_mask) * -1e9, dim=-1)
                    sl_t_target = torch.sigmoid(sl_t)
                    sl_d_target = torch.sigmoid(sl_d)
                    sl_w_target = torch.sigmoid(sl_w)

                new_log_probs = (log_probs_all * mb_one_hot_actions).sum(dim=-1)
                entropy = -(new_probs * log_probs_all).sum(dim=-1).mean()

                log_diff = torch.clamp(new_log_probs - mb_old_log_probs, -5.0, 5.0)
                ratios = torch.exp(log_diff)
                if num_updates == 0:
                    # 第0エポック・第0ミニバッチでは方策が未更新なので比は 1.0 のはず。
                    # ここが 1 から離れる = サンプリングと更新で状態表現か train/eval モードが違う。
                    self.last_first_epoch_ratio = float(ratios.mean().item())
                ratios_bounded = torch.clamp(ratios, 0.0, 3.0)

                surr1 = ratios_bounded * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                v_clipped = mb_old_values + torch.clamp(new_values - mb_old_values, -self.clip_eps, self.clip_eps)
                vf_loss1 = F.smooth_l1_loss(new_values, mb_returns, reduction="none")
                vf_loss2 = F.smooth_l1_loss(v_clipped, mb_returns, reduction="none")
                value_loss = torch.max(vf_loss1, vf_loss2).mean()

                # 信頼できる次元のみに条件付けた KL (48..53 は SL 事前分布が死んでいるため除外)
                trust = self.kl_trust_mask.unsqueeze(0) * mb_masks
                sl_p_trusted = sl_probs * trust
                new_p_trusted = new_probs * trust
                sl_p_trusted = sl_p_trusted / sl_p_trusted.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                new_p_trusted = new_p_trusted / new_p_trusted.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                kl_div = (
                    sl_p_trusted
                    * (
                        torch.log(torch.clamp(sl_p_trusted, min=1e-8))
                        - torch.log(torch.clamp(new_p_trusted, min=1e-8))
                    )
                ).sum(dim=-1).mean()

                loss_aux_t = directml_safe_bce_with_logits(aux_t, sl_t_target)
                loss_aux_d = directml_safe_bce_with_logits(aux_d, sl_d_target)
                loss_aux_w = directml_safe_bce_with_logits(aux_w, sl_w_target)
                aux_loss = 0.05 * (loss_aux_t + loss_aux_d + loss_aux_w)

                # 【新增】计算自我模仿学习损失 (SIL Loss)
                sil_loss_val = 0.0
                if current_phase < 3:
                    hero_batch = self.hero_buffer.sample(128)
                    if hero_batch is not None:
                        h_s2d, h_cvec, h_seqh, h_actions, h_masks = hero_batch
                        h_s2d_t = torch.tensor(np.array(h_s2d), dtype=torch.float32, device=self.device)
                        h_cvec_t = torch.tensor(np.array(h_cvec), dtype=torch.float32, device=self.device)
                        h_seqh_t = torch.tensor(np.array(h_seqh), dtype=torch.int64, device=self.device)
                        h_acts_t = torch.tensor(h_actions, dtype=torch.int64, device=self.device)
                        h_masks_t = torch.tensor(np.array(h_masks), dtype=torch.float32, device=self.device)

                        h_p_out, _, _, _, _ = self.model(h_s2d_t, h_cvec_t, h_seqh_t, rl_mode=True)
                        h_masked_logits = h_p_out + (1.0 - h_masks_t) * -1e9
                        sil_loss_val = F.cross_entropy(h_masked_logits, h_acts_t)

                total_loss = policy_loss + 1.0 * value_loss - 0.01 * entropy + self.kl_beta * kl_div + aux_loss + 0.1 * sil_loss_val

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_ppo_loss += total_loss.item()
                total_entropy_val += entropy.item()
                num_updates += 1

        self.buffer.clear()
        # 【修正】サンプリング (中央バッチ推論) は必ず eval モードで行う。
        # 以前は train() のままだったため 18 個の Dropout2d(p=0.30) が有効になり、
        # BatchNorm も 1~30 サンプルのミニバッチ統計を使っていた。
        self.model.eval()
        if num_updates > 0:
            return total_ppo_loss / num_updates, total_entropy_val / num_updates
        return 0.0, 0.0


# ==========================================
# 7. メイン実行スクリプト (動的フェーズ統合 & 安全保存対応版)
# ==========================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    print("=" * 60)
    print("🚀 PPO マルチエージェント自己対局 (動的フェーズ統合 & 安全なチェックポイント保存)")
    print("=" * 60)

    NUM_WORKERS = 30
    STEPS_PER_WORKER = 256

    # --- 学習制御のしきい値 (旧: 関数内に直書き) ---------------------------
    # プラトー判定は「直近 N 回の評価順位の回帰直線」で行う。
    PLATEAU_WINDOW = 5
    PLATEAU_SLOPE_MAX = 0.02  # 順位の傾き (1 評価あたりの改善量)
    PLATEAU_RESID_MAX = 0.06  # 回帰残差の標準偏差
    # 昇格の黄金指標。RANK_TARGET を切れば無条件、そうでなければ素点条件つき
    RANK_TARGET = 2.40
    RANK_TARGET_SOFT = 2.45
    NET_TARGET_SOFT = -1200.0
    PHASE2_WIN_RATE_MIN = 0.05  # 流局聴牌罰符だけで昇格するのを防ぐ
    EVAL_INTERVAL = 500
    EVAL_HANCHAN = 2500

    reward_config = RewardConfig()
    print(f" -> [Reward] 報酬設定: {reward_config.as_dict()}")
    TARGET_BUFFER_SIZE = NUM_WORKERS * STEPS_PER_WORKER

    if torch_directml.is_available():
        device = torch_directml.device()
    else:
        device = torch.device("cpu")

    model = SmartMahjongMultiTaskNet(input_channels=256, num_blocks=18).to(device)
    sl_base_model = SmartMahjongMultiTaskNet(input_channels=256, num_blocks=18).to(device)

    base_policy_path = "smart_mahjong_base_policy_v2.pth"

    # 【追加】死んだアクションヘッド (立直/ロン/ツモ/九種/パス) の修復スイッチ。
    # 旧データセットで学習された重みからリスタートする場合は必ず 1 にする。
    # 阶段2 で正しいラベルの SL を再学習したら 0 に戻すこと。
    REPAIR_HEAD = os.environ.get("MJ_REPAIR_HEAD", "1") == "1"
    CALL_LOGIT_PENALTY = float(os.environ.get("MJ_CALL_PENALTY", DEFAULT_CALL_LOGIT_PENALTY))

    loaded_payloads: dict[str, object] = {}

    def load_policy(path, repair=REPAIR_HEAD):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        loaded_payloads[path] = payload
        sd = adapt_policy_state_dict(extract_model_state(payload))
        if repair:
            sd = repair_policy_head(sd, call_logit_penalty=CALL_LOGIT_PENALTY)
        return sd

    resume_path = None
    current_phase = 1

    for p in [3, 2, 1]:
        latest_files = glob.glob(f"smart_mahjong_ppo_latest_phase{p}_iter*.pth")
        latest_files.sort(key=os.path.getmtime, reverse=True)

        candidates = []
        if latest_files:
            candidates.append(latest_files[0])

        candidates.extend([f"smart_mahjong_ppo_TRUE_BEST_phase{p}.pth", f"smart_mahjong_ppo_best_phase{p}.pth"])

        found_path = next((path for path in candidates if os.path.exists(path)), None)
        if found_path:
            resume_path = found_path
            current_phase = p
            break

    if resume_path:
        model.load_state_dict(load_policy(resume_path))
        print(
            f" -> [Info] レジューム（断点続行）: Phase {current_phase} の履歴重み {resume_path} の検出・読み込みに成功しました"
        )

        if current_phase > 1:
            prev_phase = current_phase - 1
            sl_candidates = [
                f"smart_mahjong_ppo_TRUE_BEST_phase{prev_phase}.pth",
                f"smart_mahjong_ppo_latest_phase{prev_phase}.pth",
            ]
            sl_resume_path = next((path for path in sl_candidates if os.path.exists(path)), base_policy_path)
            if os.path.exists(sl_resume_path):
                sl_base_model.load_state_dict(load_policy(sl_resume_path))
                print(f" -> [Info] 動的SLベースライン: Phase {prev_phase} のモデルで初期化しました ({sl_resume_path})")
        else:
            if os.path.exists(base_policy_path):
                sl_base_model.load_state_dict(load_policy(base_policy_path))
                print(f" -> [Info] SLベースポリシーを読み込みました: {base_policy_path}")
    else:
        if os.path.exists(base_policy_path):
            model.load_state_dict(load_policy(base_policy_path))
            sl_base_model.load_state_dict(load_policy(base_policy_path))
            print(
                " -> [Info] 利用可能なチェックポイントが検出されませんでした。SLベースポリシーから全く新しい Phase 1 の学習を開始します"
            )

    # 【新增】共享内存 Phase，用于 Worker 节点动态提取塑形权重
    shared_phase = mp.Value("i", current_phase)

    phase_lr_map = {1: 1.5e-4, 2: 3e-5, 3: 1e-5}
    phase_kl_map = {1: 0.01, 2: 0.03, 3: 0.05}
    current_lr = phase_lr_map[current_phase]
    current_kl = phase_kl_map[current_phase]
    print(f" -> [Info] 現在のシステム設定: Phase = {current_phase}, 学習率 = {current_lr}, KLペナルティ = {current_kl}")

    trainer = PPOKLPenaltyTrainer(
        model, sl_base_model, device, num_workers=NUM_WORKERS, lr=current_lr, kl_beta=current_kl
    )
    # 【修正】サンプリングは常に eval モード (Dropout 無効・BN は移動平均) で行う
    trainer.model.eval()
    ckpt_manager = CheckpointManager(max_keep=3)
    logger = TrainingLogger()

    dead_now, _, _ = diagnose_policy_head(trainer.model.state_dict())
    print(
        f" -> [Head Check] 起動時の死行: {[A.ACTION_NAMES[i] for i in dead_now] if dead_now else 'なし'}"
    )

    trajectory_queue: mp.Queue = mp.Queue(maxsize=NUM_WORKERS * 4 * STEPS_PER_WORKER)

    # 【新增】建立集中式批量推理 IPC 通道
    request_queue: mp.Queue = mp.Queue()
    parent_pipes = []
    child_pipes = []
    for _ in range(NUM_WORKERS):
        p, c = mp.Pipe()
        parent_pipes.append(p)
        child_pipes.append(c)

    workers = []
    for i in range(NUM_WORKERS):
        proc = mp.Process(
            target=async_environment_worker,
            args=(
                i, request_queue, child_pipes[i], trajectory_queue, STEPS_PER_WORKER,
                shared_phase, reward_config,
            ),
        )
        proc.start()
        workers.append(proc)

    reward_history_window = []
    eval_rank_history = []

    ppo_loss_history = []
    reward_history = []
    entropy_history = []

    best_eval_rank = float("inf")
    best_eval_net = -float("inf")

    it = 0
    if resume_path:
        restored_it = restore_training_state(loaded_payloads.get(resume_path), trainer)
        if restored_it is not None:
            it = int(restored_it)
            print(
                f" -> [Info] 学習状態を復元しました (optimizer / kl_beta / RNG)。"
                f"第 {it} イテレーションから再開します"
            )
        else:
            match = re.search(r"_iter(\d+)\.pth", resume_path)
            if match:
                it = int(match.group(1))
                print(f" -> [Info] イテレーションカウントの復元: 第 {it} イテレーションから学習を再開します")
            else:
                print(
                    " -> [Warn] 旧形式のチェックポイントです。イテレーション番号が復元できないため、"
                    "ログの切り詰めをスキップします"
                )
                it = -1

    if it >= 0:
        logger.truncate_after(it)
    else:
        it = 0

    try:
        while True:
            it += 1
            iteration_reward = 0.0
            added_steps = 0

            # [追加] 日常追跡指標
            total_hands, total_wins, total_deal_ins, total_shanten_reduction = 0, 0, 0, 0
            reward_parts_sum = dict.fromkeys(COMPONENTS, 0.0)
            finished_hanchan = 0

            import queue

            rollout_pbar = tqdm(
                total=TARGET_BUFFER_SIZE, desc=f"Iter [{it}] Phase {current_phase} Async Rollout", leave=False
            )
            while added_steps < TARGET_BUFFER_SIZE:
                # 1. Workerからの完全な軌跡データを収集（ノンブロッキング）
                while not trajectory_queue.empty():
                    step_data = trajectory_queue.get()
                    trainer.buffer.add(step_data["worker_id"], step_data)
                    iteration_reward += step_data["reward"]
                    added_steps += 1
                    rollout_pbar.update(1)

                    if step_data["done"] and "metrics" in step_data:
                        m = step_data["metrics"]
                        total_hands += m["hand_count"]
                        total_wins += m["win_count"]
                        total_deal_ins += m["deal_in_count"]
                        total_shanten_reduction += m["shanten_reduction"]
                        finished_hanchan += 1
                        for name, value in step_data.get("reward_parts", {}).items():
                            reward_parts_sum[name] = reward_parts_sum.get(name, 0.0) + value

                    if added_steps >= TARGET_BUFFER_SIZE:
                        break

                if added_steps >= TARGET_BUFFER_SIZE:
                    break

                # 2. 集中型GPUバッチ推論 (Central Batch Inference)
                requests = []
                try:
                    # 最大0.01秒ブロックして、少なくとも1つのリクエストを捕捉
                    req = request_queue.get(timeout=0.01)
                    requests.append(req)
                    # [追加] マイクロバッチ遅延 (Micro-Batching Delay): 強制的に0.002秒余分に待機し、大きなバッチを構成してGPUを最大限に活用する
                    while len(requests) < NUM_WORKERS:
                        try:
                            requests.append(request_queue.get(timeout=0.002))
                        except queue.Empty:
                            break
                except queue.Empty:
                    pass

                if requests:
                    # バッチの結合
                    b_s2d = torch.tensor(np.array([r["state_2d"] for r in requests]), dtype=torch.float32, device=device)
                    b_cvec = torch.tensor(np.array([r["cond_vec"] for r in requests]), dtype=torch.float32, device=device)
                    b_seqh = torch.tensor(np.array([r["seq_hist"] for r in requests]), dtype=torch.int64, device=device)
                    b_mask = torch.tensor(np.array([r["mask"] for r in requests]), dtype=torch.float32, device=device)

                    # GPUフォワードパス
                    with torch.no_grad():
                        p_out, v_score, _, _, _ = trainer.model(b_s2d, b_cvec, b_seqh, rl_mode=True)
                        # 【重大な修正】torch.distributions.Categorical は DirectML 上で
                        # マスクした非合法アクションを返すことがある (log_prob が
                        # log(float32 eps) = -15.94 に張り付く)。その結果ロールアウトの
                        # 行動は方策からのサンプルではなくなり、PPO の重要度比は
                        # 第0エポックからクリップ上限に飽和していた。
                        actions, log_probs = directml_safe_masked_sample(p_out, b_mask)

                    actions_np = actions.cpu().numpy()
                    v_score_np = v_score.squeeze(-1).cpu().numpy()
                    log_probs_np = log_probs.cpu().numpy()

                    # Pipe経由で結果を対応するWorkerに正確に分配
                    for idx, r in enumerate(requests):
                        w_id = r["worker_id"]
                        parent_pipes[w_id].send({
                            "action": int(actions_np[idx]),
                            "value": float(v_score_np[idx]),
                            "log_prob": float(log_probs_np[idx])
                        })

            rollout_pbar.close()

            update_pbar = tqdm(total=trainer.ppo_epochs, desc=f"Iter [{it}] Phase {current_phase} Optim  ", leave=False)
            ppo_loss, avg_entropy = trainer.update_from_buffer(current_phase, mini_batch_size=512)
            update_pbar.update(trainer.ppo_epochs)
            update_pbar.close()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            avg_reward = iteration_reward / TARGET_BUFFER_SIZE
            ppo_loss_history.append(ppo_loss)
            reward_history.append(avg_reward)
            entropy_history.append(avg_entropy)

            reward_history_window.append(avg_reward)
            if len(reward_history_window) > 500:
                reward_history_window.pop(0)

            # 計算日常監控指標 (Calculate daily tracking metrics)
            win_rate = total_wins / max(1, total_hands)
            deal_in_rate = total_deal_ins / max(1, total_hands)
            mean_shanten_red = total_shanten_reduction / max(1, total_hands)

            print(
                f"✅ Iter [{it:04d}] Phase {current_phase} | Loss: {ppo_loss:.4f} | R: {avg_reward:.4f} | Ent: {avg_entropy:.4f} | Win: {win_rate:.2%} | Deal-in: {deal_in_rate:.2%} | ΔShanten: +{mean_shanten_red:.2f} | ratio0: {trainer.last_first_epoch_ratio:.4f}"
            )
            if abs(trainer.last_first_epoch_ratio - 1.0) > 0.02:
                print(
                    f"⚠️ [Sanity] 第0エポックの重要度比が {trainer.last_first_epoch_ratio:.4f} です "
                    "(期待値 1.00±0.02)。サンプリングと更新で状態表現か train/eval モードが食い違っています。"
                )
            per_hanchan = max(1, finished_hanchan)
            parts = {k: v / per_hanchan for k, v in reward_parts_sum.items()}
            print(
                "   [Reward] " + "  ".join(f"{k}={v:+.4f}" for k, v in parts.items())
                + f"  (半荘 {finished_hanchan} 局分の平均)"
            )
            logger.log_train(
                it, current_phase, ppo_loss, avg_reward, avg_entropy, win_rate,
                deal_in_rate, mean_shanten_red, parts,
            )

            # ==========================================
            # 評価とフェーズ移行ロジック
            # ==========================================
            if it % EVAL_INTERVAL == 0:
                print(f"\n--- [Eval] Iter {it}: 500輪定期評価を実行します ---")

                avg_rank, avg_net, win_r, deal_r = parallel_evaluate_against_sl(
                    trainer.model, sl_base_model, total_hanchan=EVAL_HANCHAN, num_eval_workers=10
                )

                eval_rank_history.append(avg_rank)

                print(
                    f"📊 [Eval Result] 平均順位: {avg_rank:.3f} | 半荘均浄勝素点: {avg_net:.1f} pt | 和了率: {win_r:.3%} | 放銃率: {deal_r:.3%}"
                )
                logger.log_eval(it, current_phase, avg_rank, avg_net, win_r, deal_r)

                if avg_rank < best_eval_rank or (avg_rank == best_eval_rank and avg_net > best_eval_net):
                    best_eval_rank = avg_rank
                    best_eval_net = avg_net
                    true_best_path = f"smart_mahjong_ppo_TRUE_BEST_phase{current_phase}.pth"
                    ckpt_manager.safe_save(
                        build_checkpoint(
                            trainer.model, trainer, current_phase, it, reward_config,
                            extra={"eval": {"rank": avg_rank, "net": avg_net,
                                            "win_rate": win_r, "deal_in_rate": deal_r}},
                        ),
                        true_best_path,
                    )
                    print(
                        f"     🏆 [True Best] 対SL評価の最高記録更新！(Rank: {avg_rank:.3f}, Net: {avg_net:.1f}) -> {true_best_path}"
                    )

                if it >= 1000:
                    # 【修正】旧判定は avg_reward の変動係数 std/|mean| を使っていたが、
                    # 自己対局では avg_reward が 0 付近を揺れるため分母が 0 に近づき、
                    # CV が発散して「プラトー」条件がほぼ永久に成立しなかった。
                    # 評価順位そのものの回帰直線 (傾き + 残差) で判定する。
                    k_val, resid_std = 1.0, float("inf")
                    if len(eval_rank_history) >= PLATEAU_WINDOW:
                        y = np.array(eval_rank_history[-PLATEAU_WINDOW:], dtype=np.float64)
                        x = np.arange(len(y), dtype=np.float64)
                        k_val, b_val = np.polyfit(x, y, 1)
                        resid_std = float(np.std(y - (k_val * x + b_val)))

                    is_plateau = abs(k_val) < PLATEAU_SLOPE_MAX and resid_std < PLATEAU_RESID_MAX
                    # 基线超越判定：以平均顺位（Avg Rank <= 2.40）为核心黄金指标，免除微小负素点死锁
                    exceed_sl = avg_rank <= RANK_TARGET or (
                        avg_rank <= RANK_TARGET_SOFT and avg_net > NET_TARGET_SOFT
                    )

                    print(
                        f"🔍 [Plateau Check] 順位の傾き |k|={abs(k_val):.4f} (<{PLATEAU_SLOPE_MAX}) | "
                        f"残差 σ={resid_std:.4f} (<{PLATEAU_RESID_MAX}) | "
                        f"直近 {PLATEAU_WINDOW} 回評価 | Rank={avg_rank:.3f} (<={RANK_TARGET})"
                    )

                    if is_plateau:
                        if current_phase == 1:
                            if exceed_sl:
                                print(
                                    "\n🌟 [Phase Transition] SLモデルを超越、かつプラトーに到達。Phase 2 (微調整) に移行します！"
                                )
                                current_phase = 2
                                shared_phase.value = current_phase  # 【更新】通知 Worker 更新权重
                                trainer.set_learning_rate(3e-5)
                                trainer.kl_beta = 0.03
                                best_eval_rank, best_eval_net = float("inf"), -float("inf")
                                eval_rank_history.clear()

                                print(
                                    "🔄 [Model Update] 評価用SLベースラインを Phase 1 の卒業モデルに動的更新します..."
                                )
                                sl_base_model.load_state_dict(trainer.model.state_dict())

                            else:
                                print(
                                    "\n⚠️ [Warning] SLモデル未超過でプラトーに到達（局所最適に陥落）。Phase 2への移行をブロックします！"
                                )
                                new_kl = max(0.005, trainer.kl_beta * 0.5)
                                print(
                                    f" -> KLペナルティ(kl_beta)を {trainer.kl_beta:.3f} から {new_kl:.3f} へ下調し、探索を促進します。"
                                )
                                trainer.kl_beta = new_kl
                                eval_rank_history.clear()

                        elif current_phase == 2:
                            # [修正] Phase 2からPhase 3への昇格は「実際の和了」で証明する必要がある
                            # 「流局聴牌罰符 (No-Ten Bappu)」だけでポイントを稼ぎ昇格する可能性を排除。和了率 >= 5% を強制
                            phase2_exceed = exceed_sl and (win_r >= PHASE2_WIN_RATE_MIN)
                            if phase2_exceed:
                                print(
                                    f"\n🌟 [Phase Transition] Phase 2 基準達成 (Rank:{avg_rank:.3f}, WinR:{win_r:.2%})。最終段階 Phase 3 を開始します！"
                                )
                                current_phase = 3
                                shared_phase.value = current_phase  # 【更新】通知 Worker 更新权重
                                trainer.set_learning_rate(1e-5)
                                trainer.kl_beta = 0.05
                                best_eval_rank, best_eval_net = float("inf"), -float("inf")
                                eval_rank_history.clear()

                                print(
                                    "🔄 [Model Update] 評価用SLベースラインを Phase 2 の卒業モデルに動的更新します..."
                                )
                                sl_base_model.load_state_dict(trainer.model.state_dict())
                            else:
                                print(
                                    f"\n⚠️ [Warning] Phase 2 でプラトーに到達しましたが、昇格基準未達です (Rank:{avg_rank:.3f}, WinR:{win_r:.2%})。"
                                )
                                new_kl = max(0.003, trainer.kl_beta * 0.7)
                                print(
                                    f" -> KLペナルティ(kl_beta)を {trainer.kl_beta:.4f} から {new_kl:.4f} へ引き下げ、攻撃・和了への探索を促進します！"
                                )
                                trainer.kl_beta = new_kl
                                eval_rank_history.clear()

                        elif current_phase == 3:
                            # [追加] Phase 3 最終卒業条件
                            # 対戦相手は Phase 2 の卒業モデル。同レベルの対戦では Rank 2.5 が引き分け。
                            # Rank <= 2.45 かつ 純スコア > 0 を達成し、純粋なRLがUkeireヒューリスティックモデルを確実に超えたことを証明する必要がある。
                            phase3_success = (avg_rank <= RANK_TARGET_SOFT) and (avg_net > 0)

                            if phase3_success:
                                print(
                                    f"\n👑 [Grand Finale] Phase 3 完璧にクリア！(Rank:{avg_rank:.3f}, Net:{avg_net:.1f}pt)。"
                                )
                                print(" -> 純粋な強化学習が Phase 2 ヒューリスティックモデルを成功裏に超え、AIは極致に達しました！")
                                final_model_path = "smart_mahjong_ppo_final_phase3_MASTER.pth"
                                ckpt_manager.safe_save(
                                    build_checkpoint(
                                        trainer.model, trainer, current_phase, it, reward_config
                                    ),
                                    final_model_path,
                                )
                                print(f" -> 最終マスターモデルを保存しました: {final_model_path}")
                                break
                            else:
                                print(
                                    f"\n⚠️ [Warning] Phase 3 がプラトーに達しましたが、Phase 2 のベースラインに勝てませんでした (Rank:{avg_rank:.3f}, Net:{avg_net:.1f}pt)。"
                                )
                                new_kl = max(0.001, trainer.kl_beta * 0.5)
                                print(f" -> KLペナルティを {new_kl:.4f} に引き下げ、モデルの探索能力を解放し、凡庸を拒否して学習を続行します！")
                                trainer.kl_beta = new_kl
                                eval_rank_history.clear()

            if it % 10 == 0:
                ckpt_manager.save_with_rotation(
                    build_checkpoint(trainer.model, trainer, current_phase, it, reward_config),
                    "smart_mahjong_ppo_latest",
                    current_phase,
                    it,
                )

    except KeyboardInterrupt:
        print("\n[Warn] 訓練がユーザーによって中断されました。(Training interrupted by user.)")
    finally:
        print("-> [Info] ワーカープロセスを終了しています...")
        for proc in workers:
            proc.terminate()
            proc.join(timeout=2.0)
