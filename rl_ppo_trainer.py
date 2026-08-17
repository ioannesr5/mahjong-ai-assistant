import csv
import gc
import glob
import math
import os
import random
import re
from collections import deque

import numpy as np
import pymahjong  # type: ignore
import torch
import torch.nn.functional as F
import torch_directml
from mahjong.shanten import Shanten  # 【新增】外部向听数计算器 (外部シャンテン数計算器)
from torch import multiprocessing as mp
from torch import nn
from torch.optim.optimizer import Optimizer
from tqdm import tqdm


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
                    ["Iteration", "Phase", "Loss", "Reward", "Entropy", "WinRate", "DealInRate", "MeanShantenRed"]
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
                ["Iteration", "Phase", "Loss", "Reward", "Entropy", "WinRate", "DealInRate", "MeanShantenRed"],
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
                with open(path, "r", newline="", encoding="utf-8") as f:
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

    def log_train(self, it, phase, loss, reward, entropy, win_r, deal_r, shanten_red):
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
                ]
            )

    def log_eval(self, it, phase, rank, net, win_r, deal_r):
        with open(self.eval_log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([it, phase, f"{rank:.3f}", f"{net:.1f}", f"{win_r:.4f}", f"{deal_r:.4f}"])


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
class DirectMLSafeAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                step = state["step"]

                p.mul_(1 - group["lr"] * group["weight_decay"])
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad * grad, alpha=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = group["lr"] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss


# ==========================================
# 2. ネットワーク構成要素 (SmartMahjongMultiTaskNet V2)
# ==========================================
class FiLMResBlock2D(nn.Module):
    def __init__(self, channels, cond_dim, dropout_p=0.15, res_scale=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.film_gen = nn.Linear(cond_dim, channels * 2)
        self.res_scale = res_scale

    def forward(self, x, cond):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        film_params = self.film_gen(cond).view(x.size(0), -1, 1, 1)
        gamma, beta = film_params.chunk(2, dim=1)
        out = (1.0 + gamma) * out + beta
        return F.relu((out * self.res_scale) + residual)


class DirectMLSafeTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        B, T, C = src.size()
        qkv = self.qkv(src)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = self.dropout1(F.softmax(scores, dim=-1))
        attn_out = self.out_proj(torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, T, C))
        src = self.norm1(src + self.dropout2(attn_out))
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(src))))
        return self.norm2(src + self.dropout(ff_out))


class DiscardSequenceEncoder(nn.Module):
    def __init__(self, vocab_size=273, embed_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=272)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 100, embed_dim))
        self.layers = nn.ModuleList(
            [DirectMLSafeTransformerLayer(embed_dim, num_heads, embed_dim * 4, dropout=0.1) for _ in range(num_layers)]
        )

    def forward(self, x):
        seq_len = x.size(1)
        out = self.embedding(x) + self.pos_embedding[:, :seq_len, :]
        for layer in self.layers:
            out = layer(out)
        return out


class MahjongBeliefCrossAttention(nn.Module):
    def __init__(self, cnn_dim=1024, seq_dim=256, num_heads=8, dropout_p=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cnn_dim, kdim=seq_dim, vdim=seq_dim, num_heads=num_heads, dropout=dropout_p, batch_first=True
        )
        self.norm = nn.LayerNorm(cnn_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, cnn_query, seq_kv):
        q = cnn_query.unsqueeze(1)
        attn_out, _ = self.cross_attn(q, seq_kv, seq_kv)
        return self.norm(cnn_query + self.dropout(attn_out.squeeze(1)))


class SmartMahjongMultiTaskNet(nn.Module):
    def __init__(self, input_channels=256, cond_dim=16, seq_vocab=273, num_blocks=18, dropout_p=0.30):
        super().__init__()
        self.conv_init = nn.Conv2d(input_channels, 256, kernel_size=3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(256)
        self.res_blocks = nn.ModuleList(
            [FiLMResBlock2D(256, cond_dim, dropout_p, res_scale=0.1) for _ in range(num_blocks)]
        )

        self.cnn_proj = nn.Sequential(nn.Linear(256 * 4 * 9, 1024), nn.LayerNorm(1024), nn.ReLU(inplace=True))
        self.seq_encoder = DiscardSequenceEncoder(vocab_size=seq_vocab, embed_dim=256, num_heads=8, num_layers=4)
        self.cross_attention = MahjongBeliefCrossAttention(cnn_dim=1024, seq_dim=256, num_heads=8, dropout_p=dropout_p)
        self.fusion_fc = nn.Sequential(
            nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.ReLU(inplace=True), nn.Dropout(p=dropout_p)
        )

        self.policy_out = nn.Linear(1024, 54)
        self.value_head = nn.Linear(1024, 1)

        def build_aux_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Linear(256, out_dim)
            )

        self.aux_tenpai = build_aux_mlp(1024, 3)
        self.aux_danger = build_aux_mlp(1024, 102)
        self.aux_waits = build_aux_mlp(1024, 102)

    def forward(self, state_2d, cond_vec, seq_hist, rl_mode=False):
        out = F.relu(self.bn_init(self.conv_init(state_2d)))
        for block in self.res_blocks:
            out = block(out, cond_vec)
        out_flat = out.view(out.size(0), -1)

        cnn_query = self.cnn_proj(out_flat)
        seq_kv = self.seq_encoder(seq_hist)
        fused = self.cross_attention(cnn_query, seq_kv)
        hidden = self.fusion_fc(fused)

        p_out = self.policy_out(hidden).to(torch.float32)
        v_head = self.value_head(hidden).to(torch.float32)

        if rl_mode:
            hidden_aux = hidden.detach()
        else:
            hidden_aux = hidden

        aux_t = self.aux_tenpai(hidden_aux).to(torch.float32)
        aux_d = self.aux_danger(hidden_aux).to(torch.float32)
        aux_w = self.aux_waits(hidden_aux).to(torch.float32)

        return p_out, v_head, aux_t, aux_d, aux_w


def adapt_policy_state_dict(sd, target_actions=54):
    """
    47次元の旧チェックポイント重みを54次元の完全アクション空間に安全に適応・変換する
    (Safely adapts 47-dim legacy checkpoints to 54-dim full action space)
    """
    if "policy_out.weight" not in sd:
        return sd
    w = sd["policy_out.weight"]
    b = sd["policy_out.bias"]
    if w.shape[0] == target_actions:
        return sd
    if w.shape[0] == 47 and target_actions == 54:
        new_w = torch.zeros((54, w.shape[1]), dtype=w.dtype)
        new_b = torch.zeros(54, dtype=b.dtype)

        # 0..33: 通常打牌 (0..33)
        new_w[0:34] = w[0:34]
        new_b[0:34] = b[0:34]

        # 34..36: 赤宝牌打牌 (5m=4, 5p=13, 5s=22)
        new_w[34] = w[4]
        new_b[34] = b[4]
        new_w[35] = w[13]
        new_b[35] = b[13]
        new_w[36] = w[22]
        new_b[36] = b[22]

        # 37..42: チー (CHI, SLインデックス=34)
        for i in range(37, 43):
            new_w[i] = w[34]
            new_b[i] = b[34]

        # 43..44: ポン (PON, SLインデックス=37)
        for i in range(43, 45):
            new_w[i] = w[37]
            new_b[i] = b[37]

        # 45..47: カン (KAN, SLインデックス=39)
        for i in range(45, 48):
            new_w[i] = w[39]
            new_b[i] = b[39]

        # 48: リーチ (RIICHI, SLインデックス=41)
        new_w[48] = w[41]
        new_b[48] = b[41]

        # 49: ロン (RON, SL和了インデックス=42)
        new_w[49] = w[42]
        new_b[49] = b[42]

        # 50: ツモ (TSUMO, SL和了インデックス=42)
        new_w[50] = w[42]
        new_b[50] = b[42]

        # 51: 九種九牌 (PUSH, SLパスインデックス=45)
        new_w[51] = w[45]
        new_b[51] = b[45]

        # 52..53: リーチパス, 応答パス (PASS, SLパスインデックス=45)
        new_w[52] = w[45]
        new_b[52] = b[45]
        new_w[53] = w[45]
        new_b[53] = b[45]

        sd_copy = dict(sd)
        sd_copy["policy_out.weight"] = new_w
        sd_copy["policy_out.bias"] = new_b
        return sd_copy
    return sd


class PolicyInferenceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, state_2d, cond_vec, seq_hist):
        p_out, v_head, _, _, _ = self.model(state_2d, cond_vec, seq_hist, True)
        return p_out, v_head


# ==========================================
# 3. マルチエージェント自己対局環境 (監視 & 報酬シェーピング対応版)
# ==========================================
def decode_obs_93_to_256(obs_93: np.ndarray, self_scores: np.ndarray, p_id: int) -> np.ndarray:
    obs_93 = obs_93.astype(np.float32)
    state = np.zeros((256, 34), dtype=np.float32)
    state[0:4] = obs_93[0:4]
    state[4, 4] = obs_93[5, 4]
    state[5, 13] = obs_93[5, 13]
    state[6, 22] = obs_93[5, 22]

    for i in range(4):
        ob_base = 6 + i * 6
        st_base = 7 + i * 12
        state[st_base : st_base + 4] = obs_93[ob_base : ob_base + 4]
        is_dora = np.clip(obs_93[74] + obs_93[75] + obs_93[76] + obs_93[77], 0, 1)
        state[st_base + 10] = obs_93[ob_base] * is_dora
        state[st_base + 11] = obs_93[ob_base + 5]

    for i in range(4):
        ob_base = 30 + i * 10
        st_base = 55 + i * 24
        tegiri = obs_93[ob_base + 4 : ob_base + 8]
        state[st_base : st_base + 4] = tegiri
        total_disc = obs_93[ob_base : ob_base + 4]
        state[st_base + 4 : st_base + 8] = np.clip(total_disc - tegiri, 0, 1)
        state[st_base + 8] = obs_93[ob_base + 9]
        state[st_base + 9 : st_base + 13] = np.clip(total_disc - obs_93[ob_base + 9], 0, 1)

    state[151:155] = obs_93[70:74]

    vis = (
        obs_93[0]
        + obs_93[6]
        + obs_93[12]
        + obs_93[18]
        + obs_93[24]
        + obs_93[30]
        + obs_93[40]
        + obs_93[50]
        + obs_93[60]
        + obs_93[70]
    )
    vis = np.clip(vis, 0, 4)

    for i in range(4):
        ob_base = 30 + i * 10
        st_base = 171 + i * 10
        genbutsu = obs_93[ob_base] > 0
        state[st_base + 0] = genbutsu
        for suit in range(3):
            off = suit * 9
            state[st_base + 1, off + 0] = genbutsu[off + 3]
            state[st_base + 1, off + 1] = genbutsu[off + 4]
            state[st_base + 1, off + 2] = genbutsu[off + 5]
            state[st_base + 1, off + 3] = genbutsu[off + 0] * genbutsu[off + 6]
            state[st_base + 1, off + 4] = genbutsu[off + 1] * genbutsu[off + 7]
            state[st_base + 1, off + 5] = genbutsu[off + 2] * genbutsu[off + 8]
            state[st_base + 1, off + 6] = genbutsu[off + 3]
            state[st_base + 1, off + 7] = genbutsu[off + 4]
            state[st_base + 1, off + 8] = genbutsu[off + 5]

            state[st_base + 2, off + 0] = vis[off + 1] == 4
            state[st_base + 2, off + 1] = vis[off + 2] == 4
            state[st_base + 2, off + 2] = np.maximum(vis[off + 1] == 4, vis[off + 3] == 4)
            state[st_base + 2, off + 6] = np.maximum(vis[off + 5] == 4, vis[off + 7] == 4)
            state[st_base + 2, off + 7] = vis[off + 6] == 4
            state[st_base + 2, off + 8] = vis[off + 7] == 4

        is_safe = np.clip(state[st_base] + state[st_base + 1] + state[st_base + 2], 0, 1)
        state[st_base + 3, 0:27] = 1.0 - is_safe[0:27]

        for honor in range(27, 34):
            if vis[honor] == 0:
                state[st_base + 4, honor] = 1.0

    rw_idx = np.argmax(obs_93[78]) if np.any(obs_93[78]) else 27
    sw_idx = np.argmax(obs_93[79]) if np.any(obs_93[79]) else 27

    rw = rw_idx - 27
    sw = sw_idx - 27
    state[211, :] = rw / 3.0
    state[212, :] = sw / 3.0

    my_score = self_scores[p_id]
    for i in range(4):
        diff = (self_scores[i] - my_score) / 100000.0
        state[216 + i, :] = diff

    padded = np.pad(state, ((0, 0), (0, 2)), mode="constant")
    return padded.reshape(256, 4, 9)


class MultiAgentMahjongEnvWrapper:
    """
    【リファクタリングのコア】 半荘累計制環境ラッパー (Hanchan Cumulative Wrapper)
    """

    def __init__(self):
        self.env = pymahjong.MahjongEnv()
        self.shanten_calc = Shanten()  # シャンテン計算器を初期化
        self.reset_hanchan()

    def reset(self):
        return self.reset_hanchan()

    def _reset_hand_internal(self):
        """局単位の状態のみをリセットする"""
        self.env.reset()
        self.current_player = self.env.get_curr_player_id()
        self.action_history = []
        # メトリクス抽出用の変数を初期化
        self.last_discarder = -1
        self.p0_min_shanten = None  # 【修正】初期向聴数はNone（起手配牌時に記録し、初期手牌深度の誤加算を防止）
        self._pending_shanten_reduction = 0  # 【追加】未精算のシャンテン進速

    def reset_hanchan(self):
        """半荘全体をリセットし、スコアを初期化する"""
        self.scores = np.array([25000, 25000, 25000, 25000], dtype=np.float32)
        self.kyoku_count = 0
        self.is_hanchan_done = False
        self._reset_hand_internal()
        return self._get_state_dict(), self._get_mask(), self.current_player

    def step(self, action_id):
        p = self.current_player
        valid_actions = self.env.get_valid_actions()
        reward = 0.0

        info = {"hand_done": False, "p0_win": False, "p0_deal_in": False}

        # 異常打ち切り保護
        if action_id not in valid_actions:
            if p == 0:
                reward -= 1.0
            return self._get_state_dict(), self._get_mask(), reward, True, p, info

        # アクションインターセプトによるメトリクス記録 (0..33: 通常打牌, 34..36: 赤宝牌打牌)
        if 0 <= action_id <= 36:
            self.action_history.append((p, action_id))
            self.last_discarder = p

        if action_id == 49:  # ロン (RON)
            if p == 0:
                info["p0_win"] = True
            elif self.last_discarder == 0 and p != 0:
                info["p0_deal_in"] = True
        elif action_id == 50 and p == 0:  # ツモ (TSUMO)
            info["p0_win"] = True
        elif action_id == 46 and p == 0:  # 立直 (RIICHI) - Action ID 46
            # 【优化】为立直动作提供微小的正向补偿，克服模型对 1000 点罚符的恐惧
            reward += 0.1

        self.env.step(p, action_id)
        hand_done = self.env.is_over()
        info["hand_done"] = hand_done

        if hand_done:
            payoffs = self.env.get_payoffs()
            for i in range(4):
                self.scores[i] += float(payoffs[i])
            self.kyoku_count += 1

            # 【修正】毎局の即時素点報酬 (Per-Hand Payoff Reward): 局終了時の点数変動を即座に報酬化
            hand_payoff_reward = float(payoffs[0]) / 10000.0
            reward += hand_payoff_reward

            # 【コアロジック】真の半荘終了を判定: 8局打ち終えたか、または点数が0未満のプレイヤー（ハコ割れ/飛び）がいる場合
            if self.kyoku_count >= 8 or np.any(self.scores < 0):
                self.is_hanchan_done = True

                # 半荘終了時に順位ボーナス（ウマ）を追加精算する
                my_score = self.scores[0]
                rank = sum(1 for x in self.scores if x > my_score)
                
                # 【优化】加重吃四惩罚，强化避四本能 (原为 [1.0, 0.2, -0.3, -0.9])
                rank_bonuses = [1.2, 0.3, -0.1, -1.8]
                bonus = rank_bonuses[min(rank, 3)]

                reward += bonus
                self.current_player = p
            else:
                self.is_hanchan_done = False
                self._reset_hand_internal()
        else:
            self.current_player = self.env.get_curr_player_id()

        return self._get_state_dict(), self._get_mask(), reward, self.is_hanchan_done, self.current_player, info

    def _get_state_dict(self):
        if self.env.is_over():
            return {
                "state_2d": np.zeros((256, 4, 9), dtype=np.float32),
                "cond_vec": np.zeros(16, dtype=np.float32),
                "seq_hist": np.full(72, 272, dtype=np.int64),
            }

        p = self.current_player
        obs_93 = self.env.get_obs(p)

        # 【修正】シャンテン計算。チート防止メカニズムを回避するため、現在の行動プレイヤーのターンでのみ計算する。
        if p == 0:
            # 特徴行列の0から3のインデックスを合算し、34種の牌の枚数を復元する
            tiles34 = (obs_93[0] + obs_93[1] + obs_93[2] + obs_93[3]).astype(np.int32)
            hand_len = int(tiles34.sum())
            is_menzen = hand_len in [13, 14]
            try:
                current_shanten = self.shanten_calc.calculate_shanten(
                    tiles34, use_chiitoitsu=is_menzen, use_kokushi=is_menzen
                )
                if self.p0_min_shanten is None:
                    self.p0_min_shanten = current_shanten
                elif current_shanten < self.p0_min_shanten:
                    self._pending_shanten_reduction += self.p0_min_shanten - current_shanten
                    self.p0_min_shanten = current_shanten
            except ValueError:
                # 【追加】C++ コアが一時的な中間状態（牌数が12など、ルール上あり得ない枚数）
                # を返した場合は、安全にスキップしてクラッシュを防ぐ (Skip invalid transient states)
                pass

        state_2d = decode_obs_93_to_256(obs_93, self.scores, p)

        cond_vec = np.zeros(16, dtype=np.float32)
        rw_idx = np.argmax(obs_93[78]) if np.any(obs_93[78]) else 27
        sw_idx = np.argmax(obs_93[79]) if np.any(obs_93[79]) else 27

        rw = rw_idx - 27
        sw = sw_idx - 27
        cond_vec[4 + rw] = 1.0
        cond_vec[8 + sw] = 1.0
        for i in range(4):
            rel_idx = (p + i) % 4
            cond_vec[i] = (self.scores[rel_idx] - 25000) / 10000.0

        seq_hist = np.full(72, 272, dtype=np.int64)
        if hasattr(self, "action_history"):
            recent_history = self.action_history[-72:]
            for idx, (actor_id, tile_id) in enumerate(recent_history):
                # 【修正】赤宝牌（34, 35, 36）を普通の5（4, 13, 22）にマッピングし、序列履歴でのパディング化（失明）を防ぐ
                if tile_id == 34:
                    tile_id = 4
                elif tile_id == 35:
                    tile_id = 13
                elif tile_id == 36:
                    tile_id = 22
                
                rel_p = (actor_id - p) % 4
                token = int(tile_id) * 8 + rel_p * 2 + 1
                seq_hist[idx] = min(token, 272)

        return {"state_2d": state_2d, "cond_vec": cond_vec, "seq_hist": seq_hist}

    def _get_mask(self):
        mask = np.zeros(54, dtype=np.float32)
        if not self.env.is_over():
            valid_actions = self.env.get_valid_actions()
            for act in valid_actions:
                if act < 54:
                    mask[act] = 1.0
        return mask


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


def async_environment_worker(worker_id, request_queue, response_pipe, trajectory_queue, steps_to_collect, shared_phase):
    env = MultiAgentMahjongEnvWrapper()

    state_dict, mask, current_player = env.reset()
    pending_transition = None
    accumulated_reward = 0.0
    sync_counter = 0

    # ローカルメトリクスの初期化
    local_metrics = {"hand_count": 0, "win_count": 0, "deal_in_count": 0, "shanten_reduction": 0}

    while True:
        request_queue.put({
            "worker_id": worker_id,
            "state_2d": state_dict["state_2d"].astype(np.int8),  # [高速化1] 36KB -> 9KB、Pickle通信のボトルネックを完全に解消
            "cond_vec": state_dict["cond_vec"],
            "seq_hist": state_dict["seq_hist"].astype(np.int16),
            "mask": mask.astype(np.int8)
        })
        
        response = response_pipe.recv()
        action_val = response["action"]
        
        if current_player == 0:
            log_prob_val = response["log_prob"]
            value_val = response["value"]

            # [修正] アクション実行「前」に、前回の打牌以降に発生した向聴進速を抽出し、前回の pending_transition に還元する（Credit Assignment Fix）
            current_phase = shared_phase.value
            shaping_weight = 0.02 if current_phase == 1 else (0.005 if current_phase == 2 else 0.0)
            shaping_reward = env._pending_shanten_reduction * shaping_weight

            if pending_transition is not None:
                accumulated_reward += shaping_reward
                local_metrics["shanten_reduction"] += env._pending_shanten_reduction

                pending_transition["reward"] = accumulated_reward
                pending_transition["done"] = False
                trajectory_queue.put(pending_transition)
                
            accumulated_reward = 0.0
            env._pending_shanten_reduction = 0

            # [追加] 受入(Ukeire)の即時報酬
            if action_val < 34:
                obs_93 = env.env.get_obs(0)
                tiles34 = (obs_93[0] + obs_93[1] + obs_93[2] + obs_93[3]).astype(np.int32)
                if tiles34[action_val] > 0:
                    tiles34[action_val] -= 1
                    is_menzen = int(tiles34.sum()) in [13, 14]
                    try:
                        base_shanten = env.shanten_calc.calculate_shanten(tiles34, use_chiitoitsu=is_menzen, use_kokushi=is_menzen)
                        ukeire_count = 0
                        for i in range(34):
                            if tiles34[i] < 4:
                                tiles34[i] += 1
                                try:
                                    new_s = env.shanten_calc.calculate_shanten(tiles34, use_chiitoitsu=is_menzen, use_kokushi=is_menzen)
                                    if new_s < base_shanten:
                                        ukeire_count += (5 - tiles34[i])
                                except ValueError:
                                    pass
                                tiles34[i] -= 1
                        
                        # 有効牌が1枚増えるごとに +0.0005 の報酬
                        accumulated_reward += ukeire_count * 0.0005
                    except ValueError:
                        pass

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

        next_state_dict, next_mask, step_reward, done, next_player, info = env.step(action_val)

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
                pending_transition["metrics"] = local_metrics.copy()  # 半荘完了時にメトリクスを送信
                trajectory_queue.put(pending_transition)

            accumulated_reward = 0.0
            pending_transition = None
            local_metrics = {"hand_count": 0, "win_count": 0, "deal_in_count": 0, "shanten_reduction": 0}
            next_state_dict, next_mask, next_player = env.reset()

        state_dict, mask, current_player = next_state_dict, next_mask, next_player


# ==========================================
# 6. PPO エンジン (動的学習率 & 分離バッファ対応)
# ==========================================
def directml_safe_bce_with_logits(logits, targets):
    probs = torch.sigmoid(logits)
    probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
    return -(targets * torch.log(probs) + (1.0 - targets) * torch.log(1.0 - probs)).mean()


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
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.buffer = PPOBuffer(num_workers)
        self.hero_buffer = HeroReplayBuffer(max_size=10000)

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
            dones = traj["dones"]
            T = len(rewards)
            total_steps += T

            w_advantages = np.zeros(T, dtype=np.float32)
            gae = 0.0
            next_val = values[-1] if not dones[-1] else 0.0

            for t in reversed(range(T)):
                v_next = next_val if t == T - 1 else values[t + 1]
                delta = rewards[t] + self.gamma * v_next * (1 - dones[t]) - values[t]
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

        self.model.train()

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
                p_out_masked = p_out + (1.0 - mb_masks) * -1e9
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

                log_probs_all = torch.log(new_probs + 1e-8)
                new_log_probs = (log_probs_all * mb_one_hot_actions).sum(dim=-1)
                entropy = -(new_probs * log_probs_all).sum(dim=-1).mean()

                log_diff = torch.clamp(new_log_probs - mb_old_log_probs, -5.0, 5.0)
                ratios = torch.exp(log_diff)
                ratios_bounded = torch.clamp(ratios, 0.0, 3.0)

                surr1 = ratios_bounded * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                v_clipped = mb_old_values + torch.clamp(new_values - mb_old_values, -self.clip_eps, self.clip_eps)
                vf_loss1 = F.smooth_l1_loss(new_values, mb_returns, reduction="none")
                vf_loss2 = F.smooth_l1_loss(v_clipped, mb_returns, reduction="none")
                value_loss = torch.max(vf_loss1, vf_loss2).mean()

                kl_div = (sl_probs * (torch.log(torch.clamp(sl_probs, min=1e-8)) - log_probs_all)).sum(dim=-1).mean()

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
    TARGET_BUFFER_SIZE = NUM_WORKERS * STEPS_PER_WORKER

    if torch_directml.is_available():
        device = torch_directml.device()
    else:
        device = torch.device("cpu")

    model = SmartMahjongMultiTaskNet(input_channels=256, num_blocks=18).to(device)
    sl_base_model = SmartMahjongMultiTaskNet(input_channels=256, num_blocks=18).to(device)

    base_policy_path = "smart_mahjong_base_policy_v2.pth"

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
        model.load_state_dict(adapt_policy_state_dict(torch.load(resume_path, map_location="cpu", weights_only=False)))
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
                sl_base_model.load_state_dict(
                    adapt_policy_state_dict(torch.load(sl_resume_path, map_location="cpu", weights_only=False))
                )
                print(f" -> [Info] 動的SLベースライン: Phase {prev_phase} のモデルで初期化しました ({sl_resume_path})")
        else:
            if os.path.exists(base_policy_path):
                sl_base_model.load_state_dict(
                    adapt_policy_state_dict(torch.load(base_policy_path, map_location="cpu", weights_only=False))
                )
                print(f" -> [Info] SLベースポリシーを読み込みました: {base_policy_path}")
    else:
        if os.path.exists(base_policy_path):
            model.load_state_dict(
                adapt_policy_state_dict(torch.load(base_policy_path, map_location="cpu", weights_only=False))
            )
            sl_base_model.load_state_dict(
                adapt_policy_state_dict(torch.load(base_policy_path, map_location="cpu", weights_only=False))
            )
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
    ckpt_manager = CheckpointManager(max_keep=3)
    logger = TrainingLogger()

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
            args=(i, request_queue, child_pipes[i], trajectory_queue, STEPS_PER_WORKER, shared_phase)
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
        match = re.search(r"_iter(\d+)\.pth", resume_path)
        if match:
            it = int(match.group(1))
            print(f" -> [Info] イテレーションカウントの復元: 第 {it} イテレーションから学習を再開します")

    logger.truncate_after(it)

    try:
        while True:
            it += 1
            iteration_reward = 0.0
            added_steps = 0

            # [追加] 日常追跡指標
            total_hands, total_wins, total_deal_ins, total_shanten_reduction = 0, 0, 0, 0

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
                        masked_logits = p_out + (1.0 - b_mask) * -1e9
                        probs = F.softmax(masked_logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        actions = dist.sample()
                        log_probs = dist.log_prob(actions)
                        
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
                f"✅ Iter [{it:04d}] Phase {current_phase} | Loss: {ppo_loss:.4f} | R: {avg_reward:.4f} | Ent: {avg_entropy:.4f} | Win: {win_rate:.2%} | Deal-in: {deal_in_rate:.2%} | ΔShanten: +{mean_shanten_red:.2f}"
            )
            logger.log_train(
                it, current_phase, ppo_loss, avg_reward, avg_entropy, win_rate, deal_in_rate, mean_shanten_red
            )

            # ==========================================
            # 評価とフェーズ移行ロジック
            # ==========================================
            if it % 500 == 0:
                print(f"\n--- [Eval] Iter {it}: 500輪定期評価を実行します ---")

                avg_rank, avg_net, win_r, deal_r = parallel_evaluate_against_sl(
                    trainer.model, sl_base_model, total_hanchan=2500, num_eval_workers=10
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
                    ckpt_manager.safe_save(trainer.model.state_dict(), true_best_path)
                    print(
                        f"     🏆 [True Best] 対SL評価の最高記録更新！(Rank: {avg_rank:.3f}, Net: {avg_net:.1f}) -> {true_best_path}"
                    )

                if it >= 1000:
                    cv_val = np.std(reward_history_window) / (abs(np.mean(reward_history_window)) + 1e-8)

                    k_val = 1.0
                    if len(eval_rank_history) >= 3:
                        y = eval_rank_history[-3:]
                        x = np.array([0, 1, 2])
                        k_val, _ = np.polyfit(x, y, 1)

                    # 平台期判定：CV 符合强化学习探索方差 (<0.35) 且 顺位斜率平稳 (<0.05)
                    is_plateau = (cv_val < 0.35) and (abs(k_val) < 0.05)
                    # 基线超越判定：以平均顺位（Avg Rank <= 2.40）为核心黄金指标，免除微小负素点死锁
                    exceed_sl = (avg_rank <= 2.40) or (avg_rank <= 2.45 and avg_net > -1200.0)

                    print(
                        f"🔍 [Plateau Check] CV: {cv_val:.4f} (Thresh: <0.35) | Slope |k|: {abs(k_val):.4f} (Thresh: <0.05) | Rank: {avg_rank:.3f} (Thresh: <=2.40)"
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
                            phase2_exceed = exceed_sl and (win_r >= 0.05)
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
                            phase3_success = (avg_rank <= 2.45) and (avg_net > 0)
                            
                            if phase3_success:
                                print(
                                    f"\n👑 [Grand Finale] Phase 3 完璧にクリア！(Rank:{avg_rank:.3f}, Net:{avg_net:.1f}pt)。"
                                )
                                print(" -> 純粋な強化学習が Phase 2 ヒューリスティックモデルを成功裏に超え、AIは極致に達しました！")
                                final_model_path = "smart_mahjong_ppo_final_phase3_MASTER.pth"
                                ckpt_manager.safe_save(trainer.model.state_dict(), final_model_path)
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
                    trainer.model.state_dict(), "smart_mahjong_ppo_latest", current_phase, it
                )

    except KeyboardInterrupt:
        print("\n[Warn] 訓練がユーザーによって中断されました。(Training interrupted by user.)")
    finally:
        print("-> [Info] ワーカープロセスを終了しています...")
        for proc in workers:
            proc.terminate()
            proc.join(timeout=2.0)
