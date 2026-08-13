import gc
import glob
import math
import os
import re

import numpy as np
import pymahjong  # type: ignore
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch_directml
from torch import nn
from torch.optim.optimizer import Optimizer
from tqdm import tqdm


class CheckpointManager:
    """
    检查点与文件滚动管理模块 (Checkpoint & Rotation Manager)
    """

    def __init__(self, max_keep=3):
        self.max_keep = max_keep

    def safe_save(self, state_dict, filepath):
        """
        原子保存（アトミック保存）: 彻底杜绝断电导致的文件损坏（ファイル破損）
        """
        temp_path = filepath + ".tmp"
        torch.save(state_dict, temp_path)
        os.replace(temp_path, filepath)  # 原子重命名（アトミックリネーム）

    def save_with_rotation(self, state_dict, prefix, current_phase, iteration):
        """
        滚动备份（ローリングバックアップ）: 保存带迭代号的新版本，并自动清理超量旧文件
        """
        filename = f"{prefix}_phase{current_phase}_iter{iteration}.pth"
        self.safe_save(state_dict, filename)

        # 获取所有匹配的最新文件，并按修改时间倒序排列 (更新日時順にソート)
        pattern = f"{prefix}_phase{current_phase}_iter*.pth"
        files = glob.glob(pattern)
        files.sort(key=os.path.getmtime, reverse=True)

        # 删除超出 max_keep 数量的历史文件 (古いファイルを削除)
        for f in files[self.max_keep :]:
            try:
                os.remove(f)
                # print(f"     [Clean] 自动清理历史检查点 (古いチェックポイントを削除): {f}")
                # (为避免日志刷屏，此处静默清理)
            except OSError:
                pass


# CPUスレッドの爆発を防ぐ環境変数設定 (防止 CPU 线程爆炸)[cite: 5]
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ==========================================
# 1. カスタムオプティマイザ (DirectML Safe AdamW)[cite: 5]
# ==========================================
class DirectMLSafeAdamW(Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
    ):
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
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

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
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group["eps"]
                )
                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss


# ==========================================
# 2. ネットワーク構成要素 (SmartMahjongMultiTaskNet V2)[cite: 5]
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
        attn_out = self.out_proj(
            torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, T, C)
        )
        src = self.norm1(src + self.dropout2(attn_out))
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(src))))
        return self.norm2(src + self.dropout(ff_out))


class DiscardSequenceEncoder(nn.Module):
    def __init__(self, vocab_size=273, embed_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=272)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 100, embed_dim))
        self.layers = nn.ModuleList(
            [
                DirectMLSafeTransformerLayer(
                    embed_dim, num_heads, embed_dim * 4, dropout=0.1
                )
                for _ in range(num_layers)
            ]
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
            embed_dim=cnn_dim,
            kdim=seq_dim,
            vdim=seq_dim,
            num_heads=num_heads,
            dropout=dropout_p,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(cnn_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, cnn_query, seq_kv):
        q = cnn_query.unsqueeze(1)
        attn_out, _ = self.cross_attn(q, seq_kv, seq_kv)
        return self.norm(cnn_query + self.dropout(attn_out.squeeze(1)))


class SmartMahjongMultiTaskNet(nn.Module):
    def __init__(
        self,
        input_channels=256,
        cond_dim=16,
        seq_vocab=273,
        num_blocks=18,
        dropout_p=0.30,
    ):
        super().__init__()
        self.conv_init = nn.Conv2d(
            input_channels, 256, kernel_size=3, padding=1, bias=False
        )
        self.bn_init = nn.BatchNorm2d(256)
        self.res_blocks = nn.ModuleList(
            [
                FiLMResBlock2D(256, cond_dim, dropout_p, res_scale=0.1)
                for _ in range(num_blocks)
            ]
        )

        self.cnn_proj = nn.Sequential(
            nn.Linear(256 * 4 * 9, 1024), nn.LayerNorm(1024), nn.ReLU(inplace=True)
        )
        self.seq_encoder = DiscardSequenceEncoder(
            vocab_size=seq_vocab, embed_dim=256, num_heads=8, num_layers=4
        )
        self.cross_attention = MahjongBeliefCrossAttention(
            cnn_dim=1024, seq_dim=256, num_heads=8, dropout_p=dropout_p
        )
        self.fusion_fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )

        self.policy_out = nn.Linear(1024, 47)
        self.value_head = nn.Linear(1024, 1)

        def build_aux_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, out_dim),
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


class PolicyInferenceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, state_2d, cond_vec, seq_hist):
        p_out, v_head, _, _, _ = self.model(state_2d, cond_vec, seq_hist, True)
        return p_out, v_head


# ==========================================
# 3. マルチエージェント自己対局環境[cite: 5]
# ==========================================
def decode_obs_93_to_256(
    obs_93: np.ndarray, self_scores: np.ndarray, p_id: int
) -> np.ndarray:
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
        state[st_base + 9 : st_base + 13] = np.clip(
            total_disc - obs_93[ob_base + 9], 0, 1
        )

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
            state[st_base + 2, off + 2] = np.maximum(
                vis[off + 1] == 4, vis[off + 3] == 4
            )
            state[st_base + 2, off + 6] = np.maximum(
                vis[off + 5] == 4, vis[off + 7] == 4
            )
            state[st_base + 2, off + 7] = vis[off + 6] == 4
            state[st_base + 2, off + 8] = vis[off + 7] == 4

        is_safe = np.clip(
            state[st_base] + state[st_base + 1] + state[st_base + 2], 0, 1
        )
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
    def __init__(self):
        self.env = pymahjong.MahjongEnv()
        self.reset_hanchan()

    def reset(self):
        return self.reset_hanchan()

    def reset_hanchan(self):
        self.scores = np.array([25000, 25000, 25000, 25000], dtype=np.float32)
        return self.reset_hand()

    def reset_hand(self):
        self.env.reset()
        self.current_player = self.env.get_curr_player_id()
        self.action_history = []
        return self._get_state_dict(), self._get_mask(), self.current_player

    def step(self, action_id):
        p = self.current_player
        valid_actions = self.env.get_valid_actions()
        reward = 0.0

        if action_id not in valid_actions:
            if p == 0:
                reward -= 1.0
            return self._get_state_dict(), self._get_mask(), reward, True, p

        if action_id < 34:
            self.action_history.append((p, action_id))

        self.env.step(p, action_id)
        done = self.env.is_over()

        if done:
            payoffs = self.env.get_payoffs()
            for i in range(4):
                self.scores[i] += float(payoffs[i])

            base_reward = float(payoffs[0]) / 1000.0
            my_payoff = payoffs[0]
            rank = sum(1 for x in payoffs if x > my_payoff)

            rank_bonuses = [1.0, 0.2, -0.3, -0.9]
            bonus = rank_bonuses[min(rank, 3)]

            reward = base_reward + bonus
            self.current_player = p
        else:
            self.current_player = self.env.get_curr_player_id()

        return (
            self._get_state_dict(),
            self._get_mask(),
            reward,
            done,
            self.current_player,
        )

    def _get_state_dict(self):
        if self.env.is_over():
            return {
                "state_2d": np.zeros((256, 4, 9), dtype=np.float32),
                "cond_vec": np.zeros(16, dtype=np.float32),
                "seq_hist": np.full(72, 272, dtype=np.int64),
            }

        p = self.current_player
        obs_93 = self.env.get_obs(p)
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
                rel_p = (actor_id - p) % 4
                token = int(tile_id) * 8 + rel_p * 2 + 1
                seq_hist[idx] = min(token, 272)

        return {"state_2d": state_2d, "cond_vec": cond_vec, "seq_hist": seq_hist}

    def _get_mask(self):
        mask = np.zeros(47, dtype=np.float32)
        if not self.env.is_over():
            valid_actions = self.env.get_valid_actions()
            for act in valid_actions:
                if act < 47:
                    mask[act] = 1.0
        return mask


# ==========================================
# 4. 独立SL凍結評価モジュール (Independent SL Eval)
# ==========================================


def evaluation_worker(worker_id, rl_sd, sl_sd, num_hanchan, result_queue):
    """
    [新增] 独立评估子进程 (Evaluation Worker Process)
    在纯 CPU 环境下执行极速 JIT 推理，避免与主进程抢占 DirectML/GPU 资源。
    """
    # 强制将评估节点绑定在单线程，防止 CPU 线程爆炸
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

    # 关闭一致性检查以避开底层兼容性 Bug
    traced_rl = torch.jit.trace(
        rl_wrapper, (dummy_s2d, dummy_c, dummy_seq), check_trace=False
    )
    traced_sl = torch.jit.trace(
        sl_wrapper, (dummy_s2d, dummy_c, dummy_seq), check_trace=False
    )

    opp_models = [traced_sl for _ in range(3)]

    ranks, net_points = [], []
    win_count, deal_in_count, kyoku_count = 0, 0, 0

    state_dict, mask, current_player = env.reset_hanchan()
    completed_hanchan = 0
    last_discarder = -1

    while completed_hanchan < num_hanchan:
        s_2d = torch.tensor(
            state_dict["state_2d"], dtype=torch.float32, device=device
        ).unsqueeze(0)
        c_vec = torch.tensor(
            state_dict["cond_vec"], dtype=torch.float32, device=device
        ).unsqueeze(0)
        seq_h = torch.tensor(
            state_dict["seq_hist"], dtype=torch.int64, device=device
        ).unsqueeze(0)
        t_mask = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            if current_player == 0:
                p_out, _ = traced_rl(s_2d, c_vec, seq_h)
            else:
                p_out, _ = opp_models[current_player - 1](s_2d, c_vec, seq_h)

        masked_logits = p_out + (1.0 - t_mask) * -1e9
        action_val = torch.argmax(masked_logits, dim=-1).item()

        if 0 <= action_val <= 33:
            last_discarder = current_player
        elif action_val == 42:
            if current_player == 0:
                win_count += 1
            elif last_discarder == 0:
                deal_in_count += 1
        elif action_val == 43 and current_player == 0:
            win_count += 1

        next_state_dict, next_mask, _step_reward, done, next_player = env.step(
            action_val
        )

        if done:
            kyoku_count += 1
            my_score = env.scores[0]
            rank = sum(1 for x in env.scores if x > my_score) + 1
            ranks.append(rank)
            net_points.append(env.scores[0] - 25000)

            completed_hanchan += 1
            next_state_dict, next_mask, next_player = env.reset_hanchan()
            last_discarder = -1

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


def parallel_evaluate_against_sl(
    rl_model, sl_model, total_hanchan=2500, num_eval_workers=10
):
    """
    [重构] 多进程并行 SL 冻结评估模块 (Parallel SL Evaluation Module)
    将评估集规模呈指数级放大（如 250 -> 2500 局），利用多核 CPU 分布式处理，时间成本不变。
    """
    print(
        f"\n[Eval] 並列評価モジュールを起動: 総目標 {total_hanchan} 局 ({num_eval_workers} プロセスで分散実行)"
    )

    # 抽取模型权重至 CPU 并转换为纯数据字典，安全穿越进程边界
    rl_sd = {k: v.cpu() for k, v in rl_model.state_dict().items()}
    sl_sd = {k: v.cpu() for k, v in sl_model.state_dict().items()}

    hanchan_per_worker = total_hanchan // num_eval_workers
    result_queue = mp.Queue()
    workers = []

    for i in range(num_eval_workers):
        p = mp.Process(
            target=evaluation_worker,
            args=(i, rl_sd, sl_sd, hanchan_per_worker, result_queue),
        )
        p.start()
        workers.append(p)

    eval_pbar = tqdm(
        total=num_eval_workers, desc="Aggregating Parallel Evals", leave=False
    )

    all_ranks, all_nets = [], []
    total_wins, total_deals, total_kyoku = 0, 0, 0

    # 阻塞收集所有进程结果
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
# 5. マルチプロセス・ワーカー定義 (Worker Process)[cite: 5]
# ==========================================
def sync_params(src_model, dst_model):
    for src_p, dst_p in zip(src_model.parameters(), dst_model.parameters()):
        dst_p.data.copy_(src_p.data)


def async_environment_worker(
    worker_id, shared_model, trajectory_queue, steps_to_collect
):
    env = MultiAgentMahjongEnvWrapper()

    local_agent_base = SmartMahjongMultiTaskNet(input_channels=256, num_blocks=18).to(
        "cpu"
    )
    local_agent_base.eval()
    sync_params(shared_model, local_agent_base)

    wrapper = PolicyInferenceWrapper(local_agent_base)
    dummy_s2d = torch.zeros(1, 256, 4, 9, dtype=torch.float32)
    dummy_c = torch.zeros(1, 16, dtype=torch.float32)
    dummy_seq = torch.zeros(1, 72, dtype=torch.int64)
    traced_agent = torch.jit.trace(
        wrapper, (dummy_s2d, dummy_c, dummy_seq), check_trace=False
    )

    opp_models = [traced_agent for _ in range(3)]

    state_dict, mask, current_player = env.reset()
    pending_transition = None
    accumulated_reward = 0.0
    sync_counter = 0

    while True:
        if sync_counter % 64 == 0:
            sync_params(shared_model, local_agent_base)
        sync_counter += 1

        s_2d = torch.tensor(state_dict["state_2d"], dtype=torch.float32).unsqueeze(0)
        c_vec = torch.tensor(state_dict["cond_vec"], dtype=torch.float32).unsqueeze(0)
        seq_h = torch.tensor(state_dict["seq_hist"], dtype=torch.int64).unsqueeze(0)
        t_mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            if current_player == 0:
                p_out, v_score = traced_agent(s_2d, c_vec, seq_h)
            else:
                p_out, _ = opp_models[current_player - 1](s_2d, c_vec, seq_h)

        masked_logits = p_out + (1.0 - t_mask) * -1e9
        probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        action_val = action.item()

        if current_player == 0:
            log_prob_val = dist.log_prob(action).item()
            value_val = v_score.item()

            if pending_transition is not None:
                pending_transition["reward"] = accumulated_reward
                pending_transition["done"] = False
                trajectory_queue.put(pending_transition)
                accumulated_reward = 0.0

            # 【核心修正】: 注入 worker_id，以便主进程解包时将序列隔离 (Inject worker_id for trajectory isolation)
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

        next_state_dict, next_mask, step_reward, done, next_player = env.step(
            action_val
        )

        if pending_transition is not None:
            accumulated_reward += float(step_reward)

        if done:
            if pending_transition is not None:
                pending_transition["reward"] = accumulated_reward
                pending_transition["done"] = True
                trajectory_queue.put(pending_transition)

            accumulated_reward = 0.0
            pending_transition = None
            next_state_dict, next_mask, next_player = env.reset()

        state_dict, mask, current_player = next_state_dict, next_mask, next_player


# ==========================================
# 6. PPO エンジン (動的学習率 & 分離バッファ対応)
# ==========================================
def directml_safe_bce_with_logits(logits, targets):
    probs = torch.sigmoid(logits)
    probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
    return -(
        targets * torch.log(probs) + (1.0 - targets) * torch.log(1.0 - probs)
    ).mean()


class PPOBuffer:
    """
    【核心修正】: 支持多 Worker 物理隔离的轨迹缓存 (Trajectory buffer supporting multi-worker isolation)
    """

    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.clear()

    def clear(self):
        # 按照 worker_id 独立存储 (Store independently by worker_id)
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
    def __init__(
        self, model, sl_model, device, num_workers, lr=5e-5, kl_beta=0.05, ppo_epochs=4
    ):
        self.device = device
        self.model = model.to(self.device)
        self.sl_model = sl_model.to(self.device)
        self.sl_model.eval()
        self.optimizer = DirectMLSafeAdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-3
        )
        self.kl_beta = kl_beta
        self.clip_eps = 0.2
        self.ppo_epochs = ppo_epochs
        self.gamma = 0.99
        self.gae_lambda = 0.95
        # 初始化时传入 Worker 数量 (Pass number of workers during init)
        self.buffer = PPOBuffer(num_workers)

    def set_learning_rate(self, new_lr):
        for g in self.optimizer.param_groups:
            g["lr"] = new_lr

    def update_from_buffer(self, mini_batch_size=256):
        # 【核心修正】: 为每个 Worker 独立计算 GAE，彻底杜绝时序错位 (Calculate GAE independently per worker to eliminate sequence misalignment)
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

            # GAE (Generalized Advantage Estimation) 严格时序回溯 (Strict time-series backpropagation)
            for t in reversed(range(T)):
                v_next = next_val if t == T - 1 else values[t + 1]
                delta = rewards[t] + self.gamma * v_next * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                w_advantages[t] = gae

            all_advantages.extend(w_advantages)
            all_returns.extend(w_advantages + np.array(values))

            # 直接合并该 Worker 的轨迹数据 (Flatten the trajectory data for this worker)
            all_s_2d.extend(traj["states_2d"])
            all_c_vec.extend(traj["cond_vecs"])
            all_seq_h.extend(traj["seq_hists"])
            all_actions.extend(traj["actions"])
            all_masks.extend(traj["masks"])
            all_old_log_probs.extend(traj["log_probs"])
            all_old_values.extend(values)

        if total_steps == 0:
            return 0.0, 0.0

        self.model.train()

        s_2d = torch.tensor(np.array(all_s_2d), dtype=torch.float32, device=self.device)
        c_vec = torch.tensor(
            np.array(all_c_vec), dtype=torch.float32, device=self.device
        )
        seq_h = torch.tensor(np.array(all_seq_h), dtype=torch.int64, device=self.device)
        actions = torch.tensor(all_actions, dtype=torch.int64, device=self.device)
        masks = torch.tensor(
            np.array(all_masks), dtype=torch.float32, device=self.device
        )
        old_log_probs = torch.tensor(
            all_old_log_probs, dtype=torch.float32, device=self.device
        )

        advantages = torch.tensor(
            all_advantages, dtype=torch.float32, device=self.device
        )
        returns = torch.tensor(all_returns, dtype=torch.float32, device=self.device)
        old_values_tensor = torch.tensor(
            all_old_values, dtype=torch.float32, device=self.device
        )

        adv_mean = advantages.mean()
        adv_std = torch.sqrt(torch.mean((advantages - adv_mean) ** 2) + 1e-8)
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        classes = torch.arange(47, device=self.device).unsqueeze(0)
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

                p_out, v_score, aux_t, aux_d, aux_w = self.model(
                    mb_s_2d, mb_c_vec, mb_seq_h, rl_mode=True
                )
                new_values = v_score.squeeze(-1)
                p_out_masked = p_out + (1.0 - mb_masks) * -1e9
                new_probs = F.softmax(p_out_masked, dim=-1)

                with torch.no_grad():
                    sl_out, _, sl_t, sl_d, sl_w = self.sl_model(
                        mb_s_2d, mb_c_vec, mb_seq_h, rl_mode=False
                    )

                    if mb_masks.size(-1) == 34:
                        full_mask = torch.zeros(
                            mb_masks.size(0), 47, device=self.device
                        )
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
                surr2 = (
                    torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                v_clipped = mb_old_values + torch.clamp(
                    new_values - mb_old_values, -self.clip_eps, self.clip_eps
                )
                vf_loss1 = F.smooth_l1_loss(new_values, mb_returns, reduction="none")
                vf_loss2 = F.smooth_l1_loss(v_clipped, mb_returns, reduction="none")
                value_loss = torch.max(vf_loss1, vf_loss2).mean()

                kl_div = (
                    (
                        sl_probs
                        * (torch.log(torch.clamp(sl_probs, min=1e-8)) - log_probs_all)
                    )
                    .sum(dim=-1)
                    .mean()
                )

                loss_aux_t = directml_safe_bce_with_logits(aux_t, sl_t_target)
                loss_aux_d = directml_safe_bce_with_logits(aux_d, sl_d_target)
                loss_aux_w = directml_safe_bce_with_logits(aux_w, sl_w_target)
                aux_loss = 0.05 * (loss_aux_t + loss_aux_d + loss_aux_w)

                total_loss = (
                    policy_loss
                    + 1.0 * value_loss
                    - 0.01 * entropy
                    + self.kl_beta * kl_div
                    + aux_loss
                )

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
        else:
            return 0.0, 0.0


# ==========================================
# 7. メイン実行スクリプト (動的フェーズ統合版)
# ==========================================

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    print("=" * 60)
    print("🚀 PPO Multi-Agent Self-Play (Unified Dynamic Phase & Safe Checkpoints)")
    print("=" * 60)

    NUM_WORKERS = 10
    STEPS_PER_WORKER = 256
    TARGET_BUFFER_SIZE = NUM_WORKERS * STEPS_PER_WORKER

    if torch_directml.is_available():
        device = torch_directml.device()
    else:
        device = torch.device("cpu")

    model = SmartMahjongMultiTaskNet(input_channels=256, num_blocks=18).to(device)
    sl_base_model = SmartMahjongMultiTaskNet(input_channels=256, num_blocks=18).to(
        device
    )

    base_policy_path = "smart_mahjong_base_policy_v2.pth"

    if os.path.exists(base_policy_path):
        sl_base_model.load_state_dict(
            torch.load(base_policy_path, map_location="cpu", weights_only=False)
        )
        print(f" -> [Info] SLベースポリシーを読み込みました: {base_policy_path}")
    else:
        print(f" -> [Error] 致命的错误: 找不到 SL ベースポリシー {base_policy_path}")

    # ==========================================
    # 动态探测断点路径，支持滚动备份解析 (優先順位に基づくレジューム)
    # ==========================================
    resume_path = None
    current_phase = 1

    for p in [3, 2, 1]:
        # 获取当前 Phase 下存在的 latest 滚动备份列表
        latest_files = glob.glob(f"smart_mahjong_ppo_latest_phase{p}_iter*.pth")
        latest_files.sort(key=os.path.getmtime, reverse=True)

        candidates = []
        if latest_files:
            candidates.append(latest_files[0])  # 优先级1：该阶段时间最新的 latest 备份

        candidates.extend(
            [
                f"smart_mahjong_ppo_TRUE_BEST_phase{p}.pth",  # 优先级2：该阶段独立评估最优
                f"smart_mahjong_ppo_best_phase{p}.pth",  # 优先级3：兼容旧代码遗迹
            ]
        )

        found_path = next((path for path in candidates if os.path.exists(path)), None)
        if found_path:
            resume_path = found_path
            current_phase = p
            break

    if resume_path:
        model.load_state_dict(
            torch.load(resume_path, map_location="cpu", weights_only=False)
        )
        print(
            f" -> [Info] 断点续训 (レジューム): 成功探测并加载 Phase {current_phase} 的历史权重 {resume_path}"
        )
    elif os.path.exists(base_policy_path):
        model.load_state_dict(
            torch.load(base_policy_path, map_location="cpu", weights_only=False)
        )
        print(
            " -> [Info] 未检测到可用断点，从 SL ベースポリシー 开始全新 Phase 1 训练"
        )

    shared_model = SmartMahjongMultiTaskNet(input_channels=256, num_blocks=18).to("cpu")
    shared_model.load_state_dict(model.state_dict())
    shared_model.share_memory()

    phase_lr_map = {1: 5e-5, 2: 1e-5, 3: 5e-6}
    current_lr = phase_lr_map[current_phase]
    print(f" -> [Info] 当前系统设定: Phase = {current_phase}, 学习率 = {current_lr}")

    trainer = PPOKLPenaltyTrainer(
        model,
        sl_base_model,
        device,
        num_workers=NUM_WORKERS,
        lr=current_lr,
        kl_beta=0.05,
    )

    # 实例化检查点管理器，默认保留最近 3 个 latest 备份
    ckpt_manager = CheckpointManager(max_keep=3)

    trajectory_queue = mp.Queue(maxsize=NUM_WORKERS * 4 * STEPS_PER_WORKER)
    workers = []
    for i in range(NUM_WORKERS):
        p = mp.Process(
            target=async_environment_worker,
            args=(i, shared_model, trajectory_queue, STEPS_PER_WORKER),
        )
        p.start()
        workers.append(p)

    reward_history_window = []
    eval_rank_history = []

    ppo_loss_history = []
    reward_history = []
    entropy_history = []

    best_eval_rank = float("inf")
    best_eval_net = -float("inf")
    it = 0
    if resume_path:
        # 匹配如 "_iter350.pth" 中的数字
        match = re.search(r'_iter(\d+)\.pth', resume_path)
        if match:
            it = int(match.group(1))
            print(f" -> [Info] 恢复迭代计数 (イテレーション復元): 从第 {it} 轮开始续训")

    try:
        while True:
            it += 1
            iteration_reward = 0.0
            added_steps = 0

            rollout_pbar = tqdm(
                total=TARGET_BUFFER_SIZE,
                desc=f"Iter [{it}] Phase {current_phase} Async Rollout",
                leave=False,
            )
            while added_steps < TARGET_BUFFER_SIZE:
                step_data = trajectory_queue.get()
                trainer.buffer.add(step_data["worker_id"], step_data)
                iteration_reward += step_data["reward"]
                added_steps += 1
                rollout_pbar.update(1)
            rollout_pbar.close()

            update_pbar = tqdm(
                total=trainer.ppo_epochs,
                desc=f"Iter [{it}] Phase {current_phase} Optim  ",
                leave=False,
            )
            ppo_loss, avg_entropy = trainer.update_from_buffer(mini_batch_size=256)
            update_pbar.update(trainer.ppo_epochs)
            update_pbar.close()

            sync_params(trainer.model.to("cpu"), shared_model)
            trainer.model.to(device)

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

            print(
                f"✅ Iter [{it:04d}] Phase {current_phase} | Loss: {ppo_loss:.4f} | R: {avg_reward:.4f} | Ent: {avg_entropy:.4f}"
            )

            # ==========================================
            # 評価とフェーズ移行ロジック (Eval & Phase Transition)
            # ==========================================
            if it % 500 == 0:
                print(f"\n--- [Eval] Iter {it}: 500輪定期評価を実行します ---")
                avg_rank, avg_net, win_r, deal_r = parallel_evaluate_against_sl(
                    trainer.model,
                    sl_base_model,
                    total_hanchan=2500,
                    num_eval_workers=10,
                )

                eval_rank_history.append(avg_rank)

                print(
                    f"📊 [Eval Result] 平均順位: {avg_rank:.3f} | 局均浄勝素点: {avg_net:.1f} pt | 和了率: {win_r:.3%} | 放銃率: {deal_r:.3%}"
                )

                # 【原子保存应用 1】: 保存独立评估最佳模型
                if avg_rank < best_eval_rank or (
                    avg_rank == best_eval_rank and avg_net > best_eval_net
                ):
                    best_eval_rank = avg_rank
                    best_eval_net = avg_net
                    true_best_path = (
                        f"smart_mahjong_ppo_TRUE_BEST_phase{current_phase}.pth"
                    )
                    ckpt_manager.safe_save(trainer.model.state_dict(), true_best_path)
                    print(
                        f"     🏆 [True Best] 対SL評価の最高記録更新！(Rank: {avg_rank:.3f}, Net: {avg_net:.1f}) -> {true_best_path}"
                    )

                if it >= 1000:
                    cv_val = np.std(reward_history_window) / (
                        abs(np.mean(reward_history_window)) + 1e-8
                    )

                    k_val = 1.0
                    if len(eval_rank_history) >= 3:
                        y = eval_rank_history[-3:]
                        x = np.array([0, 1, 2])
                        k_val, _ = np.polyfit(x, y, 1)

                    is_plateau = (cv_val < 0.02) and (abs(k_val) < 0.005)
                    exceed_sl = (avg_rank <= 2.42) and (avg_net > 100.0)

                    print(
                        f"🔍 [Plateau Check] CV: {cv_val:.4f} (Thresh: <0.02) | Slope |k|: {abs(k_val):.4f} (Thresh: <0.005)"
                    )

                    if is_plateau:
                        if current_phase == 1:
                            if exceed_sl:
                                print(
                                    "\n🌟 [Phase Transition] SLモデルを超越、かつプラトーに到達。Phase 2 (微調整) に移行します！"
                                )
                                current_phase = 2
                                trainer.set_learning_rate(1e-5)
                                best_eval_rank, best_eval_net = (
                                    float("inf"),
                                    -float("inf"),
                                )
                                eval_rank_history.clear()
                            else:
                                print(
                                    "\n⚠️ [Warning] SLモデル未超過でプラトーに到達（局所最適に陥落）。Phase 2への移行をブロックします！"
                                )
                                new_kl = max(0.02, trainer.kl_beta * 0.5)
                                print(
                                    f" -> KLペナルティ(kl_beta)を {trainer.kl_beta:.3f} から {new_kl:.3f} へ下調し、探索を促進します。"
                                )
                                trainer.kl_beta = new_kl
                                eval_rank_history.clear()

                        elif current_phase == 2:
                            print(
                                "\n🌟 [Phase Transition] Phase 2 でプラトーに到達。最終段階 Phase 3 (極限微調整) に移行します！"
                            )
                            current_phase = 3
                            trainer.set_learning_rate(5e-6)
                            best_eval_rank, best_eval_net = float("inf"), -float("inf")
                            eval_rank_history.clear()

                        elif current_phase == 3:
                            print(
                                "\n🛑 [Training Complete] Phase 3 でプラトーに到達しました。学習を完了し、最終モデルを保存して終了します。"
                            )
                            final_model_path = "smart_mahjong_ppo_final_phase3.pth"
                            ckpt_manager.safe_save(
                                trainer.model.state_dict(), final_model_path
                            )
                            print(f" -> 最終モデルを保存: {final_model_path}")
                            break

            # 【滚动备份应用】: 每 50 轮执行一次安全的带序号最新进度保存，并自动清理超量文件
            if it % 50 == 0:
                ckpt_manager.save_with_rotation(
                    trainer.model.state_dict(),
                    "smart_mahjong_ppo_latest",
                    current_phase,
                    it,
                )

    except KeyboardInterrupt:
        print(
            "\n[Warn] 訓練がユーザーによって中断されました。(Training interrupted by user.)"
        )
    finally:
        print("-> [Info] ワーカープロセスを終了しています...")
        for p in workers:
            p.terminate()
            p.join(timeout=2.0)
