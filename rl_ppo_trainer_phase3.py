import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import os
import random
import time
import math
import matplotlib.pyplot as plt
import torch_directml  # AMD GPU 用 DirectML バックエンド
from tqdm import tqdm  # 端末プログレスバー用
import pymahjong # type: ignore


# CPUスレッドの爆発を防ぐ環境変数設定 (防止 CPU 线程爆炸)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ==========================================
# 1. カスタムオプティマイザ (DirectML Safe AdamW)
# ==========================================
from torch.optim.optimizer import Optimizer

class DirectMLSafeAdamW(Optimizer):
    """
    DirectML環境下での 'aten::lerp' CPUフォールバックを完全に回避するためのカスタムAdamW。
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(DirectMLSafeAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                step = state['step']

                p.mul_(1 - group['lr'] * group['weight_decay'])
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad * grad, alpha=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss

# ==========================================
# 2. ネットワーク構成要素 (Network Components)
# ==========================================

class FiLMResBlock2D(nn.Module):
    def __init__(self, channels, cond_dim, dropout_p=0.15):
        super(FiLMResBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.film_gen = nn.Linear(cond_dim, channels * 2)

    def forward(self, x, cond):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        film_params = self.film_gen(cond).view(x.size(0), -1, 1, 1)
        gamma, beta = film_params.chunk(2, dim=1)
        out = (1.0 + gamma) * out + beta
        out += residual
        return F.relu(out)

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
        super(DiscardSequenceEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=272)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 100, embed_dim))
        self.layers = nn.ModuleList([
            DirectMLSafeTransformerLayer(embed_dim, num_heads, embed_dim * 4, dropout=0.1) 
            for _ in range(num_layers)
        ])

    def forward(self, x):
        seq_len = x.size(1)
        out = self.embedding(x) + self.pos_embedding[:, :seq_len, :]
        for layer in self.layers:
            out = layer(out)
        return out[:, -1, :] 

class SmartMahjongMultiTaskNet(nn.Module):
    def __init__(self, input_channels=128, cond_dim=16, seq_vocab=273, num_blocks=10, dropout_p=0.30):
        super(SmartMahjongMultiTaskNet, self).__init__()
        self.conv_init = nn.Conv2d(input_channels, 256, kernel_size=3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(256)
        self.res_blocks = nn.ModuleList([FiLMResBlock2D(256, cond_dim, dropout_p) for _ in range(num_blocks)])
        
        self.cnn_proj = nn.Sequential(
            nn.Linear(256 * 4 * 9, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True)
        )
        
        self.seq_encoder = DiscardSequenceEncoder(vocab_size=seq_vocab, embed_dim=256, num_heads=8, num_layers=4)
        
        fused_dim = 1024 + 256
        self.fusion_fc = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        )
        self.policy_discard = nn.Linear(1024, 34)
        self.policy_action = nn.Linear(1024, 6)
        self.policy_riichi = nn.Linear(1024, 2)
        self.value_head = nn.Linear(1024, 1)
        
        self.aux_tenpai = nn.Linear(1024, 3)     
        self.aux_danger = nn.Linear(1024, 102)   

    def forward(self, state_2d, cond_vec, seq_hist, rl_mode=False):
        out = F.relu(self.bn_init(self.conv_init(state_2d)))
        for block in self.res_blocks:
            out = block(out, cond_vec)
        out_flat = out.view(out.size(0), -1)
        
        cnn_feat = self.cnn_proj(out_flat)
        seq_feat = self.seq_encoder(seq_hist) 
        fused = torch.cat([cnn_feat, seq_feat], dim=1) 
        hidden = self.fusion_fc(fused)
        
        p_disc = self.policy_discard(hidden).to(torch.float32)
        p_act = self.policy_action(hidden).to(torch.float32)
        p_riichi = self.policy_riichi(hidden).to(torch.float32)
        v_head = self.value_head(hidden).to(torch.float32)
        
        if rl_mode:
            aux_t = torch.empty(0, device=hidden.device)
            aux_d = torch.empty(0, device=hidden.device)
        else:
            aux_t = self.aux_tenpai(hidden).to(torch.float32)
            aux_d = self.aux_danger(hidden).to(torch.float32)

        return p_disc, p_act, p_riichi, v_head, aux_t, aux_d

# ==========================================
# 3. マルチエージェント自己対局環境 (Multi-Agent Self-Play Environment)
# ==========================================

class MultiAgentMahjongEnvWrapper:
    """
    リアルな日本麻雀ルールに基づく、マルチエージェント対応の半荘戦シミュレータ。
    """
    def __init__(self):
        self.env = pymahjong.MahjongEnv()
        self.reset_hanchan()

    def reset(self):
        return self.reset_hanchan()

    def reset_hanchan(self):
        self.scores = np.array([25000, 25000, 25000, 25000], dtype=np.float32)
        self.round_wind = 0  
        self.round_num = 1   
        self.dealer = 0      
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
            reward = float(payoffs[0]) / 1000.0
            self.current_player = p
        else:
            self.current_player = self.env.get_curr_player_id()
            
        return self._get_state_dict(), self._get_mask(), reward, done, self.current_player

    def _get_state_dict(self):
        p = self.current_player
        
        seq_hist = np.full(72, 272, dtype=np.int64) 
        if hasattr(self, 'action_history'):
            recent_history = self.action_history[-72:] 
            for idx, (actor_id, tile_id) in enumerate(recent_history):
                rel_p = (actor_id - p) % 4
                cut_type = 1 
                token = int(tile_id) * 8 + rel_p * 2 + cut_type
                seq_hist[idx] = min(token, 272)
                
        if self.env.is_over():
            return {
                'state_2d': np.zeros((128, 4, 9), dtype=np.float32),
                'cond_vec': np.zeros(16, dtype=np.float32),
                'seq_hist': seq_hist
            }

        obs = self.env.get_obs(p)
        state_2d = np.zeros((128, 4, 9), dtype=np.float32)
        hand_counts = np.sum(obs[0:4, :], axis=0) 

        for i in range(34):
            count = int(hand_counts[i])
            if count > 0:
                suit = i // 9
                num = i % 9 if suit < 3 else i - 27
                count = min(count, 4)
                state_2d[0, suit, num] = float(count)

        cond_vec = np.zeros(16, dtype=np.float32)
        wind_idx = 4 + self.round_wind
        if wind_idx < 8:
            cond_vec[wind_idx] = 1.0
        cond_vec[12] = (self.scores[p] - 25000.0) / 100000.0
        cond_vec[13] = float(self.round_num) / 4.0

        return {
            'state_2d': state_2d,
            'cond_vec': cond_vec,
            'seq_hist': seq_hist
        }

    def _get_mask(self):
        mask = np.zeros(34, dtype=np.float32)
        if not self.env.is_over():
            valid_actions = self.env.get_valid_actions()
            for act in valid_actions:
                if act < 34:
                    mask[act] = 1.0
        return mask

# ==========================================
# 4. 対戦相手プール管理 (Opponent Pool Manager)
# ==========================================

class OpponentPoolManager:
    """
    对手池管理器：仅收录打破历史最高纪录的巅峰版本及其里程碑。
    """
    def __init__(self, base_path, pool_dir="model_pool"):
        self.pool_dir = pool_dir
        self.base_path = base_path
        self.history_paths = []
        os.makedirs(self.pool_dir, exist_ok=True)

    def add_peak_history(self, state_dict, iteration, reward):
        path = os.path.join(self.pool_dir, f"peak_iter_{iteration}_rew_{reward:.4f}.pth")
        torch.save(state_dict, path)
        self.history_paths.append(path)
        if len(self.history_paths) > 10:
            old_path = self.history_paths.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)
        print(f"     [*] 对手池已更新：成功收录新的巅峰模型快照 (Iteration {iteration})")

    def sample_opponent_paths(self):
        opponents = []
        for _ in range(3):
            r = random.random()
            if r < 0.60:
                opponents.append("latest")
            elif r < 0.90 and len(self.history_paths) > 0:
                opponents.append(random.choice(self.history_paths))
            else:
                opponents.append("base")
        return opponents

# ==========================================
# 5. マルチプロセス・ワーカー定義 (Multiprocessing Worker)
# ==========================================

def async_environment_worker(worker_id, model_queue, trajectory_queue, steps_to_collect, base_policy_path):
    """
    4人のエージェント推論を統合したワーカー。厳密なブロック同期でオンポリシーを維持。
    """
    env = MultiAgentMahjongEnvWrapper()
    
    local_agent = SmartMahjongMultiTaskNet().to('cpu')
    local_agent.eval()
    
    opp_models = [SmartMahjongMultiTaskNet().to('cpu') for _ in range(3)]
    for opp in opp_models:
        opp.eval()
        
    if os.path.exists(base_policy_path):
        base_state = torch.load(base_policy_path, map_location='cpu', weights_only=False)
        for opp in opp_models:
            opp.load_state_dict(base_state)
    else:
        base_state = local_agent.state_dict()

    state_dict, mask, current_player = env.reset()
    
    pending_transition = None
    accumulated_reward = 0.0
    
    while True:
        msg = model_queue.get() 
        if msg == "TERMINATE":
            break
            
        cmd, opp_paths = msg
        if cmd == "SYNC":
            time.sleep(random.uniform(0.0, 1.0))
            for _ in range(10): 
                try:
                    latest_state_dict = torch.load("sync_current_model.pth", map_location='cpu', weights_only=False)
                    break
                except Exception:
                    time.sleep(0.1)
            
            local_agent.load_state_dict(latest_state_dict)
            
            for i, path in enumerate(opp_paths):
                if path == "latest":
                    opp_models[i].load_state_dict(latest_state_dict)
                elif path == "base":
                    opp_models[i].load_state_dict(base_state)
                else:
                    opp_models[i].load_state_dict(torch.load(path, map_location='cpu', weights_only=False))

        trajectory_chunk = []
        steps_collected = 0
        
        while steps_collected < steps_to_collect:
            s_2d = torch.tensor(state_dict['state_2d'], dtype=torch.float32).unsqueeze(0)
            c_vec = torch.tensor(state_dict['cond_vec'], dtype=torch.float32).unsqueeze(0)
            seq_h = torch.tensor(state_dict['seq_hist'], dtype=torch.int64).unsqueeze(0)
            t_mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                if current_player == 0:
                    p_disc, _, _, v_score, _, _ = local_agent(s_2d, c_vec, seq_h, rl_mode=True)
                else:
                    p_disc, _, _, _, _, _ = opp_models[current_player - 1](s_2d, c_vec, seq_h, rl_mode=True)
                    
            masked_logits = p_disc + (1.0 - t_mask) * -1e9
            probs = F.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            action_val = action.item()
            
            if current_player == 0:
                log_prob_val = dist.log_prob(action).item()
                value_val = v_score.item()
                
                if pending_transition is not None:
                    pending_transition['reward'] = accumulated_reward
                    pending_transition['done'] = False
                    trajectory_chunk.append(pending_transition)
                    steps_collected += 1
                    accumulated_reward = 0.0
                    
                    if steps_collected >= steps_to_collect:
                        break
                    
                pending_transition = {
                    'state_2d': state_dict['state_2d'],
                    'cond_vec': state_dict['cond_vec'],
                    'seq_hist': state_dict['seq_hist'],
                    'action': action_val,
                    'mask': mask,
                    'log_prob': log_prob_val,
                    'value': value_val
                }
                
            next_state_dict, next_mask, step_reward, done, next_player = env.step(action_val)
            
            if pending_transition is not None:
                accumulated_reward += float(step_reward)
            
            if done:
                if pending_transition is not None:
                    pending_transition['reward'] = accumulated_reward
                    pending_transition['done'] = True
                    trajectory_chunk.append(pending_transition)
                    steps_collected += 1
                
                accumulated_reward = 0.0
                pending_transition = None
                next_state_dict, next_mask, next_player = env.reset()
                
                if steps_collected >= steps_to_collect:
                    state_dict, mask, current_player = next_state_dict, next_mask, next_player
                    break
                    
            state_dict, mask, current_player = next_state_dict, next_mask, next_player
                
        trajectory_queue.put(trajectory_chunk)

# ==========================================
# 6. PPO エンジン (PPO KL-Penalty Engine)
# ==========================================

class PPOBuffer:
    def __init__(self):
        self.states_2d, self.cond_vecs, self.seq_hists = [], [], []
        self.actions, self.masks, self.log_probs = [], [], []
        self.rewards, self.state_values, self.dones = [], [], []

    def clear(self):
        self.states_2d.clear(); self.cond_vecs.clear(); self.seq_hists.clear()
        self.actions.clear(); self.masks.clear(); self.log_probs.clear()
        self.rewards.clear(); self.state_values.clear(); self.dones.clear()

class PPOKLPenaltyTrainer:
    def __init__(self, model, sl_model, device, lr=3e-4, kl_beta=0.05, ppo_epochs=4):
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
        self.buffer = PPOBuffer()

    def update_from_buffer(self, mini_batch_size=256):
        # 報酬がない場合は更新スキップ (若无奖励数据则跳过更新)
        if len(self.buffer.rewards) == 0:
            return 0.0, 0.0

        self.model.train()
        s_2d = torch.tensor(np.array(self.buffer.states_2d), dtype=torch.float32, device=self.device)
        c_vec = torch.tensor(np.array(self.buffer.cond_vecs), dtype=torch.float32, device=self.device)
        seq_h = torch.tensor(np.array(self.buffer.seq_hists), dtype=torch.int64, device=self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.int64, device=self.device)
        masks = torch.tensor(np.array(self.buffer.masks), dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        
        rewards = self.buffer.rewards
        old_values = self.buffer.state_values
        dones = self.buffer.dones

        advantages = []
        steps_per_worker = 256 
        num_workers = len(rewards) // steps_per_worker
        
        for w in range(num_workers):
            start_idx = w * steps_per_worker
            end_idx = start_idx + steps_per_worker
            
            w_rewards = rewards[start_idx:end_idx]
            w_values = old_values[start_idx:end_idx]
            w_dones = dones[start_idx:end_idx]
            
            next_val = w_values[-1] if not w_dones[-1] else 0.0 
            
            w_advantages = []
            gae = 0.0
            for t in reversed(range(steps_per_worker)):
                if t == steps_per_worker - 1:
                    v_next = next_val
                else:
                    v_next = w_values[t + 1]
                
                delta = w_rewards[t] + self.gamma * v_next * (1 - w_dones[t]) - w_values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - w_dones[t]) * gae
                w_advantages.insert(0, gae)
                
            advantages.extend(w_advantages)
            
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        old_values_tensor = torch.tensor(old_values, dtype=torch.float32, device=self.device)
        returns = advantages + old_values_tensor
        
        adv_mean = advantages.mean()
        adv_std = torch.sqrt(torch.mean((advantages - adv_mean) ** 2) + 1e-8)
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        classes = torch.arange(34, device=self.device).unsqueeze(0)
        one_hot_actions = (actions.unsqueeze(1) == classes).to(torch.float32)

        # 損失とエントロピーの初期化 (初始化损失与熵累计器)
        total_ppo_loss = 0.0
        total_entropy_val = 0.0
        batch_size = len(rewards)
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
                
                p_disc, _, _, v_score, _, _ = self.model(mb_s_2d, mb_c_vec, mb_seq_h)
                new_values = v_score.squeeze(-1)
                p_disc_masked = p_disc + (1.0 - mb_masks) * -1e9
                new_probs = F.softmax(p_disc_masked, dim=-1)
                
                with torch.no_grad():
                    sl_disc, _, _, _, _, _ = self.sl_model(mb_s_2d, mb_c_vec, mb_seq_h)
                    sl_disc_masked = sl_disc + (1.0 - mb_masks) * -1e9
                    sl_probs = F.softmax(sl_disc_masked, dim=-1)
                
                log_probs_all = torch.log(new_probs + 1e-8)
                new_log_probs = (log_probs_all * mb_one_hot_actions).sum(dim=-1)
                
                # エントロピーの計算 (计算策略熵)
                entropy = -(new_probs * log_probs_all).sum(dim=-1).mean()
                
                log_diff = torch.clamp(new_log_probs - mb_old_log_probs, -5.0, 5.0)
                ratios = torch.exp(log_diff)
                ratios_bounded = torch.clamp(ratios, 0.0, 3.0)
                
                surr1 = ratios_bounded * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                v_clipped = mb_old_values + torch.clamp(new_values - mb_old_values, -self.clip_eps, self.clip_eps)
                vf_loss1 = F.smooth_l1_loss(new_values, mb_returns, reduction='none')
                vf_loss2 = F.smooth_l1_loss(v_clipped, mb_returns, reduction='none')
                value_loss = torch.max(vf_loss1, vf_loss2).mean()
                
                sl_probs_safe = torch.clamp(sl_probs, min=1e-8)
                kl_div = (sl_probs * (torch.log(sl_probs_safe) - log_probs_all)).sum(dim=-1).mean()
                
                total_loss = policy_loss + 1.0 * value_loss - 0.01 * entropy + self.kl_beta * kl_div
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # 損失とエントロピーを累積する (累加损失与熵)
                total_ppo_loss += total_loss.item()
                total_entropy_val += entropy.item()
                num_updates += 1
            
        self.buffer.clear()
        
        # PPO LossとEntropyの平均値をタプルとして返す (同时返回 PPO 损失与平均熵)
        if num_updates > 0:
            return total_ppo_loss / num_updates, total_entropy_val / num_updates
        else:
            return 0.0, 0.0

def save_rl_training_curve(loss_history, reward_history, chart_path='rl_training_curve_phase3.png'): # 【変更】 Phase3用の出力ファイル名に変更 (修改为Phase3专用的图表名称以防覆盖)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='PPO Loss', color='purple')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('PPO Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(reward_history, label='Avg Reward', color='darkorange')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('Multi-Agent Self-Play Reward')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()

# ==========================================
# 7. メイン実行スクリプト (Main Routine)
# ==========================================

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    print("="*50)
    print("🚀 PPO Multi-Agent Self-Play Pipeline (4-Player) - Phase 3") # 【変更】 Phase 3 
    print("="*50)
    
    NUM_WORKERS = 10                 
    STEPS_PER_WORKER = 512          
    TOTAL_ITERATIONS = 5000         
    TARGET_BUFFER_SIZE = NUM_WORKERS * STEPS_PER_WORKER 
    
    if torch_directml.is_available():
        device = torch_directml.device()
    else:
        device = torch.device('cpu')
    
    model = SmartMahjongMultiTaskNet()
    sl_base_model = SmartMahjongMultiTaskNet()
    
    # 【変更】 Phase2の最終状態である同期モデルをベースポリシーとして読み込む 
    # (修改：默认加载上一阶段最后一刻的同步模型权重)
    base_policy_path = "sync_current_model.pth"
    if os.path.exists(base_policy_path):
        sl_base_model.load_state_dict(torch.load(base_policy_path, map_location='cpu', weights_only=False))
        model.load_state_dict(torch.load(base_policy_path, map_location='cpu', weights_only=False))
        print(f" -> [Info] RLフェーズ2のポリシーを読み込みました: {base_policy_path}")
    
    # 学習率は現状 5e-5 を維持しています (当前学习率维持在 5e-5)
    trainer = PPOKLPenaltyTrainer(model, sl_base_model, device, lr=5e-5, kl_beta=0.05)
    pool_manager = OpponentPoolManager(base_policy_path)
    
    trajectory_queue = mp.Queue()
    model_queues = [mp.Queue() for _ in range(NUM_WORKERS)]
    
    sync_model_path = "sync_current_model.pth"
    torch.save(model.state_dict(), sync_model_path)
    initial_opps = pool_manager.sample_opponent_paths()
    
    workers = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=async_environment_worker, args=(i, model_queues[i], trajectory_queue, STEPS_PER_WORKER, base_policy_path))
        p.start()
        workers.append(p)
        model_queues[i].put(("SYNC", initial_opps)) 
    
    ppo_loss_history = []
    reward_history = []
    best_avg_reward = -float('inf')

    try:
        for it in range(1, TOTAL_ITERATIONS + 1):
            collected_steps = 0
            iteration_reward = 0.0
            
            rollout_pbar = tqdm(total=TARGET_BUFFER_SIZE, desc=f"Iter [{it}/{TOTAL_ITERATIONS}] Rollout", leave=False)
            while collected_steps < TARGET_BUFFER_SIZE:
                chunk = trajectory_queue.get()
                added_steps = 0
                for step_data in chunk:
                    if len(trainer.buffer.rewards) >= TARGET_BUFFER_SIZE:
                        break 
                        
                    trainer.buffer.states_2d.append(step_data['state_2d'])
                    trainer.buffer.cond_vecs.append(step_data['cond_vec'])
                    trainer.buffer.seq_hists.append(step_data['seq_hist'])
                    trainer.buffer.actions.append(step_data['action'])
                    trainer.buffer.masks.append(step_data['mask'])
                    trainer.buffer.log_probs.append(step_data['log_prob'])
                    trainer.buffer.rewards.append(step_data['reward'])
                    trainer.buffer.state_values.append(step_data['value'])
                    trainer.buffer.dones.append(step_data['done'])
                    iteration_reward += step_data['reward']
                    added_steps += 1
                    
                collected_steps += added_steps
                rollout_pbar.update(added_steps)
            rollout_pbar.close()
            
            update_pbar = tqdm(total=trainer.ppo_epochs, desc=f"Iter [{it}/{TOTAL_ITERATIONS}] Optim  ", leave=False)
            
            # PPO LossとEntropyの2つの戻り値を受け取る (接收返回的 PPO 损失与策略熵两个参数)
            ppo_loss, avg_entropy = trainer.update_from_buffer(mini_batch_size=512) 
            
            update_pbar.update(trainer.ppo_epochs)
            update_pbar.close()
            
            avg_reward = iteration_reward / TARGET_BUFFER_SIZE
            ppo_loss_history.append(ppo_loss)
            reward_history.append(avg_reward)

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                # 【変更】 Phase3の最高記録を上書きしないよう、保存先ファイル名を変更 
                # (修改：更改保存名称，防止覆盖第二阶段的历史最高记录点)
                best_path = "smart_mahjong_ppo_best_phase3.pth"
                torch.save(trainer.model.state_dict(), best_path)
                print(f"     [*] 新的最高奖励！已更新最优模型存档 -> {best_path} (Reward: {avg_reward:.4f})")
                pool_manager.add_peak_history(trainer.model.state_dict(), it, avg_reward)

            torch.save(trainer.model.state_dict(), sync_model_path)
            new_opps = pool_manager.sample_opponent_paths()
                
            for q in model_queues:
                q.put(("SYNC", new_opps))
            
            # ターミナル出力にエントロピー(Entropy)を追加する (在终端输出日志中追加显示 Entropy 数值)
            print(f"✅ Iter [{it:04d}/{TOTAL_ITERATIONS}] | PPO Loss: {ppo_loss:.4f} | Avg Step Reward: {avg_reward:.4f} | Entropy: {avg_entropy:.4f}")

    except KeyboardInterrupt:
        print("\n[Warn] 訓練がユーザーによって中断されました。(Training interrupted by user.)")
    finally:
        print("-> [Info] ワーカープロセスを終了しています...")
        for q in model_queues:
            q.put("TERMINATE")
        for p in workers:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
                
        save_rl_training_curve(ppo_loss_history, reward_history, chart_path='rl_training_curve_phase3.png')
        print("🎉 自己対局パイプラインの実行が完了しました。(Pipeline finished.)")