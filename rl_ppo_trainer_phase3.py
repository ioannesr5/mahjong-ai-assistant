import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import os
import random
import time
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import pymahjong # type: ignore
import gc

from torch.optim.optimizer import Optimizer
import torch_directml

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ==========================================
# 1. カスタムオプティマイザ (DirectML Safe AdamW)
# ==========================================
class DirectMLSafeAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0.0: raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(DirectMLSafeAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
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
# 2. ネットワーク構成要素 (SmartMahjongMultiTaskNet V2)
# ==========================================
class FiLMResBlock2D(nn.Module):
    def __init__(self, channels, cond_dim, dropout_p=0.15, res_scale=0.1):
        super(FiLMResBlock2D, self).__init__()
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
        return out

class MahjongBeliefCrossAttention(nn.Module):
    def __init__(self, cnn_dim=1024, seq_dim=256, num_heads=8, dropout_p=0.1):
        super(MahjongBeliefCrossAttention, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=cnn_dim, kdim=seq_dim, vdim=seq_dim, num_heads=num_heads, dropout=dropout_p, batch_first=True)
        self.norm = nn.LayerNorm(cnn_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, cnn_query, seq_kv):
        q = cnn_query.unsqueeze(1)
        attn_out, _ = self.cross_attn(q, seq_kv, seq_kv)
        return self.norm(cnn_query + self.dropout(attn_out.squeeze(1)))

class SmartMahjongMultiTaskNet(nn.Module):
    def __init__(self, input_channels=256, cond_dim=16, seq_vocab=273, num_blocks=18, dropout_p=0.30):
        super(SmartMahjongMultiTaskNet, self).__init__()
        self.conv_init = nn.Conv2d(input_channels, 256, kernel_size=3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(256)
        self.res_blocks = nn.ModuleList([FiLMResBlock2D(256, cond_dim, dropout_p, res_scale=0.1) for _ in range(num_blocks)])
        
        self.cnn_proj = nn.Sequential(nn.Linear(256 * 4 * 9, 1024), nn.LayerNorm(1024), nn.ReLU(inplace=True))
        self.seq_encoder = DiscardSequenceEncoder(vocab_size=seq_vocab, embed_dim=256, num_heads=8, num_layers=4)
        self.cross_attention = MahjongBeliefCrossAttention(cnn_dim=1024, seq_dim=256, num_heads=8, dropout_p=dropout_p)
        self.fusion_fc = nn.Sequential(nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.ReLU(inplace=True), nn.Dropout(p=dropout_p))
        
        self.policy_out = nn.Linear(1024, 47)
        self.value_head = nn.Linear(1024, 1)
        
        def build_aux_mlp(in_dim, out_dim):
            return nn.Sequential(nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Linear(256, out_dim))
            
        self.aux_tenpai = build_aux_mlp(1024, 3)
        self.aux_danger = build_aux_mlp(1024, 102)     
        self.aux_waits  = build_aux_mlp(1024, 102)     

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
        
        if rl_mode: hidden_aux = hidden.detach() 
        else: hidden_aux = hidden
            
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
# 3. マルチエージェント自己対局環境
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

    vis = obs_93[0] + obs_93[6] + obs_93[12] + obs_93[18] + obs_93[24] + \
          obs_93[30] + obs_93[40] + obs_93[50] + obs_93[60] + obs_93[70] 
    vis = np.clip(vis, 0, 4)

    for i in range(4):
        ob_base = 30 + i * 10 
        st_base = 171 + i * 10
        genbutsu = obs_93[ob_base] > 0 
        state[st_base + 0] = genbutsu
        for suit in range(3):
            off = suit * 9
            state[st_base+1, off+0] = genbutsu[off+3] 
            state[st_base+1, off+1] = genbutsu[off+4] 
            state[st_base+1, off+2] = genbutsu[off+5] 
            state[st_base+1, off+3] = genbutsu[off+0] * genbutsu[off+6] 
            state[st_base+1, off+4] = genbutsu[off+1] * genbutsu[off+7] 
            state[st_base+1, off+5] = genbutsu[off+2] * genbutsu[off+8] 
            state[st_base+1, off+6] = genbutsu[off+3] 
            state[st_base+1, off+7] = genbutsu[off+4] 
            state[st_base+1, off+8] = genbutsu[off+5] 

            state[st_base+2, off+0] = (vis[off+1] == 4)
            state[st_base+2, off+1] = (vis[off+2] == 4)
            state[st_base+2, off+2] = np.maximum(vis[off+1]==4, vis[off+3]==4)
            state[st_base+2, off+6] = np.maximum(vis[off+5]==4, vis[off+7]==4)
            state[st_base+2, off+7] = (vis[off+6] == 4)
            state[st_base+2, off+8] = (vis[off+7] == 4)

        is_safe = np.clip(state[st_base] + state[st_base+1] + state[st_base+2], 0, 1)
        state[st_base+3, 0:27] = 1.0 - is_safe[0:27]

        for honor in range(27, 34):
            if vis[honor] == 0:
                state[st_base+4, honor] = 1.0

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

    padded = np.pad(state, ((0,0), (0,2)), mode='constant')
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
            if p == 0: reward -= 1.0
            return self._get_state_dict(), self._get_mask(), reward, True, p

        if action_id < 34: 
            self.action_history.append((p, action_id))

        self.env.step(p, action_id)
        done = self.env.is_over()
        
        if done:
            payoffs = self.env.get_payoffs()
            for i in range(4): self.scores[i] += float(payoffs[i])
            
            base_reward = float(payoffs[0]) / 1000.0
            my_payoff = payoffs[0]
            rank = sum(1 for x in payoffs if x > my_payoff)
            
            rank_bonuses = [1.0, 0.2, -0.3, -0.9]
            bonus = rank_bonuses[min(rank, 3)]
            
            reward = base_reward + bonus
            self.current_player = p
        else:
            self.current_player = self.env.get_curr_player_id()
            
        return self._get_state_dict(), self._get_mask(), reward, done, self.current_player

    def _get_state_dict(self):
        if self.env.is_over():
            return { 'state_2d': np.zeros((256, 4, 9), dtype=np.float32), 'cond_vec': np.zeros(16, dtype=np.float32), 'seq_hist': np.full(72, 272, dtype=np.int64) }

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
        if hasattr(self, 'action_history'):
            recent_history = self.action_history[-72:] 
            for idx, (actor_id, tile_id) in enumerate(recent_history):
                rel_p = (actor_id - p) % 4
                token = int(tile_id) * 8 + rel_p * 2 + 1 
                seq_hist[idx] = min(token, 272)
                
        return { 'state_2d': state_2d, 'cond_vec': cond_vec, 'seq_hist': seq_hist }

    def _get_mask(self):
        mask = np.zeros(47, dtype=np.float32)
        if not self.env.is_over():
            valid_actions = self.env.get_valid_actions()
            for act in valid_actions:
                if act < 47: mask[act] = 1.0
        return mask

# ==========================================
# 4. マルチプロセス・ワーカー定義
# ==========================================
def sync_params(src_model, dst_model):
    for src_p, dst_p in zip(src_model.parameters(), dst_model.parameters()):
        dst_p.data.copy_(src_p.data)

def async_environment_worker(worker_id, shared_model, trajectory_queue, steps_to_collect):
    env = MultiAgentMahjongEnvWrapper()
    
    local_agent_base = SmartMahjongMultiTaskNet(input_channels=256, num_blocks=18).to('cpu')
    local_agent_base.eval()
    sync_params(shared_model, local_agent_base)
    
    wrapper = PolicyInferenceWrapper(local_agent_base)
    dummy_s2d = torch.zeros(1, 256, 4, 9, dtype=torch.float32)
    dummy_c = torch.zeros(1, 16, dtype=torch.float32)
    dummy_seq = torch.zeros(1, 72, dtype=torch.int64)
    traced_agent = torch.jit.trace(wrapper, (dummy_s2d, dummy_c, dummy_seq))
    
    opp_models = [traced_agent for _ in range(3)]

    state_dict, mask, current_player = env.reset()
    pending_transition = None
    accumulated_reward = 0.0
    sync_counter = 0

    while True:
        if sync_counter % 64 == 0: sync_params(shared_model, local_agent_base)
        sync_counter += 1

        s_2d = torch.tensor(state_dict['state_2d'], dtype=torch.float32).unsqueeze(0)
        c_vec = torch.tensor(state_dict['cond_vec'], dtype=torch.float32).unsqueeze(0)
        seq_h = torch.tensor(state_dict['seq_hist'], dtype=torch.int64).unsqueeze(0)
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
                pending_transition['reward'] = accumulated_reward
                pending_transition['done'] = False
                trajectory_queue.put(pending_transition)
                accumulated_reward = 0.0
                
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
                trajectory_queue.put(pending_transition)
            
            accumulated_reward = 0.0
            pending_transition = None
            next_state_dict, next_mask, next_player = env.reset()
                
        state_dict, mask, current_player = next_state_dict, next_mask, next_player

# ==========================================
# 5. PPO エンジン (Phase 3 LR = 5e-6)
# ==========================================

def directml_safe_bce_with_logits(logits, targets):
    """ DirectMLのCPUフォールバックを回避するための手動BCE実装 """
    probs = torch.sigmoid(logits)
    probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
    return -(targets * torch.log(probs) + (1.0 - targets) * torch.log(1.0 - probs)).mean()

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
    def __init__(self, model, sl_model, device, lr=5e-6, kl_beta=0.05, ppo_epochs=4):
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
        if len(self.buffer.rewards) == 0: return 0.0, 0.0 

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
                v_next = next_val if t == steps_per_worker - 1 else w_values[t + 1]
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

        classes = torch.arange(47, device=self.device).unsqueeze(0)
        one_hot_actions = (actions.unsqueeze(1) == classes).to(torch.float32)

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
                
                p_out, v_score, aux_t, aux_d, aux_w = self.model(mb_s_2d, mb_c_vec, mb_seq_h, rl_mode=True)
                new_values = v_score.squeeze(-1)
                p_out_masked = p_out + (1.0 - mb_masks) * -1e9
                new_probs = F.softmax(p_out_masked, dim=-1)
                
                with torch.no_grad():
                    sl_out, _, sl_t, sl_d, sl_w = self.sl_model(mb_s_2d, mb_c_vec, mb_seq_h, rl_mode=False)
                    
                    if mb_masks.size(-1) == 34:
                        full_mask = torch.zeros(mb_masks.size(0), 47, device=self.device)
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
                vf_loss1 = F.smooth_l1_loss(new_values, mb_returns, reduction='none')
                vf_loss2 = F.smooth_l1_loss(v_clipped, mb_returns, reduction='none')
                value_loss = torch.max(vf_loss1, vf_loss2).mean()
                
                kl_div = (sl_probs * (torch.log(torch.clamp(sl_probs, min=1e-8)) - log_probs_all)).sum(dim=-1).mean()
                
                loss_aux_t = directml_safe_bce_with_logits(aux_t, sl_t_target)
                loss_aux_d = directml_safe_bce_with_logits(aux_d, sl_d_target)
                loss_aux_w = directml_safe_bce_with_logits(aux_w, sl_w_target)
                aux_loss = 0.05 * (loss_aux_t + loss_aux_d + loss_aux_w)
                
                total_loss = policy_loss + 1.0 * value_loss - 0.01 * entropy + self.kl_beta * kl_div + aux_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_ppo_loss += total_loss.item()
                total_entropy_val += entropy.item() 
                num_updates += 1
            
        self.buffer.clear()
        if num_updates > 0: return total_ppo_loss / num_updates, total_entropy_val / num_updates
        else: return 0.0, 0.0

def save_rl_training_curve(loss_history, reward_history, entropy_history, chart_path='rl_training_curve_phase3_v2.png'):
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(loss_history, label='PPO Loss', color='purple')
    plt.legend(); plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot(reward_history, label='Avg Reward (with Rank Bonus)', color='darkorange')
    plt.legend(); plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(entropy_history, label='Policy Entropy', color='teal')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()

# ==========================================
# 6. メイン実行スクリプト Phase 3
# ==========================================
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    print("="*60)
    print("🚀 PPO Multi-Agent Self-Play V2 - Phase 3 (Ultimate Fine-tuning)") 
    print("="*60)
    
    NUM_WORKERS = 10                 
    STEPS_PER_WORKER = 256          
    TOTAL_ITERATIONS = 5000         
    TARGET_BUFFER_SIZE = NUM_WORKERS * STEPS_PER_WORKER 
    
    if torch_directml.is_available(): device = torch_directml.device()
    else: device = torch.device('cpu')
    
    model = SmartMahjongMultiTaskNet(input_channels=256, num_blocks=18).to(device)
    sl_base_model = SmartMahjongMultiTaskNet(input_channels=256, num_blocks=18).to(device)
    
    # Phase 3: Phase 2 のベストモデルを読み込む (Load Phase 2 best model)
    base_policy_path = "smart_mahjong_ppo_final_v2.pth"
    if os.path.exists(base_policy_path):
        sl_base_model.load_state_dict(torch.load(base_policy_path, map_location='cpu', weights_only=False))
        model.load_state_dict(torch.load(base_policy_path, map_location='cpu', weights_only=False))
        print(f" -> [Info] RLフェーズ2の最良ポリシーを読み込みました: {base_policy_path}")
    else:
        print(f" -> [Error] {base_policy_path} が見つかりません。")
    
    shared_model = SmartMahjongMultiTaskNet(input_channels=256, num_blocks=18).to('cpu')
    shared_model.load_state_dict(model.state_dict())
    shared_model.share_memory() 
    
    # Phase 3: 学習率を 5e-6 に減衰して極限微調整 (Extreme Fine-tuning LR)
    trainer = PPOKLPenaltyTrainer(model, sl_base_model, device, lr=5e-6, kl_beta=0.05)
    
    trajectory_queue = mp.Queue(maxsize=NUM_WORKERS * 4 * STEPS_PER_WORKER)
    
    workers = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=async_environment_worker, args=(i, shared_model, trajectory_queue, STEPS_PER_WORKER))
        p.start()
        workers.append(p)
    
    ppo_loss_history = []
    reward_history = []
    entropy_history = [] 
    best_avg_reward = -float('inf')

    try:
        for it in range(1, TOTAL_ITERATIONS + 1):
            iteration_reward = 0.0
            added_steps = 0
            
            rollout_pbar = tqdm(total=TARGET_BUFFER_SIZE, desc=f"Iter [{it}/{TOTAL_ITERATIONS}] Async Rollout", leave=False)
            while added_steps < TARGET_BUFFER_SIZE:
                step_data = trajectory_queue.get() 
                
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
                rollout_pbar.update(1)
            rollout_pbar.close()
            
            update_pbar = tqdm(total=trainer.ppo_epochs, desc=f"Iter [{it}/{TOTAL_ITERATIONS}] Optim  ", leave=False)
            ppo_loss, avg_entropy = trainer.update_from_buffer(mini_batch_size=256) 
            update_pbar.update(trainer.ppo_epochs)
            update_pbar.close()

            sync_params(trainer.model.to('cpu'), shared_model)
            trainer.model.to(device)

            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
            avg_reward = iteration_reward / TARGET_BUFFER_SIZE
            ppo_loss_history.append(ppo_loss)
            reward_history.append(avg_reward)
            entropy_history.append(avg_entropy) 

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                # Phase 3 独自のセーブパス (Save specifically for Phase 3)
                best_path = "smart_mahjong_ppo_best_phase3_v2.pth"
                torch.save(trainer.model.state_dict(), best_path)
                print(f"     [*] 新的最高奖励！(包含顺位分) -> {best_path} (Reward: {avg_reward:.4f})")

            if it % 1000 == 0:  
                current_path = "smart_mahjong_ppo_current_v2.pth" 
                torch.save(trainer.model.state_dict(), current_path)
                print(f"     [Save] 当前最新检查点已保存 (最新チェックポイントを保存しました) -> {current_path}")

            print(f"✅ Iter [{it:04d}/{TOTAL_ITERATIONS}] | PPO Loss: {ppo_loss:.4f} | Avg Step Reward: {avg_reward:.4f} | Entropy: {avg_entropy:.4f}")
    except KeyboardInterrupt:
        print("\n[Warn] 訓練がユーザーによって中断されました。(Training interrupted by user.)")
    finally:
        print("-> [Info] ワーカープロセスを終了しています...")
        for p in workers:
            p.terminate()
            p.join(timeout=2.0)

        final_path = "smart_mahjong_ppo_final_v2.pth"
        torch.save(trainer.model.state_dict(), final_path)
        print(f" -> [Save] 5000轮全量训练结束，强制保存最终模型: {final_path}")   

        save_rl_training_curve(ppo_loss_history, reward_history, entropy_history, chart_path='rl_training_curve_phase3_v2.png')
        print("🎉 V2 Phase 3 非同期自己対局パイプラインの実行が完了しました。(Pipeline finished.)")