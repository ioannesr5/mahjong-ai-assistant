import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from torch.optim.optimizer import Optimizer
import torch_directml
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ==========================================
# 1. ネットワーク構成要素 (Network Components V2)
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

# ==========================================
# 2. 動的データ拡張 (On-the-fly Data Augmentation)
# ==========================================
def apply_dynamic_augmentation_2d(state_2d, target_discard):
    perm = torch.randperm(3) 
    if not torch.equal(perm, torch.tensor([0, 1, 2])):
        new_state = state_2d.clone()
        
        # 1. 花色空間の置換
        new_state[:, :3, :] = state_2d[:, perm, :]
        
        # 【完全修正】: 赤ドラチャネル (4, 5, 6) の置換
        # new_state の 4,5,6 チャネルには既に「花色が置換された状態」の赤ドラフラグが存在します。
        # ただし、元のチャネル位置のままでは物理的な花色と一致しません。
        # permに従って正しいチャネル位置(4, 5, 6)へアサインし直します。
        temp_red = new_state[4:7, :, :].clone()
        for i in range(3):
            new_state[perm[i] + 4, :, :] = temp_red[i, :, :]
            
        state_2d = new_state
        
        if target_discard < 27:
            suit = target_discard // 9
            num = target_discard % 9
            new_suit = (perm == suit).nonzero(as_tuple=True)[0].item()
            target_discard = new_suit * 9 + num

    if torch.rand(1).item() < 0.5:
        state_2d[:, :3, :] = torch.flip(state_2d[:, :3, :], dims=[2])
        if target_discard < 27:
            suit = target_discard // 9
            num = target_discard % 9
            target_discard = suit * 9 + (8 - num)

    return state_2d, target_discard

# ==========================================
# 3. データセットとローダー (Dataset & DataLoader)
# ==========================================
class MahjongSupervisedDataset(Dataset):
    def __init__(self, h5_path, is_train=False):
        self.h5_path = h5_path
        self.is_train = is_train
        self.dataset_file = None
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f['state_2d'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.dataset_file is None:
            self.dataset_file = h5py.File(self.h5_path, 'r')
            
        state_2d = torch.from_numpy(self.dataset_file['state_2d'][idx]).float()
        cond_vec = torch.from_numpy(self.dataset_file['cond_vec'][idx]).float()
        seq_hist = torch.tensor(self.dataset_file['seq_hist'][idx], dtype=torch.long)
        
        t_disc = torch.tensor(self.dataset_file['target_discards'][idx], dtype=torch.long)
        t_act = torch.tensor(self.dataset_file['target_actions'][idx], dtype=torch.long)
        m_disc = torch.tensor(self.dataset_file['mask_discards'][idx], dtype=torch.float32)
        m_act = torch.tensor(self.dataset_file['mask_actions'][idx], dtype=torch.float32)
        
        t_score = torch.tensor(self.dataset_file['target_score'][idx], dtype=torch.float32)
        t_tenpai = torch.from_numpy(self.dataset_file['target_tenpai'][idx]).float()
        t_danger = torch.from_numpy(self.dataset_file['target_danger'][idx]).float()
        t_waits = torch.from_numpy(self.dataset_file['target_waits'][idx]).float() 
        
        if self.is_train and m_disc.item() == 1.0:
            state_2d, t_disc = apply_dynamic_augmentation_2d(state_2d, t_disc.item())
            t_disc = torch.tensor(t_disc, dtype=torch.long)
            
        return {
            'state_2d': state_2d, 'cond_vec': cond_vec, 'seq_hist': seq_hist,
            'target_discard': t_disc, 'target_action': t_act, 'm_disc': m_disc, 'm_act': m_act,
            'target_value': t_score, 'target_tenpai': t_tenpai, 'target_danger': t_danger,
            'target_waits': t_waits
        }

# ==========================================
# 4. カスタム・オプティマイザ (Custom Optimizer)
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
# 5. マルチタスク訓練ループ (Training Pipeline)
# ==========================================
def directml_safe_bce_with_logits(logits, targets):
    probs = torch.sigmoid(logits)
    probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
    return -(targets * torch.log(probs) + (1.0 - targets) * torch.log(1.0 - probs)).mean()

def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('V2 Supervised Training Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_curve_sl_v2.png")
    print(" -> [Info] 訓練グラフを保存しました: training_curve_sl_v2.png")
    plt.close()

def train_supervised_multitask(h5_train_path, h5_val_path, epochs=50, batch_size=256, accumulation_steps=4, lr=1e-4, patience=4):
    if torch_directml.is_available():
        device = torch_directml.device()
        print(f"訓練デバイス (Device): DirectML - {torch_directml.device_name(0)}")
    else:
        device = torch.device('cpu')
        print("訓練デバイス (Device): CPU")
        
    train_dataset = MahjongSupervisedDataset(h5_train_path, is_train=True)
    val_dataset = MahjongSupervisedDataset(h5_val_path, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = SmartMahjongMultiTaskNet(input_channels=256, num_blocks=18, dropout_p=0.30).to(device)
    optimizer = DirectMLSafeAdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    warmup_epochs = 2
    scheduler_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=(epochs - warmup_epochs), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])
    
    ce_loss_smooth = nn.CrossEntropyLoss(label_smoothing=0.05, reduction='none')
    mse_loss = nn.MSELoss()
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    lambda_policy, lambda_value, lambda_tenpai, lambda_danger, lambda_waits = 1.0, 0.5, 0.5, 1.0, 1.0
    
    history = {'train_loss': [], 'val_loss': []}
    
    # 0: PASS->45, 1: CHI->34, 2: PON->37, 3: KAN->39, 4: RIICHI->41, 5: HORA->42
    act_map = torch.tensor([45, 34, 37, 39, 41, 42], device=device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch+1}/{epochs}] Train", leave=False)
        for i, batch in train_pbar:
            state_2d = batch['state_2d'].to(device)
            cond_vec = batch['cond_vec'].to(device)
            seq_hist = batch['seq_hist'].to(device)
            
            t_disc = batch['target_discard'].to(device)
            t_act = batch['target_action'].to(device)
            m_disc = batch['m_disc'].to(device)
            m_act = batch['m_act'].to(device)
            t_score = batch['target_value'].to(device)
            
            t_tenpai = batch['target_tenpai'].to(device)
            t_danger = batch['target_danger'].to(device)
            t_waits = batch['target_waits'].to(device) 
            
            t_target_47 = torch.where(m_disc == 1.0, t_disc, act_map[t_act])
            valid_mask = torch.clamp(m_disc + m_act, 0.0, 1.0)
            
            p_out, v_score, a_tenpai, a_danger, a_waits = model(state_2d, cond_vec, seq_hist, rl_mode=False)
            
            p_out_masked = p_out + (1.0 - valid_mask.unsqueeze(1)) * -1e9
            loss_policy = (ce_loss_smooth(p_out_masked, t_target_47) * valid_mask).mean()
            loss_value = mse_loss(v_score.squeeze(-1), t_score)
            
            loss_aux_tenpai = directml_safe_bce_with_logits(a_tenpai, t_tenpai)
            loss_aux_danger = directml_safe_bce_with_logits(a_danger, t_danger)
            loss_aux_waits = directml_safe_bce_with_logits(a_waits, t_waits) 
            
            total_loss = (
                lambda_policy * loss_policy + 
                lambda_value * loss_value + 
                lambda_tenpai * loss_aux_tenpai + 
                lambda_danger * loss_aux_danger +
                lambda_waits * loss_aux_waits
            )
            
            loss_to_backward = total_loss / accumulation_steps
            loss_to_backward.backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            train_loss += total_loss.item()
            if i % 50 == 0:
                train_pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- 検証フェーズ (Validation Phase) ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Val", leave=False)
            for batch in val_pbar:
                state_2d = batch['state_2d'].to(device)
                cond_vec = batch['cond_vec'].to(device)
                seq_hist = batch['seq_hist'].to(device)
                
                t_disc = batch['target_discard'].to(device)
                t_act = batch['target_action'].to(device)
                m_disc = batch['m_disc'].to(device)
                m_act = batch['m_act'].to(device)
                t_score = batch['target_value'].to(device)
                
                t_tenpai = batch['target_tenpai'].to(device)
                t_danger = batch['target_danger'].to(device)
                t_waits = batch['target_waits'].to(device)
                
                t_target_47 = torch.where(m_disc == 1.0, t_disc, act_map[t_act])
                valid_mask = torch.clamp(m_disc + m_act, 0.0, 1.0)
                
                p_out, v_score, a_tenpai, a_danger, a_waits = model(state_2d, cond_vec, seq_hist, rl_mode=False)
                
                p_out_masked = p_out + (1.0 - valid_mask.unsqueeze(1)) * -1e9
                loss_policy = (ce_loss_smooth(p_out_masked, t_target_47) * valid_mask).mean()
                loss_value = mse_loss(v_score.squeeze(-1), t_score)

                loss_aux_tenpai = directml_safe_bce_with_logits(a_tenpai, t_tenpai)
                loss_aux_danger = directml_safe_bce_with_logits(a_danger, t_danger)
                loss_aux_waits = directml_safe_bce_with_logits(a_waits, t_waits)

                total_loss = (
                    lambda_policy * loss_policy + 
                    lambda_value * loss_value + 
                    lambda_tenpai * loss_aux_tenpai + 
                    lambda_danger * loss_aux_danger +
                    lambda_waits * loss_aux_waits
                )

                val_loss += total_loss.item()
                val_pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})
                
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6e}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            save_path = "smart_mahjong_base_policy_v2.pth"
            torch.save(model.state_dict(), save_path)
            print(f" -> [Update] ベースポリシー保存: {save_path}")
        else:
            early_stop_counter += 1
            print(f" -> [Info] Val Loss が改善されません ({early_stop_counter}/{patience})")
            if early_stop_counter >= patience:
                print(f"*** Early Stopping Triggered! {epoch+1} エポック目で訓練を早期終了します ***")
                break

    plot_training_history(history)

if __name__ == "__main__":
    train_h5 = "data/train_dataset.h5"
    val_h5 = "data/val_dataset.h5"
    if os.path.exists(train_h5) and os.path.exists(val_h5):
        train_supervised_multitask(train_h5, val_h5, epochs=50, batch_size=256, accumulation_steps=4, lr=1e-4, patience=4)
    else:
        print("データセットが見つかりません。(Dataset not found.)")