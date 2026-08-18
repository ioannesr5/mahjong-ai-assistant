"""
ネットワーク定義とカスタム最適化器の唯一の定義場所 (Single definition site)

【リファクタリング】従来 supervised_trainer.py と rl_ppo_trainer.py に
同じモデル定義がコピーされており、片方だけ変更されると SL と RL で
別のネットワークを使う事故が起こり得た。ここに集約する。

アーキテクチャは凍結対象であり、本ファイルの内容は移設前と一字一句同一である:
  256x4x9 入力 -> 18 x FiLM-ResBlock(256ch) -> 1024 latent
  捨て牌系列 (72 token, vocab 273) -> 4 層 Transformer (embed 256)
  Cross-Attention 融合 -> policy(54) / value(1) / tenpai(3) / danger(102) / waits(102)
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.optimizer import Optimizer


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


def directml_safe_bce_with_logits(logits, targets):
    probs = torch.sigmoid(logits)
    probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
    return -(targets * torch.log(probs) + (1.0 - targets) * torch.log(1.0 - probs)).mean()


def masked_log_probs(logits, mask, neg_inf=-1e9):
    """合法手マスクを適用した log-softmax。サンプリング側と更新側で必ずこれを使う。"""
    masked = logits + (1.0 - mask) * neg_inf
    return torch.log(F.softmax(masked, dim=-1) + 1e-8), masked


def directml_safe_masked_sample(logits, mask, generator=None):
    """
    合法手マスク付きのカテゴリカル・サンプリング (Gumbel-max 法)。

    【重大な修正】torch.distributions.Categorical は DirectML バックエンドでは
    多項サンプリングが正しく動かず、**マスクで禁止したアクションを返す**ことがある。
    実測では返された行動の log_prob が log(1e-7) ≈ -15.94 に張り付き、
    PPO の重要度比が第0エポックから exp(5) のクリップ上限 (≈148) に飽和していた。
    つまりロールアウトの行動は「方策からのサンプル」になっていなかった。

    Gumbel-max は elementwise 演算と argmax だけで構成されるため DirectML でも安全。
    マスクされた要素は -1e9 のままなので選ばれることはない。

    Returns:
        (actions, log_probs) — log_probs は更新側とまったく同じ式で計算した値
    """
    log_probs, _ = masked_log_probs(logits, mask)
    # 乱数は CPU で作って転送する (デバイス側 RNG の実装差を避けるため)
    uniform = torch.rand(log_probs.shape, generator=generator, device="cpu")
    uniform = uniform.clamp_(1e-20, 1.0).to(log_probs.device)
    gumbel = -torch.log(-torch.log(uniform))
    actions = torch.argmax(log_probs + gumbel, dim=-1)
    chosen = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    return actions, chosen
