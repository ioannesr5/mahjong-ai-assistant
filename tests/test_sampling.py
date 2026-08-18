"""方策サンプリングの回帰テスト (PPO の重要度比が壊れないことを保証する)"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from models import directml_safe_masked_sample, masked_log_probs

try:
    import torch_directml

    DEVICES = [torch.device("cpu")]
    if torch_directml.is_available():
        DEVICES.append(torch_directml.device())
except ImportError:  # pragma: no cover
    DEVICES = [torch.device("cpu")]


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: str(d))
def test_sampler_never_returns_a_masked_action(device):
    """
    回帰テスト: torch.distributions.Categorical は DirectML 上でマスク済みの
    非合法アクションを返すことがあり、返り値の log_prob が
    log(float32 eps) = -15.94 に張り付いていた。
    その結果ロールアウトの行動が方策からのサンプルでなくなり、
    PPO の重要度比が第0エポックからクリップ上限 (exp(5) ≈ 148) に飽和していた。
    """
    torch.manual_seed(0)
    logits = torch.randn(512, 54, device=device) * 3.0
    mask = (torch.rand(512, 54, device=device) < 0.15).float()
    mask[:, 0] = 1.0  # 必ず 1 つは合法にする

    actions, log_probs = directml_safe_masked_sample(logits, mask)
    chosen_legal = mask.gather(1, actions.unsqueeze(1)).squeeze(1)
    assert int((chosen_legal == 0).sum()) == 0, "非合法アクションがサンプルされた"
    assert float(log_probs.min()) > -30.0, "log_prob がクリップ値に張り付いている"


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: str(d))
def test_sampling_log_prob_matches_the_update_path(device):
    """
    サンプリング時に記録する log_prob と、更新時に再計算する log_prob が
    同一の式・同一の値であること (= 第0エポックの重要度比が 1.0 になること)。
    """
    torch.manual_seed(1)
    logits = torch.randn(256, 54, device=device)
    mask = (torch.rand(256, 54, device=device) < 0.3).float()
    mask[:, 0] = 1.0

    actions, sampled_log_probs = directml_safe_masked_sample(logits, mask)

    # 更新側と同じ経路で再計算する
    log_probs_all, _ = masked_log_probs(logits, mask)
    classes = torch.arange(54, device=device).unsqueeze(0)
    one_hot = (actions.unsqueeze(1) == classes).to(torch.float32)
    recomputed = (log_probs_all * one_hot).sum(dim=-1)

    ratio = torch.exp(recomputed - sampled_log_probs)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5), (
        f"重要度比が 1.0 になっていません: mean={float(ratio.mean()):.4f}"
    )


def test_masked_log_probs_suppresses_illegal_actions():
    logits = torch.zeros(1, 54)
    mask = torch.zeros(1, 54)
    mask[0, [3, 7]] = 1.0
    log_probs, _ = masked_log_probs(logits, mask)
    probs = log_probs.exp()
    assert probs[0, 3] == pytest.approx(0.5, abs=1e-4)
    assert probs[0, 7] == pytest.approx(0.5, abs=1e-4)
    assert float(probs[0, [i for i in range(54) if i not in (3, 7)]].max()) < 1e-6


def test_sampler_distribution_is_unbiased():
    """Gumbel-max が正しい多項分布を再現していること"""
    torch.manual_seed(2)
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    mask = torch.ones(1, 4)
    target = F.softmax(logits, dim=-1)[0].numpy()

    counts = np.zeros(4)
    batch = logits.repeat(40000, 1)
    actions, _ = directml_safe_masked_sample(batch, mask.repeat(40000, 1))
    for a in actions.numpy():
        counts[a] += 1
    empirical = counts / counts.sum()
    assert np.abs(empirical - target).max() < 0.01, f"{empirical} vs {target}"
