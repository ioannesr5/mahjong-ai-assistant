"""
方策行動プローブ (Policy Behavior Probe)

チェックポイントを自己対局環境に流し込み、「和了できる場面で実際に和了するか」
「鳴ける場面でどれだけ鳴くか」を実測する診断スクリプト。
修正方案.md の §1.3 / 阶段1 验收标准 で使用する。

usage:
    .venv/Scripts/python.exe probe_policy_behavior.py <checkpoint.pth> [steps]
"""

import collections
import sys
import types

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# rl_ppo_trainer.py の __main__ ブロックを実行せずにモジュールとして読み込む
_src = open("rl_ppo_trainer.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0]
rl = types.ModuleType("rl")
rl.__dict__["__name__"] = "rl"
exec(compile(_src, "rl_ppo_trainer.py", "exec"), rl.__dict__)

# pymahjong 54次元アクション空間 (env_pymahjong.py の定数と一致)
RIICHI, RON, TSUMO = 48, 49, 50
GROUPS = {
    "chi": range(37, 43),
    "pon": range(43, 45),
    "kan": range(45, 48),
    "riichi": [RIICHI],
    "ron": [RON],
    "tsumo": [TSUMO],
    "pass": [52, 53],
}


def probe(ckpt_path: str, max_steps: int = 6000):
    env = rl.MultiAgentMahjongEnvWrapper()
    model = rl.SmartMahjongMultiTaskNet(input_channels=256, num_blocks=18)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(rl.adapt_policy_state_dict(sd))
    model.eval()

    state, mask, player = env.reset_hanchan()
    avail: collections.Counter = collections.Counter()
    taken: collections.Counter = collections.Counter()
    p0_steps = 0
    missed_agari = []

    for _ in range(max_steps):
        with torch.no_grad():
            p_out, _, _, _, _ = model(
                torch.tensor(state["state_2d"]).unsqueeze(0),
                torch.tensor(state["cond_vec"]).unsqueeze(0),
                torch.tensor(state["seq_hist"]).unsqueeze(0),
                True,
            )
        masked = p_out + (1.0 - torch.tensor(mask)) * -1e9
        action = int(masked.argmax())

        if player == 0:
            p0_steps += 1
            for name, ids in GROUPS.items():
                if any(mask[i] > 0 for i in ids):
                    avail[name] += 1
                    if action in ids:
                        taken[name] += 1
                    elif name in ("ron", "tsumo") and len(missed_agari) < 10:
                        missed_agari.append(
                            (name, action, float(masked[0, ids[0]]), float(masked[0, action]))
                        )

        state, mask, _r, done, player, _info = env.step(action)
        if done:
            state, mask, player = env.reset_hanchan()

    print(f"checkpoint : {ckpt_path}")
    print(f"p0 決定ステップ数: {p0_steps}\n")
    print(f"{'action':<8}{'avail':>8}{'taken':>8}{'rate':>10}")
    for name in GROUPS:
        a, t = avail[name], taken[name]
        rate = f"{t / a:.1%}" if a else "-"
        print(f"{name:<8}{a:>8}{t:>8}{rate:>10}")

    call_avail = sum(avail[k] for k in ("chi", "pon", "kan"))
    call_taken = sum(taken[k] for k in ("chi", "pon", "kan"))
    if call_avail:
        print(f"\n鳴き命中率 (目標 25~40%): {call_taken / call_avail:.1%}")
    agari_avail = avail["ron"] + avail["tsumo"]
    if agari_avail:
        print(f"和了実行率 (目標 >95%): {(taken['ron'] + taken['tsumo']) / agari_avail:.1%}")
    else:
        print("\n⚠️ 和了機会が一度も発生しなかった（門前率≈0 の可能性）")
    for m in missed_agari:
        print(f"  見逃し: {m[0]} logit={m[2]:.3f} → 選択 action={m[1]} logit={m[3]:.3f}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "smart_mahjong_ppo_TRUE_BEST_phase2.pth"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    np.random.seed(0)
    torch.manual_seed(0)
    probe(path, steps)
