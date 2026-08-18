"""自己対局環境ラッパーの追跡・半荘連続性・報酬の回帰テスト"""

import numpy as np
import pytest

import actions as A
from feature_extractor import DECISION_DISCARD, DECISION_RESPONSE, DECISION_RIICHI
from mahjong_env import TILE_ID_TO_STR, MultiAgentMahjongEnvWrapper, tiles_from_obs
from reward import RewardConfig, hanchan_rank_bonus, rank_of


@pytest.fixture(scope="module")
def rollout():
    """ランダム打ちで数半荘まわし、各ステップで追跡の整合性を検証した結果を集める"""
    env = MultiAgentMahjongEnvWrapper()
    state, mask, _ = env.reset_hanchan()
    rng = np.random.default_rng(20260818)
    results = {
        "hanchan": [],
        "decision_types": set(),
        "meld_types": set(),
        "states": [],
        "kyoku_counts": [],
        "reconcile_added": 0,
    }
    for _ in range(60000):
        env.assert_tracking_matches_observation()
        results["decision_types"].add(env._decision_type())
        if len(results["states"]) < 200:
            results["states"].append(state["state_2d"])
        action = int(rng.choice(np.nonzero(mask)[0]))
        state, mask, _r, done, _p, info = env.step(action, strict=True)
        for seat in env.seats:
            results["meld_types"].update(seat.meld_types)
        if done:
            results["hanchan"].append(info["final_scores"])
            results["kyoku_counts"].append(env.kyoku_count)
            results["reconcile_added"] += env.reconcile_added
            state, mask, _ = env.reset_hanchan()
            if len(results["hanchan"]) >= 8:
                break
    return results


def test_scores_are_conserved_across_the_hanchan(rollout):
    """
    回帰テスト: 旧実装は毎局 env.reset() を引数なしで呼び、点数が 25000 に戻り
    立直棒が消滅していたため 4 家の合計点が保存されなかった。
    (「平均順位 1.94 なのに平均素点 -355」という矛盾の原因)
    """
    assert rollout["hanchan"], "半荘が 1 つも終わっていない"
    for scores in rollout["hanchan"]:
        assert sum(scores) == 100000, scores


def test_hanchan_runs_through_south_round(rollout):
    """東 1 局〜南 4 局 (連荘を含む) まで進むこと。旧実装は常に東場のままだった。"""
    assert min(rollout["kyoku_counts"]) >= 8


def test_discard_and_response_decisions_are_exposed(rollout):
    # 立直はランダム打ちでは門前聴牌に到達しにくく数半荘では現れないことがあるため、
    # 判定ロジック自体は test_decision_type_mapping で直接検証する。
    assert {DECISION_DISCARD, DECISION_RESPONSE} <= rollout["decision_types"]


def test_decision_type_mapping():
    """合法手の集合から決定タイプが正しく決まること"""
    env = MultiAgentMahjongEnvWrapper()
    env.reset_hanchan()
    original = env.env.get_valid_actions

    def patch(actions):
        env.env.get_valid_actions = lambda: list(actions)

    try:
        patch([A.RIICHI, A.PASS_RIICHI])
        assert env._decision_type() == DECISION_RIICHI
        assert env._in_riichi_stage2()

        patch([A.PON, A.PASS_RESPONSE])
        assert env._decision_type() == DECISION_RESPONSE
        assert not env._in_riichi_stage2()

        patch([0, 1, 2])
        assert env._decision_type() == DECISION_DISCARD
    finally:
        env.env.get_valid_actions = original


def test_meld_types_are_classified(rollout):
    """副露は観測との突き合わせで種別まで復元される (旧実装は全て 'pon' 扱いだった)"""
    assert {"chi", "pon"} <= rollout["meld_types"]


def test_state_uses_the_same_schema_as_the_supervised_pipeline(rollout):
    """
    RL と SL は同一の MahjongFeatureExtractor256 を通るので、
    未使用チャネルや形状が食い違うことは原理的に起こらない。
    """
    for state in rollout["states"][:50]:
        assert state.shape == (256, 4, 9)
        assert not state[225:].any(), "未使用チャネルに値が書かれている"


def test_context_channels_are_populated(rollout):
    """
    回帰テスト: 旧 decode_obs_93_to_256 は本場/供託/残り牌/オーラス (213-215, 220) を
    一切埋めず、点差 (216-219) も絶対座席で書いていた。
    """
    flat = np.stack([s.reshape(256, 36) for s in rollout["states"]])
    tiles_left = flat[:, 215, 0]
    assert tiles_left.max() > 0.0
    assert tiles_left.min() < tiles_left.max(), "残り牌数が変化していない (巡目が見えていない)"
    # 自分との点差 (216) は常に 0
    assert np.allclose(flat[:, 216, 0], 0.0)


def test_hand_reconstruction_matches_observation():
    env = MultiAgentMahjongEnvWrapper()
    env.reset_hanchan()
    obs = env.env.get_obs(env.current_player)
    hand = tiles_from_obs(obs)
    assert len(hand) in (13, 14)
    for tile in hand:
        base = tile[:2] if tile.endswith("r") else tile
        assert base in TILE_ID_TO_STR


def test_illegal_action_raises_in_strict_mode():
    """
    回帰テスト: 旧実装は非合法手で done=True を返し、軌跡を偽の終端で切っていた
    (マスクのバグを隠してしまう)。
    """
    env = MultiAgentMahjongEnvWrapper()
    _, mask, _ = env.reset_hanchan()
    illegal = int(np.nonzero(mask == 0)[0][0])
    with pytest.raises(AssertionError):
        env.step(illegal, strict=True)


# ------------------------------------------------------------------ 報酬
def test_rank_bonus_ordering():
    config = RewardConfig()
    bonuses = [hanchan_rank_bonus(config, s)[0] for s in (
        [40000, 20000, 20000, 20000],
        [30000, 40000, 15000, 15000],
        [20000, 40000, 30000, 10000],
        [10000, 40000, 30000, 20000],
    )]
    assert bonuses == sorted(bonuses, reverse=True)
    assert bonuses[0] == config.rank_bonus[0]
    assert bonuses[3] == config.rank_bonus[3]


def test_rank_of_handles_ties():
    assert rank_of([25000, 25000, 25000, 25000], seat=0) == 0
    assert rank_of([25000, 30000, 25000, 25000], seat=0) == 1


def test_ukeire_is_phase_gated():
    """回帰テスト: 受入れシェーピングに phase ゲートが無く、Phase 3 でも効いていた"""
    config = RewardConfig()
    assert config.ukeire_scale(1) > 0
    assert config.ukeire_scale(3) == 0.0
    assert config.shanten_scale(3) == 0.0


def test_no_unconditional_riichi_bonus():
    """
    回帰テスト: 旧実装は action_id == 46 (実際は MINKAN) に +0.1 を与えていた。
    立直の無条件ボーナスは報酬設定から完全に削除されている。
    """
    config = RewardConfig()
    assert "riichi" not in config.as_dict()
    assert A.RIICHI == 48 and A.MINKAN == 46
