"""アクション空間の意味論が pymahjong と一致していることを検証する"""

import pymahjong as pm
import pytest
from pymahjong import MahjongEnv

import actions as A


def test_action_dim_matches_env():
    assert A.N_ACTIONS == MahjongEnv.ACTION_DIM == 54


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("CHILEFT", A.CHI_LEFT),
        ("CHIMIDDLE", A.CHI_MIDDLE),
        ("CHIRIGHT", A.CHI_RIGHT),
        ("CHILEFT_USERED", A.CHI_LEFT_RED),
        ("CHIMIDDLE_USERED", A.CHI_MIDDLE_RED),
        ("CHIRIGHT_USERED", A.CHI_RIGHT_RED),
        ("PON", A.PON),
        ("PON_USERED", A.PON_RED),
        ("ANKAN", A.ANKAN),
        ("MINKAN", A.MINKAN),
        ("KAKAN", A.KAKAN),
        ("RIICHI", A.RIICHI),
        ("RON", A.RON),
        ("TSUMO", A.TSUMO),
        ("PUSH", A.KYUSHUKYUHAI),
        ("PASS_RIICHI", A.PASS_RIICHI),
        ("PASS_RESPONSE", A.PASS_RESPONSE),
    ],
)
def test_constant_matches_pymahjong(name, expected):
    assert getattr(MahjongEnv, name) == expected


def test_action_type_table_matches_names():
    """ACTION_TYPES[i] の BaseAction 種別が ACTION_NAMES と矛盾しないこと"""
    types = MahjongEnv.ACTION_TYPES
    for action in range(A.N_ACTIONS):
        base = types[action]
        name = A.ACTION_NAMES[action]
        if action < 37:
            assert base == pm.BaseAction.Discard, name
        elif action in A.CHI_ACTIONS:
            assert base == pm.BaseAction.Chi, name
        elif action in A.PON_ACTIONS:
            assert base == pm.BaseAction.Pon, name
        elif action == A.ANKAN:
            assert base == pm.BaseAction.AnKan, name
        elif action == A.MINKAN:
            assert base == pm.BaseAction.Kan, name
        elif action == A.KAKAN:
            assert base == pm.BaseAction.KaKan, name
        elif action == A.RIICHI:
            assert base == pm.BaseAction.Riichi, name
        elif action == A.RON:
            assert base == pm.BaseAction.Ron, name
        elif action == A.TSUMO:
            assert base == pm.BaseAction.Tsumo, name
        elif action == A.KYUSHUKYUHAI:
            assert base == pm.BaseAction.Kyushukyuhai, name
        else:
            assert base == pm.BaseAction.Pass, name


def test_every_action_has_a_name():
    assert set(A.ACTION_NAMES) == set(range(A.N_ACTIONS))


def test_legacy47_mapping_is_injective():
    """旧 47 次元 -> 54 次元の写像に重複が無いこと (重みの取り違えを防ぐ)"""
    values = list(A.LEGACY47_TO_54.values())
    assert len(values) == len(set(values))
    assert len(A.LEGACY47_TO_54) == A.Legacy47.N_ACTIONS


def test_legacy_declaration_targets():
    """かつて誤っていた 3 箇所を固定する回帰テスト"""
    assert A.LEGACY47_TO_54[A.Legacy47.TSUMO] == A.TSUMO
    assert A.LEGACY47_TO_54[A.Legacy47.KYUSHUKYUHAI] == A.KYUSHUKYUHAI
    assert A.LEGACY47_TO_54[A.Legacy47.PASS_RIICHI] == A.PASS_RIICHI
    assert A.LEGACY47_TO_54[A.Legacy47.RON] == A.RON
