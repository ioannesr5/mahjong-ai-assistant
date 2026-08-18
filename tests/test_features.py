"""牌解析・特徴量・保存形式・データ拡張の一貫性テスト"""

import glob

import numpy as np
import pytest
import torch
from mahjong.shanten import Shanten

import actions as A
from feature_extractor import (
    DECISION_RESPONSE,
    SEQ_PAD_TOKEN,
    MahjongFeatureExtractor256,
    MahjongGameState,
    PlayerState,
    UnknownTileError,
    encode_discard_sequence,
    encode_discard_token,
    parse_tile,
)
from mjai_parser import (
    MjaiReplayParser,
    available_chi_actions,
    available_pon_kan_actions,
    shanten_with_melds,
)
from state_codec import pack_state, unpack_state, unpack_state_batch
from supervised_trainer import SuitPermutationAugmenter, suit_permutation_tables

# ---------------------------------------------------------------- parse_tile
ALL_MJAI_TILES = (
    [f"{n}{s}" for s in "mps" for n in range(1, 10)] + ["E", "S", "W", "N", "P", "F", "C"]
)


def test_all_tiles_map_to_distinct_ids():
    ids = [parse_tile(t)[0] for t in ALL_MJAI_TILES]
    assert sorted(ids) == list(range(34))


def test_honor_tiles_are_not_collapsed_to_1m():
    """回帰テスト: 字牌が全て 1m (ID 0) に潰れていた重大バグ"""
    for tile, expected in [("E", 27), ("S", 28), ("W", 29), ("N", 30), ("P", 31), ("F", 32), ("C", 33)]:
        assert parse_tile(tile) == (expected, False), tile


def test_red_tiles():
    assert parse_tile("5mr") == (4, True)
    assert parse_tile("0p") == (13, True)
    assert parse_tile("5sr") == (22, True)


def test_unknown_tile_raises_instead_of_silently_returning_1m():
    with pytest.raises(UnknownTileError):
        parse_tile("?")
    assert parse_tile("?", strict=False) == (0, False)


# ------------------------------------------------------------- seq_hist spec
def test_token_range_fits_embedding_vocab():
    assert encode_discard_token(33, 3, False) == 271
    assert SEQ_PAD_TOKEN == 272


def test_sequence_keeps_most_recent_events():
    events = [(i % 4, "1m", False) for i in range(100)]
    seq = encode_discard_sequence(events, self_seat=0, max_len=72)
    assert (seq != SEQ_PAD_TOKEN).sum() == 72
    # 直近 72 手が入っている: 先頭は 100-72 = 28 番目のイベント
    assert seq[0] == encode_discard_token(0, (28 % 4 - 0) % 4, False)


def test_sequence_encodes_relative_actor_and_cut_type():
    events = [(2, "5p", True), (3, "E", False)]
    seq = encode_discard_sequence(events, self_seat=1, max_len=4)
    assert seq[0] == encode_discard_token(13, 1, True)
    assert seq[1] == encode_discard_token(27, 2, False)
    assert seq[2] == SEQ_PAD_TOKEN


# --------------------------------------------------------------- state codec
def _sample_state():
    players = [PlayerState(seat=i) for i in range(4)]
    players[1].discards = ["1m", "E", "5sr"]
    players[1].is_tsumogiri = [False, True, False]
    players[2].melds = [["2p", "3p", "4p"]]
    players[2].meld_types = ["chi"]
    gs = MahjongGameState(
        self_seat=0,
        players=players,
        closed_hand=["1m", "1m", "9s", "E", "E", "C", "5mr", "3p", "3p", "3p", "7s", "8s", "9s"],
        dora_indicators=["9m"],
        round_wind=1,
        self_wind=2,
        honba=3,
        kyotaku=1,
        tiles_left=42,
        is_all_last=True,
        discard_events=[(1, "1m", False), (1, "E", True)],
        decision_type=DECISION_RESPONSE,
        last_action_tile="5sr",
        last_action_actor=3,
        drawn_tile="7s",
    )
    return MahjongFeatureExtractor256().extract(gs)


def test_pack_unpack_roundtrip_is_exact():
    feats = _sample_state()
    restored = unpack_state(*pack_state(feats["state_2d"]))
    assert np.array_equal(restored, feats["state_2d"])


def test_batch_unpack_matches_single():
    feats = _sample_state()
    packed = pack_state(feats["state_2d"])
    batch = unpack_state_batch(
        packed[0][None], packed[1][None], packed[2][None]
    )
    assert np.array_equal(batch[0], feats["state_2d"])


def test_unused_channels_stay_zero():
    feats = _sample_state()
    assert not feats["state_2d"][225:].any()


def test_decision_context_channels_are_populated():
    feats = _sample_state()
    state = feats["state_2d"].reshape(256, 36)
    assert state[221, 22] == 1.0  # 5s が鳴き対象
    assert np.isclose(state[222, 0], (3 + 1) / 4)  # rel_actor=3 -> (3+1)/4
    assert np.isclose(state[223, 0], DECISION_RESPONSE / 2.0)
    assert state[224, 24] == 1.0  # 7s をツモ


# ------------------------------------------------------------ shanten / calls
def test_shanten_accounts_for_melds():
    """回帰テスト: 副露手で門前部分 (10 枚) だけ渡して向聴数を誤っていたバグ"""
    calculator = Shanten()
    closed = ["1m", "1m", "1m", "2p", "3p", "4p", "5s", "6s", "7s", "E"]
    melds = [["9p", "9p", "9p"]]
    # 副露 1 つ + 面子3 + 単騎 E = 聴牌
    assert shanten_with_melds(calculator, closed, melds) == 0
    # 副露を渡さなければ枚数が 13/14 にならず None が返る (旧コードはここを素通ししていた)
    assert shanten_with_melds(calculator, closed, []) is None


def test_chi_detection():
    hand = ["2m", "3m", "7p", "8p"]
    assert A.CHI_LEFT in available_chi_actions(hand, "1m")  # 1m + 2m3m
    assert A.CHI_RIGHT in available_chi_actions(hand, "4m")  # 2m3m + 4m
    assert available_chi_actions(hand, "E") == []  # 字牌はチー不可


def test_chi_red_variant_detected():
    hand = ["5mr", "6m"]
    result = available_chi_actions(hand, "4m")
    assert A.CHI_LEFT in result
    assert A.CHI_LEFT_RED in result


def test_pon_and_kan_detection():
    assert available_pon_kan_actions(["5p", "5p"], "5p") == [A.PON]
    result = available_pon_kan_actions(["5p", "5p", "5p"], "5p")
    assert A.PON in result and A.MINKAN in result
    assert available_pon_kan_actions(["5p"], "5p") == []


# ---------------------------------------------------------------- 拡張の整合性
def test_suit_permutation_tables_are_inverses():
    for perm in ([1, 2, 0], [2, 0, 1], [1, 0, 2]):
        forward, inverse = suit_permutation_tables(np.array(perm))
        assert np.array_equal(inverse[forward], np.arange(34))
        assert np.array_equal(forward[27:], np.arange(27, 34))  # 字牌は不変


class _FixedPermRng:
    def __init__(self, perm):
        self._perm = np.array(perm)

    def permutation(self, n):
        assert n == 3
        return self._perm


@pytest.mark.parametrize("perm", [[1, 2, 0], [2, 0, 1], [1, 0, 2], [0, 2, 1]])
def test_augmentation_keeps_state_and_labels_consistent(perm):
    """
    回帰テスト: 3 巡回の置換で赤ドラチャネルがずれていたバグ、および
    waits / danger / legal_mask / seq_hist がラベルだけ置換されずに残っていたバグ。
    """
    feats = _sample_state()
    state = torch.from_numpy(feats["state_2d"].copy())
    seq = torch.from_numpy(feats["seq_hist"].copy())

    action = 4  # 5m 打牌
    legal = torch.zeros(A.N_ACTIONS)
    legal[[0, 4, 26, 27, 33]] = 1.0
    legal[34] = 1.0  # 赤5m 打牌も合法
    waits = torch.zeros(102)
    waits[0 * 34 + 4] = 1.0  # 下家は 5m 待ち
    danger = waits.clone()

    aug = SuitPermutationAugmenter(_FixedPermRng(perm))
    new_state, new_action, new_legal, new_waits, new_danger, new_seq = aug(
        state, action, legal, waits, danger, seq
    )

    _, inverse = suit_permutation_tables(np.array(perm))
    new_suit_of_m = int(inverse[0]) // 9

    # 1) 正解ラベルは 5m -> 置換後の花色の 5
    assert new_action == new_suit_of_m * 9 + 4
    # 2) 合法手も同じ写像で移動している
    assert new_legal[new_action] == 1.0
    assert new_legal[34 + new_suit_of_m] == 1.0
    # 3) 待ち牌 / 危険度も同期している
    assert new_waits[new_suit_of_m * 9 + 4] == 1.0
    assert new_danger[new_suit_of_m * 9 + 4] == 1.0
    assert new_waits.sum() == 1.0
    # 4) 赤ドラチャネルが正しい花色に移っている (手牌に赤5m がある)
    flat = new_state.reshape(256, 36)
    assert flat[4 + new_suit_of_m, new_suit_of_m * 9 + 4] == 1.0
    assert flat[4 + new_suit_of_m].sum() == 1.0
    # 5) 系列トークンの牌 ID も同期している (元の系列に 1m が含まれる)
    old_ids = torch.div(seq[seq != SEQ_PAD_TOKEN], 8, rounding_mode="floor")
    new_ids = torch.div(new_seq[new_seq != SEQ_PAD_TOKEN], 8, rounding_mode="floor")
    assert torch.equal(new_ids, torch.from_numpy(inverse)[old_ids])
    # 6) 字牌は動かない
    assert new_legal[27] == 1.0 and new_legal[33] == 1.0


def test_augmentation_identity_permutation_is_noop():
    feats = _sample_state()
    state = torch.from_numpy(feats["state_2d"].copy())
    seq = torch.from_numpy(feats["seq_hist"].copy())
    legal = torch.zeros(A.N_ACTIONS)
    legal[4] = 1.0
    aug = SuitPermutationAugmenter(_FixedPermRng([0, 1, 2]))
    out = aug(state, 4, legal, torch.zeros(102), torch.zeros(102), seq)
    assert torch.equal(out[0], state)
    assert out[1] == 4
    assert torch.equal(out[5], seq)


# ------------------------------------------------------------ パーサ健全性
def _replay_paths(count):
    return sorted(glob.glob("data/logs/**/*.mjson", recursive=True))[:count]


@pytest.mark.skipif(not _replay_paths(1), reason="牌譜が無い環境ではスキップ")
def test_parser_produces_all_decision_types_and_valid_labels():
    parser = MjaiReplayParser(MahjongFeatureExtractor256())
    seen_actions = set()
    seen_decisions = set()
    for path in _replay_paths(12):
        for sample in parser.parse_file(path):
            assert 0 <= sample.action < A.N_ACTIONS
            # 正解は必ず合法手マスクに含まれていること
            assert sample.legal_mask[sample.action] == 1.0
            assert sample.legal_mask.sum() >= 2
            assert sample.waits.shape == (102,)
            assert sample.tenpai.shape == (3,)
            seen_actions.add(sample.action)
            seen_decisions.add(sample.decision_type)
    assert seen_decisions == {0, 1, 2}
    for required in (A.RIICHI, A.RON, A.TSUMO, A.PASS_RESPONSE, A.PASS_RIICHI):
        assert required in seen_actions, A.ACTION_NAMES[required]


@pytest.mark.skipif(not _replay_paths(1), reason="牌譜が無い環境ではスキップ")
def test_parser_score_labels_are_not_all_zero():
    """回帰テスト: target_score が全サンプル 0 だったバグ"""
    parser = MjaiReplayParser(MahjongFeatureExtractor256())
    scores = [s.score_hand for path in _replay_paths(4) for s in parser.parse_file(path)]
    assert scores
    assert any(s != 0.0 for s in scores)
    assert max(abs(s) for s in scores) > 0.1


@pytest.mark.skipif(not _replay_paths(1), reason="牌譜が無い環境ではスキップ")
def test_parser_emits_honor_discards():
    """回帰テスト: 字牌が 1m に潰れ、打牌ラベルの 36% が 1m だったバグ"""
    parser = MjaiReplayParser(MahjongFeatureExtractor256())
    counts = np.zeros(34)
    for path in _replay_paths(6):
        for sample in parser.parse_file(path):
            if sample.action < 34:
                counts[sample.action] += 1
    honors = counts[27:].sum() / counts.sum()
    assert honors > 0.15, f"字牌の打牌比率が {honors:.1%} しかありません"
    assert counts[0] / counts.sum() < 0.12, "1m の比率が異常に高い (字牌が潰れている疑い)"
