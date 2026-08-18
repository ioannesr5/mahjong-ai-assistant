"""
アクション空間の唯一の真源 (Single Source of Truth for the Action Space)

このファイル以外の場所でアクション ID を数値リテラルとして書くことを禁止する。
定義は pymahjong.MahjongEnv のクラス定数から取得し、起動時に検証する。

参照:
  - pymahjong/env_pymahjong.py の ACTION INDICES 定義
  - pymahjong/observation_action_explanation.pdf Table 5 (v1.0 の 47 次元定義)
"""

from pymahjong import MahjongEnv as _Env

# ==========================================================================
# 現行 (pymahjong >= 1.1) の 54 次元アクション空間
# ==========================================================================
N_ACTIONS = 54

# 0..33: 通常打牌 (牌 ID と一対一)
DISCARD_START, DISCARD_END = 0, 34
# 34..36: 赤ドラ打牌 (赤5m / 赤5p / 赤5s)
DISCARD_RED_START, DISCARD_RED_END = 34, 37
RED_DISCARD_OF_TILE = {4: 34, 13: 35, 22: 36}  # 通常牌 ID -> 赤打牌アクション

# 37..42: チー (左/中/右 × 赤を使わない/使う)
CHI_LEFT, CHI_MIDDLE, CHI_RIGHT = 37, 38, 39
CHI_LEFT_RED, CHI_MIDDLE_RED, CHI_RIGHT_RED = 40, 41, 42
CHI_START, CHI_END = 37, 43

# 43..44: ポン (赤を使わない/使う)
PON, PON_RED = 43, 44
PON_START, PON_END = 43, 45

# 45..47: カン
ANKAN, MINKAN, KAKAN = 45, 46, 47
KAN_START, KAN_END = 45, 48

# 48..51: 宣言系
RIICHI = 48
RON = 49
TSUMO = 50
KYUSHUKYUHAI = 51

# 52..53: パス
PASS_RIICHI = 52
PASS_RESPONSE = 53

# --- グループ (ログ・診断・報酬シェーピング用) ---
DISCARD_ACTIONS = tuple(range(DISCARD_START, DISCARD_RED_END))  # 0..36
CHI_ACTIONS = tuple(range(CHI_START, CHI_END))
PON_ACTIONS = tuple(range(PON_START, PON_END))
KAN_ACTIONS = tuple(range(KAN_START, KAN_END))
CALL_ACTIONS = CHI_ACTIONS + PON_ACTIONS + KAN_ACTIONS  # 37..47 鳴き全般
AGARI_ACTIONS = (RON, TSUMO)
PASS_ACTIONS = (PASS_RIICHI, PASS_RESPONSE)

# SL データセットが従来カバーしていなかった＝重みが死んでいる領域。
# repair_policy_head() と KL ペナルティのマスクで使用する。
DECLARATION_ACTIONS = (RIICHI, RON, TSUMO, KYUSHUKYUHAI, PASS_RIICHI, PASS_RESPONSE)  # 48..53

ACTION_NAMES = {
    **{i: f"DISCARD_{i}" for i in range(DISCARD_START, DISCARD_END)},
    34: "DISCARD_RED_5M",
    35: "DISCARD_RED_5P",
    36: "DISCARD_RED_5S",
    37: "CHI_LEFT",
    38: "CHI_MIDDLE",
    39: "CHI_RIGHT",
    40: "CHI_LEFT_RED",
    41: "CHI_MIDDLE_RED",
    42: "CHI_RIGHT_RED",
    43: "PON",
    44: "PON_RED",
    45: "ANKAN",
    46: "MINKAN",
    47: "KAKAN",
    48: "RIICHI",
    49: "RON",
    50: "TSUMO",
    51: "KYUSHUKYUHAI",
    52: "PASS_RIICHI",
    53: "PASS_RESPONSE",
}


# ==========================================================================
# 旧 47 次元アクション空間 (pymahjong v1.0 / base_policy_v2.pth が学習したもの)
# チェックポイント移行 (adapt_policy_state_dict) 専用。新規コードでは使わないこと。
# ==========================================================================
class Legacy47:
    """observation_action_explanation.pdf Table 5 の定義"""

    N_ACTIONS = 47
    DISCARD_START, DISCARD_END = 0, 34
    CHI_SMALL, CHI_MIDDLE, CHI_LARGE = 34, 35, 36
    PON = 37
    ANKAN, KAN, KAKAN = 38, 39, 40
    RIICHI = 41
    RON = 42
    TSUMO = 43
    KYUSHUKYUHAI = 44
    PASS_RESPONSE = 45
    PASS_RIICHI = 46


# 旧 47 -> 新 54 の写像。値が None の場合は「対応する旧アクションが存在しない」。
LEGACY47_TO_54 = {
    **{i: i for i in range(34)},
    Legacy47.CHI_SMALL: CHI_LEFT,
    Legacy47.CHI_MIDDLE: CHI_MIDDLE,
    Legacy47.CHI_LARGE: CHI_RIGHT,
    Legacy47.PON: PON,
    Legacy47.ANKAN: ANKAN,
    Legacy47.KAN: MINKAN,
    Legacy47.KAKAN: KAKAN,
    Legacy47.RIICHI: RIICHI,
    Legacy47.RON: RON,
    Legacy47.TSUMO: TSUMO,
    Legacy47.KYUSHUKYUHAI: KYUSHUKYUHAI,
    Legacy47.PASS_RESPONSE: PASS_RESPONSE,
    Legacy47.PASS_RIICHI: PASS_RIICHI,
}


def verify_against_pymahjong() -> None:
    """
    pymahjong のバージョン漂流を検出する。import 時に自動実行される。
    ここが落ちたら pymahjong 側の定義が変わっているので、必ず本ファイルを更新すること。
    """
    expected = {
        "CHILEFT": CHI_LEFT,
        "CHIMIDDLE": CHI_MIDDLE,
        "CHIRIGHT": CHI_RIGHT,
        "CHILEFT_USERED": CHI_LEFT_RED,
        "CHIMIDDLE_USERED": CHI_MIDDLE_RED,
        "CHIRIGHT_USERED": CHI_RIGHT_RED,
        "PON": PON,
        "PON_USERED": PON_RED,
        "ANKAN": ANKAN,
        "MINKAN": MINKAN,
        "KAKAN": KAKAN,
        "RIICHI": RIICHI,
        "RON": RON,
        "TSUMO": TSUMO,
        "PUSH": KYUSHUKYUHAI,
        "PASS_RIICHI": PASS_RIICHI,
        "PASS_RESPONSE": PASS_RESPONSE,
    }
    mismatches = [
        f"{name}: pymahjong={getattr(_Env, name)} != actions.py={value}"
        for name, value in expected.items()
        if getattr(_Env, name) != value
    ]
    if _Env.ACTION_DIM != N_ACTIONS:
        mismatches.append(f"ACTION_DIM: pymahjong={_Env.ACTION_DIM} != actions.py={N_ACTIONS}")
    if mismatches:
        raise RuntimeError(
            "pymahjong のアクション定義が actions.py と一致しません:\n  " + "\n  ".join(mismatches)
        )


verify_against_pymahjong()
