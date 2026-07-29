from feature_extractor import MahjongFeatureExtractor128, MahjongGameState, PlayerState

if __name__ == "__main__":
    # 1. 模组测试用局面的构建（テストデータの作成）
    p0 = PlayerState(
        seat=0,
        score=35000,
        discards=["1m", "9p", "5sr"],
        is_tsumogiri=[False, True, False],
    )
    p1 = PlayerState(
        seat=1,
        score=25000,
        discards=["1z", "2z", "3z"],
        is_tsumogiri=[False, False, False],
    )
    p2 = PlayerState(
        seat=2,
        score=20000,
        discards=["5m", "6m"],
        is_tsumogiri=[True, True],
        is_riichi=True,
        riichi_turn=1,
    )
    p3 = PlayerState(
        seat=3, score=20000, discards=["9s", "8s"], is_tsumogiri=[False, False]
    )

    test_game_state = MahjongGameState(
        self_seat=0,
        players=[p0, p1, p2, p3],
        closed_hand=[
            "1m",
            "2m",
            "3m",
            "5mr",
            "6m",
            "7m",
            "1p",
            "2p",
            "3p",
            "1s",
            "1s",
            "7z",
            "7z",
        ],
        dora_indicators=["4m"],
        round_wind=0,  # 東場
        self_wind=0,  # 東家
        honba=1,
        kyotaku=1,
        tiles_left=52,
        is_all_last=False,
    )

    # 2. 特征提取器的实例化与提取运行
    extractor = MahjongFeatureExtractor128()
    feature_tensor = extractor.extract(test_game_state)

    # 3. 输出形状与数据校验
    print(f"抽出された特徴量テンソルの形状: {feature_tensor.shape}")  # (128, 34)
    print(f"データ型: {feature_tensor.dtype}")  # float32
    print(f"手牌の1m(Index 0)の枚数フラグ (Ch 0): {feature_tensor[0, 0]}")  # 1.0
    print(f"赤5m(Index 4)の所持フラグ (Ch 12): {feature_tensor[12, 4]}")  # 1.0
    print(
        f"ドラ5m(Index 4)のフラグ (Ch 8): {feature_tensor[8, 4]}"
    )  # 1.0 (ドラ指示牌 4m -> ドラ 5m)
    print(f"場風(東場)のフラグ (Ch 112): {feature_tensor[112, 0]}")  # 1.0
