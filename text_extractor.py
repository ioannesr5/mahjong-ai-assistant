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
            "1m", "2m", "3m", 
            "5mr", "6m", "7m",
            "1p", "2p", "3p",
            "1s", "1s",
            "7z", "7z",
        ],
        dora_indicators=["4m"],
        round_wind=0,  # 東場
        self_wind=0,   # 東家
        honba=1,
        kyotaku=1,
        tiles_left=52,
        is_all_last=False,
        # 新規追加: 時系列エンコーダ用のグローバル打牌履歴
        global_discards=["1m", "9p", "5sr", "1z", "2z", "3z", "5m", "6m", "9s", "8s"] 
    )

    # 2. 特征提取器的实例化与提取运行（特徴量抽出の実行）
    extractor = MahjongFeatureExtractor128()
    features = extractor.extract(test_game_state)
    
    state_2d = features['state_2d']
    cond_vec = features['cond_vec']
    seq_hist = features['seq_hist']

    # 3. 输出形状与数据校验（形状とデータの検証）
    print("=== 1. 2D 空間テンソル (Spatial Tensor) ===")
    print(f"形状: {state_2d.shape}")  # 期待値: (128, 4, 9)
    print(f"データ型: {state_2d.dtype}")  # 期待値: float32
    
    # マッピング確認: 1m は ID 0 -> Suit 0, Num 0
    print(f"手牌の1m所持フラグ (Ch 0, Suit 0, Num 0): {state_2d[0, 0, 0]}")  # 期待値: 1.0
    # マッピング確認: 赤5m は ID 4 -> Suit 0, Num 4
    print(f"赤5mの所持フラグ (Ch 12, Suit 0, Num 4): {state_2d[12, 0, 4]}")  # 期待値: 1.0
    # マッピング確認: ドラ指示牌 4m -> ドラ 5m (ID 4)
    print(f"ドラ5mのフラグ (Ch 8, Suit 0, Num 4): {state_2d[8, 0, 4]}")  # 期待値: 1.0
    # マッピング確認: 字牌はパディングにより Suit 3 に配置される (1z は ID 27 -> Suit 3, Num 0)
    print(f"場風(東場)のフラグ (Ch 112, Suit 3, Num 0): {state_2d[112, 3, 0]}")  # 期待値: 1.0

    print("\n=== 2. 条件ベクトル (Condition Vector) ===")
    print(f"形状: {cond_vec.shape}")  # 期待値: (16,)
    # 自家スコア 35000 -> (35000-25000)/10000 = 1.0
    print(f"自家スコア正規化値 (Index 0): {cond_vec[0]}")  # 期待値: 1.0
    # 場風・東 -> One-Hot (Index 4)
    print(f"場風・東のOne-Hot (Index 4): {cond_vec[4]}")  # 期待値: 1.0
    
    print("\n=== 3. 時系列シーケンス (Sequence History) ===")
    print(f"形状: {seq_hist.shape}")  # 期待値: (72,)
    print(f"データ型: {seq_hist.dtype}")  # 期待値: int64
    # 最初の打牌 "1m" (ID: 0)
    print(f"第1打牌のID (Index 0): {seq_hist[0]}")  # 期待値: 0
    # 欠損値パディング (Padding ID: 34)
    print(f"末尾のパディングID (Index -1): {seq_hist[-1]}")  # 期待値: 34