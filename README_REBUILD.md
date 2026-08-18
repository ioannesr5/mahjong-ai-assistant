# 再構築後の使い方 (Rebuild guide)

`修正方案.md` の 阶段0〜阶段5 に沿ってコードを再構築した後の手順。
**アーキテクチャ (256x4x9 入力 / 18 FiLM-ResBlock / 4 層 Transformer / Cross-Attention /
54 次元 policy + value + 3 aux) は一切変更していない。**

---

## 0. 現在の状態

| 項目 | 状態 |
|---|---|
| コード | 阶段0〜阶段5 の修正を適用済み |
| テスト | `pytest` 65 件すべて green |
| データセット | **未再構築**（`data/*.h5` は旧形式・ラベル破損のまま） |
| SL モデル | **未再学習**（`smart_mahjong_base_policy_v2.pth` は旧データ由来） |
| RL | 再開可能だが、**SL 再学習を待つのが本筋** |

---

## 1. ファイル構成

| ファイル | 役割 |
|---|---|
| `actions.py` | 54 次元アクション空間の**唯一の真源**。pymahjong の定数を import して起動時に検証 |
| `feature_extractor.py` | 256 チャネル特徴抽出器。**SL と RL が共有する唯一の実装** |
| `state_codec.py` | 状態テンソルの保存形式 (36 KB → 7.2 KB、lzf 圧縮後 実測 389 B/サンプル) |
| `mjai_parser.py` | 牌譜 → 学習サンプル (打牌 / 鳴き応答 / 立直宣言 / 和了、正例と負例の両方) |
| `data_builder.py` | HDF5 構築 (対局単位分割・test split・マニフェスト・並列解析) |
| `models.py` | ネットワーク定義の**唯一の定義場所** + DirectML 安全なサンプラ |
| `supervised_trainer.py` | 教師あり学習 (合法手マスク・決定タイプ別指標・凍結ベースライン保存) |
| `mahjong_env.py` | 自己対局環境 (公開情報を観測と突き合わせて追跡、半荘連続) |
| `reward.py` | 報酬関数と成分分解 |
| `rl_ppo_trainer.py` | PPO + KL 錨定 + SIL + フェーズ課程 |
| `probe_policy_behavior.py` | 行動プローブ (和了率・鳴き率の実測) |

---

## 2. 実行順序

### ① テストを通す

```bash
.venv/Scripts/python.exe -m pytest -q
```

### ② データセットを再構築する

旧 `data/*.h5` は
「字牌が全部 1m」「target_score が全ゼロ」「聴牌/危険度がゼロ埋め」
「立直・和了・パスのサンプルが 1 件も無い」ため**使い物にならない**。必ず作り直す。

```bash
.venv/Scripts/python.exe data_builder.py --out data_v3 --workers 12 --max-files 40000
```

目安 (実測 400 対局からの外挿):
* 1 対局あたり約 640 サンプル
* 圧縮後 約 389 B/サンプル
* 40000 対局 → 約 2600 万サンプル / 約 10 GB
* 全 184425 対局 → 約 1.2 億サンプル / 約 46 GB (ディスク残 657 GB なので収まる)

構築後に表示されるアクション分布に `RIICHI` / `RON` / `TSUMO` / `PASS_RESPONSE` /
`PASS_RIICHI` が含まれていることを必ず確認すること。

### ③ SL を再学習する

```bash
.venv/Scripts/python.exe supervised_trainer.py --data data_v3 --epochs 20
```

終了時に `policy_out` の各行のノルムが表示される。
`RIICHI` / `RON` / `TSUMO` が `**DEAD**` と出たらデータかラベルがまだおかしい。

成果物:
* `smart_mahjong_base_policy_v3.pth` — 学習済みモデル
* `sl_baseline_frozen.pth` — **凍結ベースライン。今後の全実験の比較基準。上書き禁止**
* `sl_metrics_v3.json` — 決定タイプ別の検証指標

### ④ 行動を実測する

```bash
.venv/Scripts/python.exe probe_policy_behavior.py smart_mahjong_base_policy_v3.pth 6000
```

合格ライン:
* 和了実行率 > 95%
* 鳴き命中率 25〜40%
* 立直機会が 0 でない (門前率が回復している証拠)

### ⑤ PPO を回す

```bash
.venv/Scripts/python.exe rl_ppo_trainer.py
```

再学習した SL から始める場合は、頭部修復の足場を無効にする:

```bash
MJ_REPAIR_HEAD=0 .venv/Scripts/python.exe rl_ppo_trainer.py
```

---

## 3. 常時監視すべき指標

| 指標 | 期待値 | 外れたときの意味 |
|---|---|---|
| **`ratio0`** (第0エポックの重要度比) | **1.00 ± 0.02** | サンプリングと更新で方策が食い違っている。**この値が壊れている間、PPO の更新は無意味** |
| `Win` (和了率) | > 15% | 和了アクションが死んでいる |
| 鳴き率 | 25〜40% | PASS 負例が効いていない |
| `[Reward]` の各成分 | payoff が主、shaping は従 | シェーピングが本来の目的を上回っている |
| `[Head Check] 死行` | なし | 未学習の出力行が残っている |

`ratio0` の警告は、今回発見した
「DirectML の `Categorical` が非合法アクションを返す」級のバグを
唯一検出できる指標なので、**絶対に消さないこと**。

---

## 4. 環境変数

| 変数 | 既定 | 意味 |
|---|---|---|
| `MJ_REPAIR_HEAD` | `1` | 死んだ出力行の修復。SL 再学習後は `0` |
| `MJ_CALL_PENALTY` | `0.0` | 鳴きロジットへの追加補正 (足場。SL 再学習後は不要) |

---

## 5. まだ手を付けていない項目

`修正方案.md` の以下は未着手:

* 対戦相手プール (固定 SL / 歴代最強 RL / heuristic) と bootstrap 信頼区間
* 複数 seed での学習と分散報告
* 報酬 ablation マトリクス (A: 素点のみ 〜 F: フル)
* `config/training.yaml` への設定外出し (現在は `rl_ppo_trainer.py` 冒頭の定数)
* 阶段6 の信念モデル (Opponent Belief Model)
