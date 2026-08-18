# 🀄 SmartMahjong AI (V2)

[**简体中文**](#-简体中文) | [**English**](#-english) | [**日本語**](#-日本語)

---

## 🇨🇳 简体中文

### 📖 1. 项目概述
**SmartMahjong AI** 是一个面向四人日本麻将的端到端深度强化学习决策系统。本项目结合了基于天凤专家牌谱的**多任务监督学习**与**三阶段课程强化学习**，利用深度神经网络精确建模牌效、攻防转换、敌方听牌推测及防守读牌逻辑。

---

### 🌟 2. 核心架构与算法设计

#### 2.1 256 通道高维空间特征 (`feature_extractor.py`)
将非完美信息博弈环境编码为 `256 × 4 × 9` 的空间张量：
* **0 ~ 6 通道**：自身持牌与赤宝牌标记。
* **7 ~ 54 通道**：四家副露（吃、碰、暗杠、明杠、副露内宝牌）。
* **55 ~ 150 通道**：四家舍牌历史，精准区分手切与摸切、立直宣言牌及立直后舍牌。
* **151 ~ 170 通道**：表宝牌与杠宝牌指示牌。
* **171 ~ 210 通道（防守矩阵）**：四家现物、筋牌推断、壁牌物理阻断（No-Chance）、无筋危险牌掩码及生牌字牌。
* **211 ~ 255 通道**：场风、自风、本场数、供托及点数差全局上下文。
* **序列与条件向量**：16 维局况 FiLM 条件向量与 72 巡全局复合 Token 舍牌序列。

#### 2.2 深度多任务骨干网络 (`SmartMahjongMultiTaskNet`)
* **FiLM ResBlock (18 层)**：在 CNN 残差结构中动态调制点数与局况条件，平衡全局大局观与局部牌效。
* **时序舍牌编码器 (Discard Sequence Transformer)**：4 层 Transformer 建模全卓 72 巡舍牌与摸切时序。
* **信念交叉注意力 (Mahjong Belief Cross-Attention)**：以空间 CNN 特征为 Query，舍牌 Transformer 序列为 Key/Value 进行 Cross-Attention，捕捉隐蔽听牌与待牌信息。
* **多任务联合预测 (Multi-Task Heads)**：
  1. 动作策略头（Policy Out）：54 维合法动作分布。
  2. 状态价值头（Value Head）：局收支预估。
  3. 辅助听牌头（Aux Tenpai）：预测其余 3 家听牌概率。
  4. 辅助危险度头（Aux Danger）：预测 102 维（3家 × 34牌）危险度分布。
  5. 辅助待牌头（Aux Waits）：预测 102 维敌方实际听牌待牌分布。

#### 2.3 三阶段课程强化学习与自我模仿 (`PPO + SIL`)
* **Phase 1 (牌效与做牌引导)**：引入向听进速塑形（$\Delta\text{Shanten}$）与即时进张受入（Ukeire）正向奖励，配合自我模仿学习（Hero Replay Buffer / SIL）快速掌握扎实牌效。
* **Phase 2 (攻防微调与防过拟合)**：降低塑形奖励权重，引入动态 SL 基线（KL 散度约束），以和了率 $\ge 5\%$ 与平均顺位为黄金门槛进行晋级考核。
* **Phase 3 (纯粹终局博弈)**：完全去除启发式塑形奖励，仅依靠最终素点差与马场顺位点进行自我对弈，超越 Phase 2 毕业模型。

#### 2.4 高性能异步训练引擎
* **集中式 GPU 批量推理 (Central Batch Inference)**：支持 30+ 并行 Worker 进程环境交互，跨进程 IPC 管道聚合微批次请求，彻底消除 CPU-GPU 通信瓶颈。
* **DirectML 硬件加速**：深度适配 AMD/Intel/NVIDIA GPU，定制 `DirectMLSafeAdamW` 与数值安全 BCE 损失。

---

### 📂 3. 项目结构

```text
├── feature_extractor.py   # 256通道空间特征提取与防守矩阵计算
├── data_fetcher.py        # 天凤高段位 MJAI 牌谱流式下载脚本
├── data_builder.py        # 牌谱解析、待牌矩阵计算与 HDF5 构建
├── supervised_trainer.py  # 预训练多任务监督学习引擎
├── rl_ppo_trainer.py      # 异步多进程 PPO + SIL 自对弈强化学习系统
├── plot_metrics.py        # 训练/评估指标可视化绘图工具
├── pyproject.toml         # 代码规范与 Ruff Linter 配置
└── LICENSE                # MIT 开源许可证
```

---

### 🚀 4. 环境配置与快速开始

#### 4.1 安装依赖
```bash
git clone [https://github.com/your-username/smart-mahjong-ai.git](https://github.com/your-username/smart-mahjong-ai.git)
cd smart-mahjong-ai
pip install numpy torch torch-directml h5py requests tqdm matplotlib mahjong pymahjong
```

#### 4.2 数据获取与预训练 (SL Phase)
```bash
# 1. 下载天凤 2024 年 MJAI 牌谱数据 (约 33 万局)
python data_fetcher.py

# 2. 构建多任务 HDF5 训练/验证数据集
python data_builder.py

# 3. 启动监督学习多任务预训练 (产出 smart_mahjong_base_policy_v2.pth)
python supervised_trainer.py
```

#### 4.3 强化学习自对弈训练 (RL Phase)
```bash
# 启动异步多进程 PPO 自对弈训练 (支持断点自动恢复与阶段自动晋级)
python rl_ppo_trainer.py
```

#### 4.4 可视化分析
```bash
# 生成训练稳定性、顺位进化及攻防风格变迁图表
python plot_metrics.py
```

---

## 🇬🇧 English

### 📖 1. Project Overview
**SmartMahjong AI** is an end-to-end deep reinforcement learning system tailored for 4-player Japanese Riichi Mahjong. By combining **Multi-Task Supervised Pre-training** on expert game logs with a **3-Phase Curriculum Reinforcement Learning** architecture, the model masters tile efficiency, danger inference, and spatial defensive reading.

---

### 🌟 2. Core Architecture & Features

#### 2.1 256-Channel Spatial Representation (`feature_extractor.py`)
Encodes the imperfect-information game state into a `256 × 4 × 9` spatial tensor:
* **Channels 0–6**: Closed hand tiles and red dora flags.
* **Channels 7–54**: Four-player melds (Chi, Pon, Ankan, Minkan, Dora in melds).
* **Channels 55–150**: Detailed discard tracking with Tedashi vs. Tsumogiri, Riichi declaration, and post-Riichi history.
* **Channels 151–170**: Dora and Kan-dora indicators.
* **Channels 171–210 (Defense Matrix)**: Genbutsu, Suji inference, No-Chance blockers, Musuji danger masks, and Shonpai honors.
* **Channels 211–255**: Global game context (Round/Seat winds, Honba, Kyotaku, Score differentials).
* **Sequence & Conditions**: 16-dim context FiLM vector + 72-token composite discard sequence.

#### 2.2 Neural Network Backbone (`SmartMahjongMultiTaskNet`)
* **18-Block FiLM ResNet**: Dynamic conditioning with round state and point standings.
* **Discard Sequence Transformer**: 4-layer self-attention over the global 72-step discard history.
* **Mahjong Belief Cross-Attention**: Fuses 2D spatial queries with sequential discard keys/values to read hidden opponent states.
* **Multi-Task Auxiliary Heads**: Jointly predicts Policy (54 actions), Hand Value, Opponent Tenpai status, 102-dim Danger distribution, and 102-dim Opponent Wait (Waits) distribution.

#### 2.3 3-Phase Curriculum PPO + SIL Pipeline
* **Phase 1 (Tile Efficiency & Formation)**: Shanten reduction shaping ($\Delta\text{Shanten}$) + Ukeire heuristic reward + Self-Imitation Learning (Hero Replay Buffer).
* **Phase 2 (Attack/Defense Balance)**: Annealed shaping weights + KL penalty against dynamic SL baselines + Promotion gate (Win Rate $\ge 5\%$, Avg Rank $\le 2.40$).
* **Phase 3 (Pure Endgame Self-Play)**: Complete elimination of heuristic shaping; zero-sum optimization driven solely by raw score balance and placement Uma.

#### 2.4 High-Throughput Asynchronous IPC Engine
* **Centralized GPU Batch Inference**: Micro-batched GPU forwarding serving 30+ asynchronous worker processes via inter-process pipes.
* **DirectML Acceleration**: Native support for AMD/Intel/NVIDIA hardware with `DirectMLSafeAdamW`.

---

### 📂 3. Repository Layout

```text
├── feature_extractor.py   # 256-channel extractor & defense matrix computation
├── data_fetcher.py        # Streamlined downloader for Tenhou MJAI logs
├── data_builder.py        # Log parser & HDF5 dataset builder with wait labeling
├── supervised_trainer.py  # Multi-task supervised pre-training pipeline
├── rl_ppo_trainer.py      # Multi-process asynchronous PPO + SIL self-play engine
├── plot_metrics.py        # Training & evaluation visualization script
├── pyproject.toml         # Ruff linter & formatting specifications
└── LICENSE                # MIT Open Source License
```

---

### 🚀 4. Quick Start

#### 4.1 Installation
```bash
git clone [https://github.com/your-username/smart-mahjong-ai.git](https://github.com/your-username/smart-mahjong-ai.git)
cd smart-mahjong-ai
pip install numpy torch torch-directml h5py requests tqdm matplotlib mahjong pymahjong
```

#### 4.2 Data Pipeline & Supervised Learning
```bash
# 1. Download Tenhou 2024 expert MJAI logs
python data_fetcher.py

# 2. Build multi-task HDF5 dataset
python data_builder.py

# 3. Train base policy network (Outputs smart_mahjong_base_policy_v2.pth)
python supervised_trainer.py
```

#### 4.3 Reinforcement Learning Self-Play
```bash
# Launch asynchronous PPO training with automatic phase progression
python rl_ppo_trainer.py
```

#### 4.4 Plot Metrics
```bash
# Generate high-resolution metric and behavioral shift graphs
python plot_metrics.py
```

---

## 🇯🇵 日本語

### 📖 1. プロジェクト概要
**SmartMahjong AI** は、四人打ち麻雀（リーチ麻雀）を対象としたエンドツーエンドの深層強化学習意思決定システムです。天鳳の鳳凰卓・上級牌譜に基づく**マルチタスク教師あり学習**と、独自の**3段階カリキュラム強化学習（PPO + SIL）**を融合させ、高度な牌効率、攻防判断、敵の聴牌読み、および筋・壁・現物による厳密な防守推論を実現しています。

---

### 🌟 2. 主要アーキテクチャとアルゴリズム設計

#### 2.1 256チャンネル空間特徴量 (`feature_extractor.py`)
不完全情報ゲームの状態を `256 × 4 × 9` の空間テンソルにエンコード：
* **チャンネル 0 ~ 6**：自家手牌および赤ドラフラグ。
* **チャンネル 7 ~ 54**：四家の副露情報（チー、ポン、暗槓、明槓、副露内ドラ）。
* **チャンネル 55 ~ 150**：四家の捨て牌履歴（手切りとツモ切りの厳密な分離、立直宣言牌、立直後捨て牌）。
* **チャンネル 151 ~ 170**：表ドラおよび槓ドラ表示牌。
* **チャンネル 171 ~ 210（防守マトリックス）**：現物、スジ推論、ノーチャンス（カベ）、無筋危険牌マスク、生牌字牌情報。
* **チャンネル 211 ~ 255**：場風、自風、本場、供託、点数差などの局勢コンテキスト。
* **系列・条件ベクトル**：16次元 FiLM 条件ベクトルおよび 72 巡複合トークン打牌系列。

#### 2.2 深層マルチタスクモデル (`SmartMahjongMultiTaskNet`)
* **18層 FiLM ResBlock**：点況・局勢情報を CNN 特徴量へ動的注入（FiLM 変調）。
* **打牌系列 Transformer エンコーダ**：全卓 72 巡の捨て牌履歴を系列処理。
* **信念クロスアテンション（Mahjong Belief Cross-Attention）**：空間特徴量を Query、捨て牌系列を Key/Value として照合し、他家の隠蔽状態（聴牌・待ち牌）を推論。
* **マルチタスク補助ヘッド（Multi-Task Heads）**：
  1. 方策ヘッド（Policy）：54 次元のアクション確率分布。
  2. 状態価値ヘッド（Value）：局収支の期待値。
  3. 聴牌予測ヘッド（Aux Tenpai）：他家 3 名の聴牌確率。
  4. 危険牌予測ヘッド（Aux Danger）：102 次元（3家 × 34牌）の放銃危険度。
  5. 待ち牌予測ヘッド（Aux Waits）：102 次元の敵方待ち牌分布。

#### 2.3 3段階カリキュラム強化学習 (`PPO + SIL`)
* **Phase 1（牌効率・進行加速）**：向聴進速シェーピング（$\Delta\text{Shanten}$）と受入（Ukeire）報酬を付与。自己模倣学習（Hero Replay Buffer / SIL）により堅実な牌効率を即座に定着。
* **Phase 2（攻防微調整・過学習抑制）**：シェーピング報酬を縮小し、動的SLベースラインとのKLペナルティ（KL Divergence）で探索を安定化。和了率 $\ge 5\%$ および平均順位により昇格判定。
* **Phase 3（純粋強化学習・対戦収束）**：シェーピング報酬を完全撤廃し、素点収支と最終順位ウマのみを目的関数として自己対戦。

#### 2.4 高性能非同期訓練エンジン
* **集中型 GPU バッチ推論 (Central Batch Inference)**：30 以上の並列 Worker プロセスを IPC パイプで統括し、マイクロバッチ化して GPU 推論を最大効率化。
* **DirectML ハードウェア最適化**：AMD/Intel/NVIDIA GPU に最適化された `DirectMLSafeAdamW` と数値安定化損失関数を搭載。

---

### 📂 3. ディレクトリ構成

```text
├── feature_extractor.py   # 256チャンネル特徴量抽出および防守マトリックス生成
├── data_fetcher.py        # 天鳳 2024 年 MJAI 牌譜ストリーミング取得スクリプト
├── data_builder.py        # 牌譜パース、待ち牌計算、HDF5 データセット構築
├── supervised_trainer.py  # マルチタスク教師あり学習パイプライン
├── rl_ppo_trainer.py      # 非同期マルチプロセス PPO + SIL 自己対局強化学習システム
├── plot_metrics.py        # 訓練・評価メトリクス可視化プロット作成ツール
├── pyproject.toml         # Ruff リンター・フォーマッタ設定
└── LICENSE                # MIT オープンソースライセンス
```

---

### 🚀 4. インストールと実行手順

#### 4.1 依存パッケージの導入
```bash
git clone [https://github.com/your-username/smart-mahjong-ai.git](https://github.com/your-username/smart-mahjong-ai.git)
cd smart-mahjong-ai
pip install numpy torch torch-directml h5py requests tqdm matplotlib mahjong pymahjong
```

#### 4.2 データ準備と教師あり学習 (SL Phase)
```bash
# 1. 天鳳 2024 年 MJAI 牌譜データ (約33万局) の自動取得
python data_fetcher.py

# 2. マルチタスク HDF5 データセットの構築
python data_builder.py

# 3. 教師あり学習ベースモデルの事前訓練 (smart_mahjong_base_policy_v2.pth を生成)
python supervised_trainer.py
```

#### 4.3 強化学習自己対戦の実行 (RL Phase)
```bash
# 非同期マルチプロセス PPO 自己対戦の開始 (チェックポイント自動再開・自動昇格対応)
python rl_ppo_trainer.py
```

#### 4.4 メトリクスの可視化
```bash
# 学習曲線、平均順位進化、攻防プレイスタイル変遷グラフの生成
python plot_metrics.py
```

---

## 📜 ライセンス (License)

本プロジェクトは [MIT License](LICENSE) のもとで公開されています。  
Copyright (c) 2026 雑魚寝
