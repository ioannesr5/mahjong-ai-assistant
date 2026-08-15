import csv
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# Windows環境向けの日本語フォント設定 (日本語の文字化けを防ぐ)
# ==========================================
matplotlib.rcParams["font.family"] = ["MS Gothic", "Yu Gothic", "Meiryo", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


def smooth_data(data, window_size=10):
    """
    データノイズを減らすための移動平均関数 (Moving Average)
    """
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def load_train_data(filepath):
    """訓練ログ (rl_train_log.csv) を読み込む"""
    iters, rewards, entropies = [], [], []
    win_rates, deal_rates, shantens = [], [], []

    if not os.path.exists(filepath):
        print(f"[Warn] 訓練ログが見つかりません: {filepath}")
        return iters, rewards, entropies, win_rates, deal_rates, shantens

    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                iters.append(int(row["Iteration"]))
                rewards.append(float(row["Reward"]))
                entropies.append(float(row["Entropy"]))

                # 新しく追加されたメトリクスの互換性チェック (古いログには存在しない可能性があるため)
                if "MeanShantenRed" in row:
                    win_rates.append(float(row["WinRate"]))
                    deal_rates.append(float(row["DealInRate"]))
                    shantens.append(float(row["MeanShantenRed"]))
            except (ValueError, KeyError):
                continue

    return iters, rewards, entropies, win_rates, deal_rates, shantens


def load_eval_data(filepath):
    """評価ログ (rl_eval_log.csv) を読み込む"""
    iters, ranks, nets, win_rates, deal_rates = [], [], [], [], []
    if not os.path.exists(filepath):
        print(f"[Warn] 評価ログが見つかりません: {filepath}")
        return iters, ranks, nets, win_rates, deal_rates

    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                iters.append(int(row["Iteration"]))
                ranks.append(float(row["AvgRank"]))
                nets.append(float(row["AvgNet"]))
                win_rates.append(float(row["WinRate"]))
                deal_rates.append(float(row["DealInRate"]))
            except (ValueError, KeyError):
                continue
    return iters, ranks, nets, win_rates, deal_rates


def plot_learning_stability(iters, rewards, entropies, save_dir):
    """図1: 学習の安定性と探索 (Learning Stability & Exploration)"""
    if not iters:
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # ノイズが多い報酬データには移動平均を適用する
    smoothed_rewards = smooth_data(rewards, window_size=20)
    adjusted_iters = iters[len(iters) - len(smoothed_rewards) :]

    color1 = "tab:blue"
    ax1.set_xlabel("イテレーション (Iteration)")
    ax1.set_ylabel("平均報酬 (Average Reward) [Smoothed]", color=color1)
    ax1.plot(adjusted_iters, smoothed_rewards, color=color1, alpha=0.8, label="Reward")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("エントロピー (Entropy)", color=color2)
    ax2.plot(iters, entropies, color=color2, alpha=0.7, label="Entropy")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("図1: 学習の安定性と探索推移")
    fig.tight_layout()

    save_path = os.path.join(save_dir, "plot_1_learning_stability.png")
    plt.savefig(save_path, dpi=300)
    print(f" -> [Info] プロット1 生成完了: {save_path}")
    plt.close()


def plot_absolute_strength(iters, ranks, nets, save_dir):
    """図2: 絶対的な強さの進化 (Absolute Strength Progression)"""
    if not iters:
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "tab:green"
    ax1.set_xlabel("定期評価イテレーション (Eval Iteration)")
    ax1.set_ylabel("平均順位 (Average Rank)", color=color1)
    ax1.plot(iters, ranks, color=color1, marker="o", linestyle="-", label="Avg Rank")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(4.0, 1.0)  # 順位は低い数字（1位）が上に来るように反転させる
    ax1.axhline(y=2.5, color="gray", linestyle="--", alpha=0.5)  # 基準線

    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel("半荘均浄勝素点 (Net Score / pt)", color=color2)
    ax2.plot(iters, nets, color=color2, marker="x", linestyle="--", label="Net Score")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("図2: 対SLベースライン 絶対的強さの進化推移")
    fig.tight_layout()

    save_path = os.path.join(save_dir, "plot_2_absolute_strength.png")
    plt.savefig(save_path, dpi=300)
    print(f" -> [Info] プロット2 生成完了: {save_path}")
    plt.close()


def plot_eval_behavioral_shift(iters, win_rates, deal_rates, save_dir):
    """図3: 評価時のプレイスタイル変遷 (Eval Behavioral Shift)"""
    if not iters:
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("定期評価イテレーション (Eval Iteration)")
    ax1.set_ylabel("確率 (Rate)")

    ax1.plot(iters, win_rates, color="tab:red", marker="^", linestyle="-", label="和了率 (Win Rate)")
    ax1.plot(iters, deal_rates, color="tab:blue", marker="v", linestyle="-", label="放銃率 (Deal-in Rate)")

    vals = ax1.get_yticks()
    ax1.set_yticklabels([f"{x:,.1%}" for x in vals])

    ax1.legend(loc="upper right")
    plt.title("図3: 評価時のプレイスタイル変遷 (和了率 vs 放銃率)")
    ax1.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()

    save_path = os.path.join(save_dir, "plot_3_eval_behavior.png")
    plt.savefig(save_path, dpi=300)
    print(f" -> [Info] プロット3 生成完了: {save_path}")
    plt.close()


def plot_train_realtime_metrics(iters, win_rates, deal_rates, shantens, save_dir):
    """図4: 訓練中のリアルタイム牌効率とプレイスタイル監視 (Realtime Train Metrics)"""
    if not win_rates or len(win_rates) < 10:
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 日常訓練データはノイズが大きいため平滑化する
    smooth_w = smooth_data(win_rates, 10)
    smooth_d = smooth_data(deal_rates, 10)
    smooth_s = smooth_data(shantens, 10)
    adj_iters = iters[len(iters) - len(smooth_w) :]

    ax1.set_xlabel("イテレーション (Iteration)")
    ax1.set_ylabel("確率 (Rate) [Smoothed]")
    ax1.plot(adj_iters, smooth_w, color="tab:red", alpha=0.7, label="和了率 (Win Rate)")
    ax1.plot(adj_iters, smooth_d, color="tab:blue", alpha=0.7, label="放銃率 (Deal-in Rate)")
    vals = ax1.get_yticks()
    ax1.set_yticklabels([f"{x:,.1%}" for x in vals])
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    color2 = "tab:green"
    ax2.set_ylabel("平均シャンテン変化 (Mean Shanten Reduction)", color=color2)
    ax2.plot(adj_iters, smooth_s, color=color2, alpha=0.8, linestyle="-.", label="Δ Shanten")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.legend(loc="upper right")

    plt.title("図4: 訓練時のリアルタイム牌効率と行動監視 (移動平均)")
    ax1.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()

    save_path = os.path.join(save_dir, "plot_4_train_realtime_metrics.png")
    plt.savefig(save_path, dpi=300)
    print(f" -> [Info] プロット4 生成完了: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 50)
    print("📊 強化学習メトリクス可視化ツール (RL Metrics Plotter)")
    print("=" * 50)

    log_directory = "logs"
    if not os.path.exists(log_directory):
        print(f"[Error] ログディレクトリ '{log_directory}' が見つかりません。")
        sys.exit(1)

    train_csv = os.path.join(log_directory, "rl_train_log.csv")
    eval_csv = os.path.join(log_directory, "rl_eval_log.csv")

    # データの読み込み
    t_iters, t_rewards, t_entropies, t_win_r, t_deal_r, t_shanten = load_train_data(train_csv)
    e_iters, e_ranks, e_nets, e_win_rates, e_deal_rates = load_eval_data(eval_csv)

    # プロットの生成
    plot_learning_stability(t_iters, t_rewards, t_entropies, log_directory)
    plot_absolute_strength(e_iters, e_ranks, e_nets, log_directory)
    plot_eval_behavioral_shift(e_iters, e_win_rates, e_deal_rates, log_directory)
    plot_train_realtime_metrics(t_iters, t_win_r, t_deal_r, t_shanten, log_directory)

    print("✨ 全てのプロット処理が正常に完了しました。")
