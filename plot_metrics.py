import csv
import os
import sys

import matplotlib
import matplotlib.pyplot as plt

# ==========================================
# Windows環境向けの日本語フォント設定 (日本語の文字化けを防ぐ)
# ==========================================
matplotlib.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_train_data(filepath):
    """ 訓練ログ (rl_train_log.csv) を読み込む """
    iters, rewards, entropies = [], [], []
    if not os.path.exists(filepath):
        print(f"[Warn] 訓練ログが見つかりません: {filepath}")
        return iters, rewards, entropies

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                iters.append(int(row['Iteration']))
                rewards.append(float(row['Reward']))
                entropies.append(float(row['Entropy']))
            except (ValueError, KeyError):
                continue
    return iters, rewards, entropies

def load_eval_data(filepath):
    """ 評価ログ (rl_eval_log.csv) を読み込む """
    iters, ranks, nets, win_rates, deal_rates = [], [], [], [], []
    if not os.path.exists(filepath):
        print(f"[Warn] 評価ログが見つかりません: {filepath}")
        return iters, ranks, nets, win_rates, deal_rates

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                iters.append(int(row['Iteration']))
                ranks.append(float(row['AvgRank']))
                nets.append(float(row['AvgNet']))
                win_rates.append(float(row['WinRate']))
                deal_rates.append(float(row['DealInRate']))
            except (ValueError, KeyError):
                continue
    return iters, ranks, nets, win_rates, deal_rates

def plot_learning_stability(iters, rewards, entropies, save_dir):
    """ 図1: 学習の安定性と探索 (Learning Stability & Exploration) """
    if not iters: return
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('イテレーション (Iteration)')
    ax1.set_ylabel('平均報酬 (Average Reward)', color=color1)
    ax1.plot(iters, rewards, color=color1, alpha=0.7, label='Reward')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('エントロピー (Entropy)', color=color2)
    ax2.plot(iters, entropies, color=color2, alpha=0.7, label='Entropy')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('学習の安定性と探索推移')
    fig.tight_layout()
    
    save_path = os.path.join(save_dir, 'plot_1_learning_stability.png')
    plt.savefig(save_path, dpi=300)
    print(f" -> [Info] プロット生成完了: {save_path}")
    plt.close()

def plot_absolute_strength(iters, ranks, nets, save_dir):
    """ 図2: 絶対的な強さの進化 (Absolute Strength Progression) """
    if not iters: return
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'tab:green'
    ax1.set_xlabel('イテレーション (Iteration)')
    ax1.set_ylabel('平均順位 (Average Rank)', color=color1)
    ax1.plot(iters, ranks, color=color1, marker='o', linestyle='-', label='Avg Rank')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(4.0, 1.0) # 順位は低い数字（1位）が上に来るように反転させる
    ax1.axhline(y=2.5, color='gray', linestyle='--', alpha=0.5) # 基準線
    
    ax2 = ax1.twinx()  
    color2 = 'tab:orange'
    ax2.set_ylabel('半荘均浄勝素点 (Net Score / pt)', color=color2)
    ax2.plot(iters, nets, color=color2, marker='x', linestyle='--', label='Net Score')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('対SLベースライン：絶対的強さの進化推移')
    fig.tight_layout()
    
    save_path = os.path.join(save_dir, 'plot_2_absolute_strength.png')
    plt.savefig(save_path, dpi=300)
    print(f" -> [Info] プロット生成完了: {save_path}")
    plt.close()

def plot_behavioral_shift(iters, win_rates, deal_rates, save_dir):
    """ 図3: 行動のシフト (Behavioral Shift: Attack vs Defense) """
    if not iters: return
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('イテレーション (Iteration)')
    ax1.set_ylabel('確率 (Rate)')
    
    ax1.plot(iters, win_rates, color='tab:red', marker='^', linestyle='-', label='和了率 (Win Rate)')
    ax1.plot(iters, deal_rates, color='tab:blue', marker='v', linestyle='-', label='放銃率 (Deal-in Rate)')
    
    # 軸をパーセンテージ表示にする
    vals = ax1.get_yticks()
    ax1.set_yticklabels([f'{x:,.1%}' for x in vals])
    
    ax1.legend(loc='upper right')
    plt.title('プレイスタイルの変遷：和了率と放銃率の推移')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    fig.tight_layout()
    
    save_path = os.path.join(save_dir, 'plot_3_behavioral_shift.png')
    plt.savefig(save_path, dpi=300)
    print(f" -> [Info] プロット生成完了: {save_path}")
    plt.close()

if __name__ == '__main__':
    print("="*50)
    print("📊 強化学習メトリクス可視化ツール (RL Metrics Plotter)")
    print("="*50)
    
    log_directory = "logs"
    if not os.path.exists(log_directory):
        print(f"[Error] ログディレクトリ '{log_directory}' が見つかりません。")
        sys.exit(1)
        
    train_csv = os.path.join(log_directory, "rl_train_log.csv")
    eval_csv = os.path.join(log_directory, "rl_eval_log.csv")
    
    # データの読み込み
    t_iters, t_rewards, t_entropies = load_train_data(train_csv)
    e_iters, e_ranks, e_nets, e_win_rates, e_deal_rates = load_eval_data(eval_csv)
    
    # プロットの生成
    plot_learning_stability(t_iters, t_rewards, t_entropies, log_directory)
    plot_absolute_strength(e_iters, e_ranks, e_nets, log_directory)
    plot_behavioral_shift(e_iters, e_win_rates, e_deal_rates, log_directory)
    
    print("✨ 全てのプロット処理が正常に完了しました。")