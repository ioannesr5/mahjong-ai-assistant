import os
import zipfile

import requests
from tqdm import tqdm

# ==========================================
# 1. 構成とパス設定 (Configuration & Paths)
# ==========================================

DATA_DIR = "data"
MODELS_DIR = os.path.join(DATA_DIR, "models")
LOGS_DIR = os.path.join(DATA_DIR, "logs")

# ダウンロード先のディレクトリを作成（创建下载目标文件夹）
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def download_file(url, dest_path, desc="Downloading"):
    """
    ストリーミングでファイルをダウンロードし、プログレスバーを表示する関数。
    （流式下载文件并显示进度条的函数）
    """
    if os.path.exists(dest_path):
        print(f"[スキップ] {dest_path} は既に存在します。(Already exists)")
        return True

    print(f"\n[接続中] {url}")

    try:
        # User-Agentを偽装して403エラーを回避（伪装UA防止被拦截）
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with (
            open(dest_path, "wb") as file,
            tqdm(
                desc=desc,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for data in response.iter_content(chunk_size=8192):
                size = file.write(data)
                bar.update(size)
        return True
    except Exception as e:
        print(f"[エラー] {desc} のダウンロードに失敗しました: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def extract_zip(zip_path, extract_to):
    """
    ZIPファイルを指定ディレクトリに展開する関数。
    （将 ZIP 文件解压到指定目录的函数）
    """
    print(f"[展開中] {zip_path} を抽出しています... (Extracting...)")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"[完了] {extract_to} に展開しました。(Extraction complete)")
    except zipfile.BadZipFile:
        print(f"[エラー] 無効なZIPファイルです: {zip_path} (Invalid ZIP file)")


# ==========================================
# 2. 天鳳牌譜データセットの取得 (Fetch Tenhou Logs)
# ==========================================


def fetch_tenhou_logs():
    """
    NikkeTryHardリポジトリから2024年のMJAI形式の牌譜を取得する。
    （从 NikkeTryHard 仓库获取 2024 年 MJAI 格式的牌谱）
    """
    print("\n--- 天鳳牌譜データセットの取得を開始します ---")

    # 約1.3GB, 33万局の高品質データ (事前MJAI変換済み)
    archive_url = "https://github.com/NikkeTryHard/tenhou-to-mjai/releases/download/v1.0.0/2024.zip"
    zip_dest = os.path.join(LOGS_DIR, "houou_2024_mjai.zip")

    success = download_file(archive_url, zip_dest, desc="Tenhou 2024 MJAI Logs")
    if success:
        # 解圧先のディレクトリ (data/logs/2024_mjai/)
        extract_to = os.path.join(LOGS_DIR, "2024_mjai")
        if not os.path.exists(extract_to):
            os.makedirs(extract_to)
        extract_zip(zip_dest, extract_to)


# ==========================================
# メイン実行ブロック (Main Execution Block)
# ==========================================
if __name__ == "__main__":
    print("データ準備スクリプトを起動します... (Starting data fetcher...)")
    fetch_tenhou_logs()
    print("\n全てのダウンロードと展開処理が完了しました！ (All tasks completed!)")
