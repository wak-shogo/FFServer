# FFServer (Force Field Server)

FFServerは、機械学習力場（Machine Learning Force Fields, MLFF）を用いた原子シミュレーションをブラウザから手軽に実行・管理するためのウェブアプリケーションです。
Streamlitによる直感的なGUIを提供し、バックグラウンドでシミュレーションジョブを処理します。

本プロジェクトは**Docker環境での実行を推奨**しており、複雑な環境構築なしに、主要な機械学習力場（CHGNet, MatterSim, ORB等）をすぐに利用できます。

## 主な機能

- **ウェブGUI (Streamlit)**: 構造ファイルのアップロード、シミュレーションパラメータの設定、結果の可視化。
- **マルチモデル対応**: CHGNet, MatterSim, ORB, MatRIS, NequIP, MatGL, MACEなどをサポート。
- **ジョブキュー管理**: バックグラウンドでのキュー処理。
- **CIFエディタ**: 元素置換（ランダム/順次）機能。
- **Jupyter Lab統合**: 同一コンテナ内でノートブックによる詳細な解析も可能。

## Windows (WSL2) での導入方法（配布イメージを使用する場合）

すでにビルド済みのコンテナイメージ（`ffserver_image.tar`）をお持ちの場合、Windows環境では以下の手順でビルドをスキップして即座に環境を構築できます（RTX 30/40/50シリーズ等のGPU環境でそのまま動作します）。

### 1. 前提条件の確認
- Windows側に **最新のNVIDIAグラフィックドライバ** がインストールされていること。
- **WSL2** (Windows Subsystem for Linux) および **Ubuntu** がインストールされていること。
- **Docker Desktop** (WSL2バックエンド有効) または WSL2上の Docker Engine がインストールされていること。

### 2. イメージの読み込み
Ubuntu（WSL2）のターミナルを開き、`ffserver_image.tar` が保存されているディレクトリ（例: ダウンロードフォルダなら `/mnt/c/Users/ユーザー名/Downloads/`）で以下のコマンドを実行します。

```bash
docker load -i ffserver_image.tar
```

### 3. アプリの起動
本リポジトリのファイル一式（`start_all.sh`など）があるディレクトリに移動し、起動スクリプトを実行します。

```bash
cd /path/to/FFServer
bash start_all.sh
```

起動後、Windows側のブラウザで以下のURLにアクセスしてください。
- **FFServer (GUI)**: `http://localhost:8501`
- **Jupyter Lab**: `http://localhost:8888`

---

## クイックスタート (Docker推奨・自分でビルドする場合)

ソースコードや設定を変更して自分でイメージをビルドする場合は、以下の手順で起動できます。

### (参考) WSL2 Ubuntu 初期セットアップ
もしWSL2をインストールしたばかりで、DockerもNVIDIA Toolkitも入っていない場合は、リポジトリ内のセットアップスクリプトをご利用ください。
```bash
bash setup_wsl2.sh
```
※実行後、WSLの再起動が必要です。Windows側にはあらかじめ**最新のNVIDIAドライバ**がインストールされている必要があります。

### 1. イメージのビルド
```bash
bash build_container.sh
```

### 2. コンテナの起動
```bash
bash start_all.sh
```

起動後、ブラウザで以下のURLにアクセスしてください。
- **FFServer (GUI)**: `http://localhost:8501`
- **Jupyter Lab**: `http://localhost:8888`

---

## 手動インストール (開発者向け)

Dockerを使用せずに直接インストールする場合の手順です。

### 1. 依存ライブラリのインストール
```bash
pip install streamlit ase pandas matplotlib torch streamlit-autorefresh joblib nglview
```

### 2. 各MLFFモデルのインストール
利用したいモデルに応じて、`chgnet`, `matgl`, `orb-models`, `nequip` などを個別にインストールしてください。

### 3. 起動
```bash
# 計算ワーカーの起動
nohup python3 worker.py &
# Streamlitアプリの起動
streamlit run app.py
```

## プロジェクト構造

- `Dockerfile`, `entrypoint.sh`: Docker環境構築用ファイル。
- `start_all.sh`, `build_container.sh`: 起動・ビルド補助スクリプト。
- `app.py`: Streamlitアプリケーション。
- `worker.py`: ジョブ処理ワーカー。
- `simulation_utils.py`: 計算ロジックユーティリティ。
- `supervisord.conf`: プロセス管理設定。

## 注意事項
- **GPU推奨**: 機械学習力場の計算にはCUDA対応のGPU環境を強く推奨します。
- **最新ハードウェア対応**: NVIDIA RTX 50シリーズ (Blackwellアーキテクチャ) 以降に対応するため、DockerイメージはCUDA 12.8 / PyTorch 2.8.0ベースで構成されています。
    - ホストマシン（Windows/Linux）には、最新のNVIDIAドライバがインストールされている必要があります。
- **メモリ**: 複数のモデルを並行してロードする場合、十分なGPUメモリ（12GB以上推奨、RTX 3090/4090/5090クラスを推奨）が必要です。

## 運用・トラブルシューティング

### プロセスの管理
コンテナ内では `supervisord` が Jupyter, Streamlit, Worker の3つのプロセスを管理しています。個別に再起動や状態確認をしたい場合は、以下のコマンドを使用できます。

```bash
# 全プロセスの状態確認
docker exec my_calc_env supervisorctl status

# 特定プロセスの再起動 (例: worker)
docker exec my_calc_env supervisorctl restart worker
```

### ログの確認
- **Workerの内部計算ログ**: `simulation_projects/worker_internal.log` に出力されます。
- **プロセスの実行ログ**: `/var/log/supervisor/` 内に各プロセスの標準出力・エラー出力が記録されています。

---

## Windowsユーザー向け専用ガイド
WSL2やDockerがインストールされていないWindows環境からの詳細なセットアップ手順については、[WINDOWS_INSTALL_GUIDE.md](file:///home/wshogo/git/FFServer/WINDOWS_INSTALL_GUIDE.md) をご参照ください。

