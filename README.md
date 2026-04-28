# FFServer (Force Field Server)

FFServerは、機械学習力場（Machine Learning Force Fields, MLFF）を用いた原子シミュレーションをブラウザから手軽に実行・管理するためのウェブアプリケーションです。
Streamlitによる直感的なGUIを提供し、バックグラウンドでシミュレーションジョブを処理します。

## 主な機能

- **ウェブGUI (Streamlit)**: 構造ファイルのアップロード、シミュレーションパラメータの設定、結果の可視化をブラウザ上で行えます。
- **マルチモデル対応**: CHGNet, MatterSim, ORB, MatRIS, NequIP, MatGLなどの主要な機械学習力場をサポートしています。
- **ジョブキュー管理**: シミュレーションをキューに登録し、バックグラウンドで順番に処理します。
- **CIFエディタ**: アップロードしたCIFファイルの元素置換（ランダム/順次）を簡単に行えます。
- **可視化機能**: 構造の最適化プロセス、NPTシミュレーションの温度・エネルギー・体積の変化をグラフ表示します。また、`nglview`による構造の3D表示も可能です。
- **バッチ処理**: 複数の温度条件での連続実行や、加熱・冷却プロセスのシミュレーションに対応しています。

## インストール方法

### 1. リポジトリのクローン
```bash
git clone https://github.com/yourusername/FFServer.git
cd FFServer
```

### 2. Python環境の構築
Python 3.10以降を推奨します。

### 3. 依存ライブラリのインストール
主要な依存関係をインストールします。

```bash
pip install streamlit ase pandas matplotlib torch streamlit-autorefresh joblib nglview
```

### 4. 機械学習力場モデルのインストール
利用したいモデルに応じて、以下のパッケージをインストールしてください。

- **CHGNet**: `pip install chgnet`
- **MatGL (M3GNetなど)**: `pip install matgl`
- **ORB**: `pip install orb-models`
- **NequIP**: `pip install nequip nequip-allegro`
- **MatterSim / MatRIS**: それぞれの公式リポジトリの指示に従ってインストールしてください。

※ GPUを利用する場合は、対応するバージョンのPyTorch (CUDA版) をインストールしてください。

## 使い方

FFServerは、ウェブインターフェースを提供する `app.py` と、計算を実行する `worker.py` の2つのプロセスで構成されます。

### 1. サーバーの起動
`start_app.sh` を使用するか、手動で以下のコマンドを実行します。

```bash
# 計算ワーカーの起動（バックグラウンド）
nohup python3 worker.py &

# Streamlitアプリの起動
streamlit run app.py
```

### 2. ブラウザでアクセス
デフォルトでは `http://localhost:8501` でアクセス可能です。

### 3. シミュレーションの手順
1.  **Sidebar**: CIFファイルをアップロードします。
2.  **Structure Editor**: 必要に応じて元素置換などを行います。
3.  **Simulation Setup**: 使用するモデル、シミュレーションモード（Optimization / NPT）、温度範囲、ステップ数などを設定します。
4.  **Run Simulation**: 「Add to Queue & Start Worker」ボタンをクリックしてジョブを登録します。
5.  **Monitoring**: ジョブの進捗やリアルタイムのデータ更新を確認できます。
6.  **Results**: 完了したジョブの結果（構造ファイル、グラフ、CSV）をダウンロードできます。

## プロジェクト構造

- `app.py`: Streamlitウェブアプリケーションのメインスクリプト。
- `worker.py`: ジョブキューを監視し、シミュレーションを実行するワーカー。
- `simulation_utils.py`: ASEや各種MLFFを用いた計算ロジックのユーティリティ。
- `visualization.py`: グラフ描画や3D表示のための関数群。
- `cif_editor.py`: CIFファイルの編集ロジック。
- `simulation_projects/`: 実行された各ジョブのデータやログが保存されるディレクトリ。
- `supervisord.conf`: Docker環境などで複数のプロセスを管理するための設定ファイル。

## 注意事項
- 大規模なシミュレーションや並列ジョブの実行には、十分なGPUメモリとシステムメモリが必要です。
- 各モデルの利用規約や引用情報については、それぞれの開発元のドキュメントを参照してください。
