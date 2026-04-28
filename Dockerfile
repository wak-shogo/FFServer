# ベースイメージとしてCUDA 11.8.0の開発環境を指定
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# apt-get実行時にインタラクティブなプロンプトを無効化
ENV DEBIAN_FRONTEND=noninteractive

# 基本的なツールのインストール
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Node.jsとnpmのインストール (JupyterLab拡張機能のため)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm@latest

# --- Pythonパッケージのインストール ---

# 作業ディレクトリを設定
WORKDIR /workspace

# 依存関係のバージョンを固定するための制約ファイルをコピー
#COPY constraints.txt .

# 1. 基本パッケージとPyTorchを、制約ファイルを使ってインストール
RUN pip3 install\
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --extra-index-url https://download.pytorch.org/whl/cu118 \
    jupyter \
    jupyterlab \
    numpy \
    pandas \
    matplotlib \
    ase \
    py3Dmol \
    nglview \
    ipywidgets \
    pymatgen \
    streamlit \
    streamlit-autorefresh \
    supervisor \
    seaborn \
    MDAnalysis \
    scipy

# 2. MLFFモデルをインストール (制約ファイルを適用)
    # torchを再度指定してアップグレードを防止
RUN pip3 install\
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --extra-index-url https://download.pytorch.org/whl/cu118 \
    chgnet \
    mattersim \
    orb-models \
    nequip-allegro \
    dgl -f https://data.dgl.ai/wheels/torch-2.4/cu118/repo.html \
    matgl
# Clone the MatRIS repository as it's not available on PyPI
RUN git clone https://github.com/HPC-AI-Team/MatRIS.git /opt/MatRIS
ENV PYTHONPATH="/opt/MatRIS"

# Download CHGNet-r2SCAN model
RUN git clone https://github.com/materialyzeai/matgl.git /opt/matgl_temp && \
    mkdir -p /opt/models && \
    cp -r /opt/matgl_temp/pretrained_models/CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES /opt/models/CHGNet_r2SCAN && \
    rm -rf /opt/matgl_temp

# 3. MACEをソースからインストール (制約ファイルを適用)
RUN git clone https://github.com/ACEsuit/mace.git /mace
WORKDIR /mace
#RUN pip3 install -c /workspace/constraints.txt -e .
# MACEインストール後に作業ディレクトリを戻す
WORKDIR /workspace

# --- コンパイル処理は削除し、起動スクリプトを配置 ---
COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

# --- Supervisorの設定 ---
RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/app.conf

# --- JupyterLabの設定 ---
RUN jupyter labextension update --all
RUN jupyter lab build

# --- 最終設定 ---
EXPOSE 8888 8501

# ENTRYPOINTを設定（ここを経由してCMDが実行される）
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []
