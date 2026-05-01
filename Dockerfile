# ベースイメージとしてCUDA 12.8.0の開発環境を指定 (RTX 5090/Blackwell対応のため)
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

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

# 1. PyTorchをインストール (StarterKit準拠のPyTorch 2.6.0 + CUDA 12.4)
RUN pip3 install --upgrade pip && \
    pip3 install \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 2. その他の基本パッケージをインストール (PyPIから)
# StarterKitに倣い、numpy<2 および torchdata==0.7.1 を指定
RUN pip3 install \
    jupyter \
    jupyterlab \
    "numpy<2" \
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
    scipy \
    torchdata==0.7.1

# 3. MLFFモデルをインストール
RUN pip3 install \
    chgnet \
    matgl \
    mattersim \
    orb-models \
    nequip-allegro \
    torch-geometric

# DGLのインストール (PyTorch 2.6 / CUDA 12.4 用の公式バイナリ)
RUN pip3 install dgl -f https://data.dgl.ai/wheels/torch-2.6/cu124/repo.html

# Clone the MatRIS repository as it's not available on PyPI
RUN git clone https://github.com/HPC-AI-Team/MatRIS.git /opt/MatRIS
ENV PYTHONPATH="/opt/MatRIS"

# Download CHGNet-r2SCAN (or PBE as available) model
RUN git clone https://github.com/materialyzeai/matgl.git /opt/matgl_temp && \
    mkdir -p /opt/models && \
    cp -r /opt/matgl_temp/pretrained_models/CHGNet-MatPES-PBE-2025.2.10-2.7M-PES /opt/models/CHGNet_r2SCAN && \
    rm -rf /opt/matgl_temp

# 4. MACEをソースからインストール
RUN git clone https://github.com/ACEsuit/mace.git /mace
WORKDIR /mace
RUN pip3 install -e .
WORKDIR /workspace

# --- プロジェクト用ディレクトリの作成 ---
RUN mkdir -p /workspace/simulation_projects

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
