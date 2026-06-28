# ==========================================================
# Stage 1: Build React frontend
# ==========================================================
FROM node:20-alpine AS frontend-builder
WORKDIR /app
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ==========================================================
# Stage 2: Main Python/PyTorch runtime
# ==========================================================
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# apt-get実行時にインタラクティブなプロンプトを無効化
ENV DEBIAN_FRONTEND=noninteractive

# 基本的なツールのインストール
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /workspace

# 2. その他の基本パッケージをインストール (PyPIから)
RUN pip3 install --default-timeout=1000 --no-cache-dir \
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
    supervisor \
    seaborn \
    MDAnalysis \
    scipy \
    torchdata==0.7.1 \
    fastapi \
    uvicorn \
    python-multipart

# 3. MLFFモデルをインストール (CHGNetのみ)
RUN pip3 install --default-timeout=1000 --no-cache-dir chgnet

# 4. PyTorchをCUDA 12.8ビルドにアップグレードして Blackwell (RTX 5090) をサポート
RUN pip3 install --no-cache-dir --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --trusted-host download.pytorch.org

# 5. Copy and install MatRIS (local copy includes Hugging Face download URL patches)
COPY MatRIS /opt/MatRIS
RUN pip3 install --no-deps /opt/MatRIS

# Copy built React frontend from Stage 1
COPY --from=frontend-builder /app/dist /opt/frontend/dist

# --- プロジェクト用ディレクトリの作成 ---
RUN mkdir -p /workspace/simulation_projects

# --- コンパイル処理は削除し、起動スクリプトを配置 ---
COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

# --- Supervisorの設定 ---
RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/app.conf

# --- 最終設定 ---
EXPOSE 8888 8501

# ENTRYPOINTを設定（ここを経由してCMDが実行される）
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []
