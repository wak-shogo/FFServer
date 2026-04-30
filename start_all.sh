#!/bin/bash

# このスクリプトがあるディレクトリに移動
cd "$(dirname "$0")"

CONTAINER_NAME="my_calc_env"
IMAGE_NAME="nequip-olm-jupyter"

echo "=== System Check ==="
echo "Directory: $(pwd)"
echo "Docker: $(docker --version 2>/dev/null || echo 'Not found')"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo 'NVIDIA GPU not detected or nvidia-smi failed')"
echo "===================="

# --- 権限の強制付与 ---
echo "Setting executable permissions for scripts..."
chmod +x entrypoint.sh start_all.sh start_app.sh build_container.sh
# ----------------------

# イメージの存在確認
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
  echo "❌ Error: Docker image '$IMAGE_NAME' not found."
  echo "Please run './build_container.sh' first."
  exit 1
fi

echo "Stopping and removing old container if it exists..."
docker stop $CONTAINER_NAME >/dev/null 2>&1
docker rm $CONTAINER_NAME >/dev/null 2>&1

echo "Starting Docker container ($CONTAINER_NAME)..."

# 1. 現代的な標準 ( --gpus all のみ) で試行
echo "Attempt 1: Starting with '--gpus all'..."
if docker run --gpus all -d --name $CONTAINER_NAME \
  -p 8888:8888 -p 8501:8501 \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/supervisord.conf":/etc/supervisor/conf.d/app.conf \
  -v "$(pwd)/entrypoint.sh":/usr/local/bin/entrypoint.sh \
  -v "$(pwd)/matris_cache":/root/.cache/matris \
  $IMAGE_NAME; then
    echo "✅ Started successfully with GPU support."
else
    echo "⚠️ Attempt 1 failed. Trying Attempt 2 (No GPU support)..."
    docker stop $CONTAINER_NAME >/dev/null 2>&1
    docker rm $CONTAINER_NAME >/dev/null 2>&1
    
    # 2. フォールバック: GPUなしで起動
    if docker run -d --name $CONTAINER_NAME \
      -p 8888:8888 -p 8501:8501 \
      -v "$(pwd)":/workspace \
      -v "$(pwd)/supervisord.conf":/etc/supervisor/conf.d/app.conf \
      -v "$(pwd)/entrypoint.sh":/usr/local/bin/entrypoint.sh \
      -v "$(pwd)/matris_cache":/root/.cache/matris \
      $IMAGE_NAME; then
        echo "⚠️ Started WITHOUT GPU support. Performance will be degraded."
    else
        echo "❌ Critical Error: Could not start container even without GPU."
        exit 1
    fi
fi

echo "---"
echo "Waiting for services to initialize (5s)..."
sleep 5

echo "=== Container Process Status (Supervisor) ==="
docker exec $CONTAINER_NAME supervisorctl status || echo "Could not retrieve status."
echo "============================================="

echo "Setup complete!"
echo "Jupyter Lab: http://localhost:8888"
echo "Streamlit App: http://localhost:8501"
