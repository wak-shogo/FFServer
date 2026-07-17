#!/bin/bash

# このスクリプトがあるディレクトリに移動
cd "$(dirname "$0")"

CONTAINER_NAME="my_calc_env"
IMAGE_NAME="nequip-olm-jupyter"

echo "=== System Check ==="
echo "Directory: $(pwd)"
echo "Docker: $(docker --version 2>/dev/null || echo 'Not found')"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo 'NVIDIA GPU not detected or nvidia-smi failed')"
# GPU detection vs toolkit installation check
has_gpu=false
if nvidia-smi -L &> /dev/null; then
    has_gpu=true
fi

if [ "$has_gpu" = true ] && ! command -v nvidia-ctk &> /dev/null; then
    echo "=========================================================================="
    echo "⚠️  GPU DETECTED ON HOST, BUT DOCKER CANNOT USE IT!"
    echo "Your NVIDIA GPU was detected via nvidia-smi, but the 'nvidia-container-toolkit'"
    echo "is not installed or configured on your host system."
    echo ""
    echo "To enable GPU support in the Docker container, please run the setup script:"
    echo "    bash setup_wsl2.sh"
    echo ""
    echo "This will install and configure the necessary components (requires sudo)."
    echo "=========================================================================="
    if [ -t 0 ]; then
        echo "Press Enter to continue in CPU-only fallback mode, or Ctrl+C to abort."
        read -r
    else
        echo "Continuing in CPU-only fallback mode (non-interactive session)..."
    fi
fi

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

echo "Stopping and removing old containers if they exist..."
docker stop $CONTAINER_NAME >/dev/null 2>&1
docker rm $CONTAINER_NAME >/dev/null 2>&1
docker stop mattersim_env >/dev/null 2>&1
docker rm mattersim_env >/dev/null 2>&1

# Docker ネットワークの作成
echo "Creating docker network..."
docker network create ffserver-net >/dev/null 2>&1 || true

# MatterSim イメージの存在確認と起動
has_mattersim=false
if [[ "$(docker images -q mattersim-service 2> /dev/null)" != "" ]]; then
  has_mattersim=true
fi

if [ "$has_mattersim" = true ]; then
  echo "Starting MatterSim container (mattersim_env)..."
  mattersim_started=false
  if [ "$has_gpu" = true ]; then
    echo "Attempting to start MatterSim container with GPU support..."
    if docker run --gpus all \
      -e NVIDIA_VISIBLE_DEVICES=all \
      -e NVIDIA_DRIVER_CAPABILITIES=all \
      -d --name mattersim_env \
      --network ffserver-net \
      -p 8502:8502 \
      mattersim-service; then
        echo "✅ MatterSim started with GPU support."
        mattersim_started=true
    fi
  fi

  if [ "$mattersim_started" = false ]; then
    echo "Starting MatterSim container in CPU-only mode..."
    if docker run -d --name mattersim_env \
      --network ffserver-net \
      -p 8502:8502 \
      mattersim-service; then
        echo "✅ MatterSim started in CPU-only mode."
    else
        echo "⚠️ Warning: Could not start MatterSim container."
    fi
  fi
else
  echo "ℹ️ MatterSim image 'mattersim-service' not found. Skipping MatterSim startup."
fi

echo "Starting Docker container ($CONTAINER_NAME)..."

# 1. 現代的な標準 ( --gpus all ) で起動を試行 (最大3回リトライ)
echo "Attempting to start container with GPU support (--gpus all)..."
max_attempts=3
success=false

for ((attempt=1; attempt<=max_attempts; attempt++)); do
  echo "Attempt $attempt of $max_attempts..."
  
  # コンテナが既に残っている場合は削除
  docker stop $CONTAINER_NAME >/dev/null 2>&1
  docker rm $CONTAINER_NAME >/dev/null 2>&1

  if docker run --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e MATTERSIM_SERVER_URL=http://mattersim_env:8502 \
    --network ffserver-net \
    -d --name $CONTAINER_NAME \
    -p 8888:8888 -p 8501:8501 \
    -v "$(pwd)":/workspace \
    -v "$(pwd)/supervisord.conf":/etc/supervisor/conf.d/app.conf \
    -v "$(pwd)/entrypoint.sh":/usr/local/bin/entrypoint.sh \
    $IMAGE_NAME; then

      echo "✅ Started successfully with GPU support."
      success=true
      break
  else
      echo "⚠️ Attempt $attempt failed. Sleeping 2 seconds..."
      sleep 2
  fi
done

if [ "$success" = false ]; then
    echo "=========================================================================="
    echo "🚨 WARNING: GPU container startup failed after $max_attempts attempts."
    echo "Falling back to Attempt 2: CPU-only mode."
    echo "=========================================================================="
    
    docker stop $CONTAINER_NAME >/dev/null 2>&1
    docker rm $CONTAINER_NAME >/dev/null 2>&1
    
    # 2. フォールバック: GPUなしで起動
    if docker run -d --name $CONTAINER_NAME \
      -e MATTERSIM_SERVER_URL=http://mattersim_env:8502 \
      --network ffserver-net \
      -p 8888:8888 -p 8501:8501 \
      -v "$(pwd)":/workspace \
      -v "$(pwd)/supervisord.conf":/etc/supervisor/conf.d/app.conf \
      -v "$(pwd)/entrypoint.sh":/usr/local/bin/entrypoint.sh \
      $IMAGE_NAME; then
        echo "⚠️⚠️⚠️ Started WITHOUT GPU support. Performance will be degraded. ⚠️⚠️⚠️"
        echo "Troubleshooting GPU issues:"
        echo "  - Make sure NVIDIA Container Toolkit is installed on the host."
        echo "  - WSL2 Users: If this occurs intermittently, run 'wsl --shutdown' in Windows PowerShell and try again."
        echo "  - Check if Docker is properly integrated with WSL2 in Docker Desktop settings."
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
