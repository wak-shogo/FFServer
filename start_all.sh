#!/bin/bash

# このスクリプト (start_all.sh) があるディレクトリに移動
cd "$(dirname "$0")"

# コンテナ名を定義
CONTAINER_NAME="my_calc_env"

# 以前のコンテナが残っている場合、停止して削除
echo "Stopping and removing old container if it exists..."
docker stop $CONTAINER_NAME >/dev/null 2>&1
docker rm $CONTAINER_NAME >/dev/null 2>&1

echo "Starting Docker container ($CONTAINER_NAME) in background..."
# 1. Docker コンテナをバックグラウンド (-d) で起動し、名前 (--name) をつける
#    $(pwd) を /workspace にマウントし、設定ファイルを上書きマウントします。
docker run --gpus all --runtime=nvidia -d --name $CONTAINER_NAME \
  -p 8888:8888 -p 8501:8501 \
  -v $(pwd):/workspace \
  -v $(pwd)/supervisord.conf:/etc/supervisor/conf.d/app.conf \
  -v $(pwd)/entrypoint.sh:/usr/local/bin/entrypoint.sh \
  -v $(pwd)/matris_cache:/root/.cache/matris \
  nequip-olm-jupyter

echo "---"
echo "✅ Setup complete!"
echo "All services are managed by Supervisor inside the container."
echo "Jupyter Lab should be running at: http://localhost:8888"
echo "Streamlit App should be running at: http://localhost:8501"
echo "---"

# (オプション) 自動でブラウザを開く場合 (WSL環境で動作する wslview を使用)
# echo "Opening browsers..."
# wslview http://localhost:8888
# wslview http://localhost:8501
