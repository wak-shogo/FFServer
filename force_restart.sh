#!/bin/bash
# force_restart.sh
# このスクリプトはアプリ全体を完全に停止させ、クリーンな状態にしてから再起動します。

PROJECTS_DIR="/workspace/simulation_projects"
QUEUE_FILE="$PROJECTS_DIR/queue.json"
CURRENT_JOB_FILE="$PROJECTS_DIR/current_job.json"
REALTIME_DATA_FILE="$PROJECTS_DIR/realtime_data.csv"
CANCEL_FLAG_FILE="$PROJECTS_DIR/cancel_job.flag"

echo "=== Force Restarting Application (Deep Clean) ==="

# 1. Supervisor自体の管理プロセスを停止
# これにより、管理下のプロセスが自動で再起動されるのを防ぎます
supervisorctl stop all

# 2. 残存している可能性のある全てのPythonプロセスを強制終了
# 並列計算などでフォークされた子プロセス（ゾンビプロセス）を確実に掃除します
echo "Killing all remaining python processes..."
pkill -9 python3
pkill -9 python

# 3. 実行中のジョブとキュー、フラグファイルを削除
echo "Clearing job queue and flags..."
mkdir -p "$PROJECTS_DIR"
rm -f "$QUEUE_FILE"
rm -f "$CURRENT_JOB_FILE"
rm -f "$REALTIME_DATA_FILE"
rm -f "$CANCEL_FLAG_FILE"

# 4. 数秒待機してポートやメモリを確実に解放
sleep 2

# 5. プロセスを再起動
echo "Starting all services via Supervisor..."
supervisorctl start all

echo "=== Restart sequence finished successfully ==="
