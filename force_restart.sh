#!/bin/bash
# force_restart.sh
# このスクリプトはアプリ全体をクリーンな状態にして再起動します。

PROJECTS_DIR="/workspace/simulation_projects"
QUEUE_FILE="$PROJECTS_DIR/queue.json"
CURRENT_JOB_FILE="$PROJECTS_DIR/current_job.json"
REALTIME_DATA_FILE="$PROJECTS_DIR/realtime_data.csv"
CANCEL_FLAG_FILE="$PROJECTS_DIR/cancel_job.flag"

echo "Force restarting application..."

# 0. プロジェクトディレクトリの作成（存在しない場合）
mkdir -p "$PROJECTS_DIR"

# 1. 実行中のジョブとキューを削除
rm -f "$QUEUE_FILE"
rm -f "$CURRENT_JOB_FILE"
rm -f "$REALTIME_DATA_FILE"
rm -f "$CANCEL_FLAG_FILE"

# 2. Supervisor経由で全プロセスを再起動
# supervisord自体を再起動するのではなく、管理下のプロセスをリスタート
supervisorctl restart all

echo "Restart command issued successfully."
