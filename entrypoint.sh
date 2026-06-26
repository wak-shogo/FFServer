#!/bin/bash
# set -e  # Disabled to prevent container exit on compilation failure

# --- No background model compilation needed for CHGNet ---

# supervisordを起動して、定義された全てのプロセス（Jupyter, worker, streamlit）の管理を開始
exec supervisord -n -c /etc/supervisor/conf.d/app.conf
