#!/bin/bash
set -e

echo "--- WSL2 Ubuntu 環境構築スクリプトを開始します ---"

# 1. システムの更新
echo "[1/5] システムパッケージを更新中..."
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# 2. Docker のインストール
echo "[2/5] Docker をインストール中..."
if ! command -v docker &> /dev/null; then
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gnupg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
else
    echo "Docker は既にインストールされています。"
fi

# 3. NVIDIA Container Toolkit のインストール
echo "[3/5] NVIDIA Container Toolkit をインストール中..."
if ! command -v nvidia-ctk &> /dev/null; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
else
    echo "NVIDIA Container Toolkit は既にインストールされています。"
fi

# 4. Docker の権限設定 (sudoなしで実行可能にする)
echo "[4/5] ユーザー権限を設定中..."
if ! groups $USER | grep &>/dev/null '\bdocker\b'; then
    sudo groupadd -f docker
    sudo usermod -aG docker $USER
    echo "ユーザー $USER を docker グループに追加しました。"
else
    echo "ユーザー $USER は既に docker グループに属しています。"
fi

# 5. Docker デーモンの設定と再起動
echo "[5/5] Docker サービスを起動中..."
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo systemctl enable docker
sudo systemctl start docker

echo "---"
echo "✅ セットアップが完了しました！"
echo "⚠️  注意: 権限設定を反映させるため、一度 WSL を再起動する必要があります。"
echo "Windows の PowerShell で 'wsl --terminate Ubuntu' を実行してから、再度 Ubuntu を開いてください。"
echo "---"
