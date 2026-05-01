#!/bin/bash
# set -e  # Disabled to prevent container exit on compilation failure

# Function to run model check in background
run_model_check() {
    echo "Background task started: Waiting for Streamlit to be ready..."
    
    # Wait for Streamlit port 8501
    count=0
    while ! timeout 1 bash -c "echo > /dev/tcp/localhost/8501" 2>/dev/null; do
        sleep 2
        count=$((count+2))
        if [ $count -ge 60 ]; then
             echo "Timeout waiting for Streamlit port 8501. Proceeding anyway..."
             break
        fi
    done
    
    if [ $count -lt 60 ]; then
        echo "Streamlit is up! Proceeding with model check."
    fi

    # Debug: Check what the container sees
    echo "Checking for NequIP model file..."
    ls -la /workspace/NequipOLM_model/ || echo "Directory /workspace/NequipOLM_model/ not found"

    # Check if model exists and has size > 0 (-s)
    if [ ! -s /workspace/NequipOLM_model/NequIP-OAM-L.nequip.pt2 ]; then
        echo "Model not found or empty. Compiling NequIP model..."
        # Try to compile, ignore failure
        nequip-compile \
            nequip.net:mir-group/NequIP-OAM-L:0.1 \
            /workspace/NequipOLM_model/NequIP-OAM-L.nequip.pt2 \
            --mode aotinductor \
            --device cuda \
            --target ase || echo "WARNING: NequIP model compilation failed. Continuing..."
    else
        echo "Model already exists at /workspace/NequipOLM_model/NequIP-OAM-L.nequip.pt2. Skipping compilation."
    fi
}

# Run the model check in the background
run_model_check &

# supervisordを起動して、定義された全てのプロセス（Jupyter, worker, streamlit）の管理を開始
exec supervisord -n -c /etc/supervisor/conf.d/app.conf
