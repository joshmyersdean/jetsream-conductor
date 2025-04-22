#!/bin/bash

CONFIG_FILE=${1:-config.yaml}

# Parse config
MASTER_ADDR=$(yq -r '.master_addr' "$CONFIG_FILE")
MASTER_PORT=$((10000 + RANDOM % 55000))
NNODES=$(yq -r '.nnodes' "$CONFIG_FILE")
NPROC_PER_NODE=$(yq -r '.nproc_per_node' "$CONFIG_FILE")
LOCAL_DIR=$(yq -r '.local_dir' "$CONFIG_FILE")
REMOTE_DIR_RAW=$(yq -r '.remote_dir' "$CONFIG_FILE")
ENV_PATH_RAW=$(yq -r '.env_path' "$CONFIG_FILE")
TRAIN_SCRIPT=$(yq -r '.train_script' "$CONFIG_FILE")
SSH_USER=$(yq -r '.ssh_user' "$CONFIG_FILE")
SSH_KEY=$(yq -r '.ssh_key' "$CONFIG_FILE")
SHOULD_SYNC=$(yq -r '.sync_code' "$CONFIG_FILE")

# Expand tilde if present
SSH_KEY="${SSH_KEY/#\~/$HOME}"
LOCAL_DIR="${LOCAL_DIR/#\~/$HOME}"
REMOTE_DIR_BASENAME=$(basename "$REMOTE_DIR_RAW")
ENV_PATH="\$HOME/${REMOTE_DIR_BASENAME}/${ENV_PATH_RAW}"
REMOTE_DIR="\$HOME/${REMOTE_DIR_BASENAME}"

PUBLIC_IPS=($(yq -r '.nodes[].public_ip' "$CONFIG_FILE"))
PRIVATE_IPS=($(yq -r '.nodes[].private_ip' "$CONFIG_FILE"))

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Nodes: $NNODES"
echo "GPUs per node: $NPROC_PER_NODE"
echo "Remote directory: $REMOTE_DIR_RAW"
echo "Training script: $TRAIN_SCRIPT"
echo "Environment path: $ENV_PATH_RAW"
echo "Sync code: $SHOULD_SYNC"

if [ "$SHOULD_SYNC" = "true" ]; then
    echo "Syncing code to all nodes..."
    for ip in "${PUBLIC_IPS[@]}"; do
        echo "  -> syncing to $ip"
        rsync -az -e "ssh -i ${SSH_KEY}" --delete --exclude='.venv' "$LOCAL_DIR"/ ${SSH_USER}@${ip}:${REMOTE_DIR_RAW}
    done
else
    echo "Skipping code sync"
fi

echo "Launching training..."
for i in "${!PUBLIC_IPS[@]}"; do
    PUBLIC_IP=${PUBLIC_IPS[$i]}
    PRIVATE_IP=${PRIVATE_IPS[$i]}
    NODE_RANK=$i

    echo "Killing any existing torchrun on $PUBLIC_IP..."
    ssh -i "$SSH_KEY" "$SSH_USER@$PUBLIC_IP" "pkill -f torchrun || true"

    echo "Launching on $PUBLIC_IP (rank $NODE_RANK)..."
    ssh -i "$SSH_KEY" "$SSH_USER@$PUBLIC_IP" bash -l -c "'
        cd $REMOTE_DIR || { echo \"[Node $NODE_RANK] cd failed\"; exit 1; }

        if [ -d ".venv" ]; then
            echo "[Node $NODE_RANK] Found existing virtual environment"
        else
            echo "[Node $NODE_RANK] Creating virtual environment..."

            if ! command -v uv &> /dev/null; then
                echo "[Node $NODE_RANK] Installing uv..."
                curl -LsSf https://astral.sh/uv/install.sh | sh
            fi
            
            source ~/.bashrc

            uv venv .venv

            source .venv/bin/activate

            uv pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision 
        fi


        echo \"[Node $NODE_RANK] PWD: \$(pwd)\"

        if ! source $ENV_PATH; then
            echo \"[Node $NODE_RANK] failed to activate env\"
            exit 1
        fi
        echo \"[Node $NODE_RANK] launching torchrun...\"

        nohup torchrun \
            --nproc_per_node=$NPROC_PER_NODE \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            $TRAIN_SCRIPT > output_rank${NODE_RANK}_nohup.log 2>&1 < /dev/null &

        echo \"[Node $NODE_RANK] torchrun launched\"
    '"
done

wait
echo "All jobs launched."
