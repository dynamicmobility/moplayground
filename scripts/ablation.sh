#!/bin/bash

# Run a MORLAX ablation sweep inside a tmux session.
#
# Usage:
#   bash scripts/ablation.sh config/mocheetah.yaml
#   bash scripts/ablation.sh config/mocheetah.yaml --hypertypes single,dual --samplings dense,sparse-heavytail --ks 4,8,16
#
# All arguments after the first are forwarded verbatim to scripts.ablation,
# so you can pass --skip-existing or override the sweep grid.

SESSION_NAME="MOPLAYGROUND_ABLATION"
PYTHON_SCRIPT="ablation"
MODULE_NAME="scripts"
CONDA_ENV_NAME="moplayground"
YAML_FILE=$1
shift
EXTRA_ARGS="$@"

if [ -z "$YAML_FILE" ]; then
    echo "Usage: bash scripts/ablation.sh <base_yaml> [extra args forwarded to scripts.ablation]"
    exit 1
fi

tmux new-session -d -s "$SESSION_NAME"

tmux send-keys -t "$SESSION_NAME" "conda activate $CONDA_ENV_NAME" C-m
tmux send-keys -t "$SESSION_NAME" "wandb login" C-m
tmux send-keys -t "$SESSION_NAME" "python3 -m $MODULE_NAME.$PYTHON_SCRIPT --base $YAML_FILE $EXTRA_ARGS" C-m

tmux attach-session -t "$SESSION_NAME"
