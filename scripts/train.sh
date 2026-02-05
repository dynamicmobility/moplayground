#!/bin/bash

SESSION_NAME="MOPLAYGROUND"
PYTHON_SCRIPT="train"
MODULE_NAME="scripts"
CONDA_ENV_NAME="moplayground"
YAML_FILE=$1

# Start a new tmux session (detached) with the specified name
tmux new-session -d -s "$SESSION_NAME"

# Send command to run the Python script with the YAML file as an argument
tmux send-keys -t "$SESSION_NAME" "conda activate $CONDA_ENV_NAME" C-m
tmux send-keys -t "$SESSION_NAME" "wandb login" C-m
tmux send-keys -t "$SESSION_NAME" "python3 -m $MODULE_NAME.$PYTHON_SCRIPT $YAML_FILE" C-m

# Attach to the session (optional, remove if you want to keep it detached)
tmux attach-session -t "$SESSION_NAME"