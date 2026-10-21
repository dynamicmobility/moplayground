# Examples

The `scripts/` directory holds the entry points for the full workflow. Run each one
from the repository root as a module (`python -m scripts.<name> ...`).

| Script | What it does | Walkthrough |
| --- | --- | --- |
| `scripts/train.py` | Trains a MORLAX, AMOR or PPO policy from a config, checkpointing as it goes | [Training a policy](training.md) |
| `scripts/rollout.py` | Rolls out a trained policy at one tradeoff, or a dummy zero/random controller, and saves a video and reward plot | [Rolling out a policy](rollout.md) |
| `scripts/train.sh` | Runs `train.py` inside a detached `tmux` session | [Training a policy](training.md#running-in-tmux) |
| `scripts/tune.py` | Runs a Weights & Biases hyperparameter sweep (configs in `config/sweeps/`) | |
| `scripts/download_model.py` | Downloads a pre-trained policy and its config from Weights & Biases | |
| `scripts/draw_frontier.py` | Evaluates a trained run over many tradeoffs and plots its Pareto front | |

Every script is driven by a YAML config. `config/` provides one per environment, with
AMOR variants in `config/amor/`; see [The config file](../guides/configuration.md).
