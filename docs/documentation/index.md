# MO-Playground

MO-Playground is a library for massively parallelized **multi-objective reinforcement
learning (MORL)** for robotics. It provides GPU-accelerated multi-objective
environments built on [MuJoCo Playground](https://playground.mujoco.org), and two
multi-objective extensions of [Brax](https://github.com/google/brax) PPO that learn a
whole Pareto front of policies in a single training run.

<div style="display: flex; gap: 0.5rem; flex-wrap: wrap; justify-content: center;">
  <video src="../static/videos/cheetah-run.mp4" width="320" autoplay loop muted playsinline></video>
  <video src="../static/videos/cheetah-energy.mp4" width="320" autoplay loop muted playsinline></video>
</div>
<p style="text-align: center;"><em>The multi-objective Cheetah with all weight on running speed (left)
and on energy efficiency (right).</em></p>

## Concepts

- **Vector rewards.** Each environment returns one reward per objective. The config
  file groups named reward terms into objectives, so the same environment can be
  trained on different objective sets without code changes.
- **Tradeoffs.** A tradeoff (also called a *directive*) is a weight vector on the
  probability simplex, one weight per objective. Policies are conditioned on it, and
  the scalarized reward `tradeoff · reward` drives each PPO update.
- **Algorithms.** The `algorithm:` key of the config selects one of three trainers:

    | Algorithm | Policy | Module |
    | --- | --- | --- |
    | `morlax` | A hypernetwork maps the tradeoff to the weights of the policy and value MLPs | [`moppo.morlax`][moplayground.moppo.morlax] |
    | `amor` | The tradeoff is concatenated to the observation of a single policy and value MLP | [`moppo.amor`][moplayground.moppo.amor] |
    | `ppo` | Single-objective PPO on a fixed scalarization (`default_scalarization`) | `minimal_mjx` |

- **Built on minimal_mjx.** Environments, the training loop, checkpointing and
  rollouts come from [minimal_mjx](https://dynamicmobility.github.io/minimal-mjx/).
  MO-Playground adds the multi-objective layer on top.

## Install

```bash
pip install moplayground                     # from PyPI
```

To train on a GPU, or to work on the code, install from source into the provided
conda environment:

```bash
git clone https://github.com/dynamicmobility/moplayground.git
cd moplayground
conda env create -f environment.yml          # Linux / CUDA (mac_environment.yml on macOS)
conda activate moplayground
pip install -e ".[dev]"
```

Training was tested on Ubuntu 22.04 with Python 3.12 and CUDA 13. Rollouts and
evaluation run on the CPU.

## Quick start

From the repository root:

```bash
# Train MORLAX on the multi-objective Cheetah
python -m scripts.train config/mocheetah.yaml

# Roll out the trained policy at a chosen tradeoff and save a video
python -m scripts.rollout <save_dir>/<name>/config.yaml --tradeoff 1 0 0
```

## Environments

| `env:` | Class | Objectives in the provided config |
| --- | --- | --- |
| `MOCheetah` | [`MOCheetah`][moplayground.envs.dmcontrol.cheetah.MOCheetah] | running speed, energy efficiency, height |
| `MOHopper` | [`MOHopper`][moplayground.envs.dmcontrol.hopper.MOHopper] | running speed, jump height |
| `MOWalker` | [`MOWalker`][moplayground.envs.dmcontrol.walker.MOWalker] | running speed, energy efficiency |
| `MOAnt` | [`MOAnt`][moplayground.envs.dmcontrol.ant.MOAnt] | x velocity, y velocity |
| `MOHumanoid` | [`MOHumanoid`][moplayground.envs.dmcontrol.humanoid.MOHumanoid] | speed, energy efficiency |
| `NaviGait` | [`Bruce`][moplayground.envs.locomotion.bruce.navigait.Bruce] | joint and base tracking, arm swing, rigid arms, smoothness, efficiency |

## Where to go next

<div class="grid cards" markdown>

-   **[Training a policy](examples/training.md)**

    A walk through `scripts/train.py`, and what a finished run directory contains.

-   **[Rolling out a policy](examples/rollout.md)**

    Roll out a trained policy at a chosen tradeoff, or a dummy zero/random controller.

-   **[Writing an environment](guides/environment.md)**

    Subclass `MultiObjectiveBase` and define objectives for your own robot.

-   **[Writing an algorithm](guides/algorithm.md)**

    The contract a multi-objective trainer must meet, and the building blocks provided.

-   **[API reference](api/index.md)**

    Every public function and class, organized by module.

</div>

## Citation

Janwani, Novoseller, Lawhern and Tucker. *MO-Playground: Massively Parallelized
Multi-Objective Reinforcement Learning for Robotics.* IEEE Robotics and Automation
Letters, 2026. [doi:10.1109/LRA.2026.3700381](https://doi.org/10.1109/LRA.2026.3700381),
[arXiv:2603.09237](https://arxiv.org/abs/2603.09237v1).

```bibtex
@ARTICLE{moplayground,
  author={Janwani, Neil and Novoseller, Ellen and Lawhern, Vernon J. and Tucker, Maegan},
  journal={IEEE Robotics and Automation Letters},
  title={MO-Playground: Massively Parallelized Multi-Objective Reinforcement Learning for Robotics},
  year={2026},
  volume={11},
  number={8},
  pages={9431-9438},
  keywords={Algorithms;International trade;Learning (artificial intelligence);Optimization;Training;Graphics processing units;Reinforcement learning;Timing;Conferences;Vectors;Reinforcement learning;machine learning for robot control;humanoid and bipedal locomotion},
  doi={10.1109/LRA.2026.3700381}}
```
