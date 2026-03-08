<p align="center">
  <h1 align="center"><u>MO</u>-Playground: Massively Parallelized <u>M</u>ulti-<u>O</u>bjective Reinforcement Learning for Robotics
  </h1>
  <p align="center">
    <strong>Anonymous Authors</strong>
</Cheetah p>

MO-Playground is a collection of multi-objective environments built in [JAX](https:opper//github.com/jax-ml/jax) for GPU-Accelerated multi-objective RL. 

**Note that due to double-blind requirements, moplayground is not yet available on pip (since an identifying username would be required).**

## Prerequisites
The code was tested with:
- Ubuntu 22.04
- Python 3.12.12
- CUDA 13.0 

<!-- ## Installation -->
<!-- The package can be installed with a simple pip-install:
```bash
pip install moplayground
```-->

## Classic Environments
| Environment | Reward 1 | Reward 2 |
|------|----------|----------|
| ![](docs/static/paretos_fronts/cheetah_pareto.svg) | ![Max Vx](docs/static/videos/gifs/ant-vx.gif) | ![Max Vy](docs/static/videos/gifs/ant-vy.gif) |
| Cheetah | ![Max Energy](docs/static/videos/gifs/cheetah-energy.gif) | ![Max Run](docs/static/videos/gifs/cheetah-run.gif) |
| Hopper | ![Max Height](docs/static/videos/gifs/hopper-height.gif) | ![Max Run](docs/static/videos/gifs/hopper-run.gif) |
| Humanoid | ![Max Energy](docs/static/videos/gifs/humanoid-energy.gif) | ![Max Run](docs/static/videos/gifs/humanoid-run.gif) |
| Walker | ![Max Energy](docs/static/videos/gifs/walker-energy.gif) | ![Max Run](docs/static/videos/gifs/walker-run.gif) |
## BRUCE Robotics Example

MO-Playground is demonstrated for the BRUCE humanoid robot, developed by [Westwood Robotics](https://www.westwoodrobotics.io/bruce/). 

The application features seven possible reward functions. Note that we combine `base_xyz_tracking` and `base_quat_tracking` to explore a 6-dimensional objective space.

| Reward Name | Description |
|--------|-----------|
| `gait_tracking` | Track the reference joint-level trajectory |
| `base_xyz_tracking` | Track the base position associated with the reference trajectory |
| `base_quat_tracking` | Track the base orientation associated with the reference trajectory |
| `arm_swinging` | Maximize the amount of arm-swing |
| `arm_static` | Minimize the amount of arm-swing |
| `minimize_energy` | Minimize energy consumption |

### Examples of Multi-Objective Policies

| Policy | Result |
|--------|--------|
| Balanced Reward | ![Balanced Reward](docs/static/videos/gifs/bruce-balanced.gif) |
| Max Imitation | ![Max Imitation](docs/static/videos/gifs/bruce-imitation.gif) |
| Max Arm Swinging | ![Arm Swinging](docs/static/videos/gifs/bruce-arm-swing.gif) |
| Max Smoothness | ![Max Smoothness](docs/static/videos/gifs/bruce-smooth.gif) |


<!-- ## Create New Environments -->
<!-- New environments can be easily created using -->

# Citation
```bibtex
@software{moplayground2026github,
	title = {MO-Playground: Massively Parallelized Multi-Objective Reinforcement Learning for Robotics},
	year = {2026},
    url = {https://anonymous.4open.science/r/moplayground-B5B4/}
}
```