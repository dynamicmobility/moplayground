import os
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
from moplayground.eval.rollout import main
from minimal_mjx.learning.startup import read_config
from pathlib import Path

main(Path('output/videos'))