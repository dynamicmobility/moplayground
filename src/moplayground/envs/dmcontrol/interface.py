from pathlib import Path
import mujoco as mj
from mujoco_playground._src import mjx_env
from mujoco import mjx
import jax
import os
import pathlib
INTERFACE_PATH = pathlib.Path(__file__).resolve().parent
class CheetahInterface:
    XML         = Path(INTERFACE_PATH / 'model/cheetah.xml')
    DEFAULT_FF  = [0.0, 0.0, 0.0]
    DEFAULT_JT  = [0.0] * 6
    
# class HopperInterface:
#     XML        = Path(f'{SUBMODULE_LOCATION}envs/moplayground/dmcontrol/model/hopper.xml')
#     DEFAULT_FF = [0.0, 1.0, 0.0]
#     DEFAULT_JT = [0.0] * 4
    
# class AntInterface:
#     XML        = Path(f'{SUBMODULE_LOCATION}envs/moplayground/dmcontrol/model/ant.xml')
#     DEFAULT_FF = [0.0, 0.0, 1.0] + [1.0, 0.0, 0.0, 0.0]
#     DEFAULT_JT = [0.0, 0.5,
#                   0.0, -0.5,
#                   0.0, -0.5,
#                   0.0, 0.5]
    
# class WalkerInterface:
#     XML        = Path(f'{SUBMODULE_LOCATION}envs/moplayground/dmcontrol/model/walker.xml')
#     DEFAULT_FF = [0.0, 1.5, 0.0]
#     DEFAULT_JT = [0.0] * 6

# class HumanoidInterface:
#     XML           = Path(f'{SUBMODULE_LOCATION}envs/moplayground/dmcontrol/model/humanoid.xml')
#     MJ_MODEL      = mj.MjModel.from_xml_path(XML.as_posix())
#     DEFAULT_FF    = [0.0, 0.0, 1.4, 1.0, 0.0, 0.0, 0.0]
#     DEFAULT_JT    = [0.0] * 21
#     HEAD_ID       = MJ_MODEL.body('head').id
#     TORSO_ID      = MJ_MODEL.body('torso').id
#     LH_ID         = MJ_MODEL.body('left_hand').id
#     RH_ID         = MJ_MODEL.body('right_hand').id
#     LF_ID         = MJ_MODEL.body('left_foot').id
#     RF_ID         = MJ_MODEL.body('right_foot').id
#     EXTREMITY_IDS = [LH_ID, RH_ID, LF_ID, RF_ID]

#     @classmethod
#     def com_vel(cls, data: mjx.Data) -> jax.Array:
#         """Returns the velocity of the center of mass in global coordinates."""
#         return mjx_env.get_sensor_data(cls.MJ_MODEL, data, "torso_subtreelinvel")