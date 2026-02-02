"""Bruce Interface"""

from pathlib import Path

# JAX and Mujoco imports
import jax
import jax.numpy as jnp
import numpy as np
import mujoco as mj
from mujoco_playground._src.dm_control_suite import common
from mujoco import mjx
from mujoco_playground._src import mjx_env
import pathlib
INTERFACE_PATH = pathlib.Path(__file__).resolve().parent
OFFICIAL_XML = Path(f'{INTERFACE_PATH}') / 'model/flat_scene_westwood.xml'

_interface_model = mj.MjModel.from_xml_path(
    OFFICIAL_XML.as_posix(), common.get_assets()
)

GROUND_GEOM = 'floor'
TORSO       = 'base_link'
RIGHT_FOOT  = 'ankle_pitch_link_r'
LEFT_FOOT  = 'ankle_pitch_link_l'

GROUND_GEOM_ID  = _interface_model.geom(GROUND_GEOM).id
TORSO_ID        = _interface_model.body(TORSO).id
RIGHT_FOOT_ID   = _interface_model.body(RIGHT_FOOT).id
LEFT_FOOT_ID    = _interface_model.body(LEFT_FOOT).id
RF_COLLISION_ID = _interface_model.geom('right_foot_collision').id 
LF_COLLISION_ID = _interface_model.geom('left_foot_collision').id 

NDOF = 10
NJOINT = 18

FEET_SENSORS = [
    'opto1',
    'opto2',
    'opto3',
    'opto4',
]

HEIGHT_FR = 'right_foot'
HEIGHT_FL = 'left_foot'
FOOT_VEL_R = 'right_foot_vel'
FOOT_VEL_L = 'left_foot_vel'

FEET_SITES = [
    '0',
    '1',
    '2',
    '3',
    'midfoot_r',
    'midfoot_l',
]

# right foot sites
RF_SITES = [
    'opto1',
    'opto2',
]

# left foot sites 
LF_SITES = [
    'opto3',
    'opto4',
]

BASE_GYRO = 'base_gyro'
BASE_ACCELEROMETER = 'base_accelerometer'
LF_ZAXIS = 'left_foot_z'
RF_ZAXIS = 'right_foot_z'
BASE_ORIENTATION = 'base_orientation'
BASE_VEL = 'base_velocimeter'

# DEFAULT_JT = [0.0] * 10
# DEFAULT_JT = [-0.008,  0.469,  0.018, -0.947, -0.469,  0.008,  0.469, -0.018, -0.947, -0.469]
DEFAULT_JT = [-0.008,  0.469,  0.018, -0.947, -0.469, 
               0.008,  0.469, -0.018, -0.947, -0.469]
DEFAULT_FF = [0.0, 0.0, 0.455, 1.0, 0.0, 0.0, 0.0]

ACCEL_NOISE       = 0.1
GYRO_NOISE        = 0.03
QPOS_NOISE        = 0.02
QVEL_NOISE        = 0.1
ATTITUDE_NOISE    = 0.05 # (in rad)
CONTACT_THRESHOLD = 0.1
MIN_BASE_HEIGHT   = 0.2

# bear2joint
T_bear2joint = np.array([[1,    0,    0,  0,  0],
                         [ 0, -0.5, +0.5,  0,  0],
                         [ 0, -0.5, -0.5,  0,  0],
                         [ 0,    0,    0,  1,  0],
                         [ 0,    0,    0, -1,  1]])

T_joint2bear = np.linalg.inv(T_bear2joint)
# print(T_joint2bear)
# quit

def ext(_np, func, q, num_free):
    free_q = q[:num_free]
    robot_q = q[num_free:]
    transformed_q = func(_np, robot_q)
    return _np.hstack([free_q, transformed_q])

##################################
def crank2pitch(
    _np,
    jt_pos: jax.Array
):
    """Convert crank to  pitch"""
    return _np.hstack((jt_pos[0:4], -jt_pos[3] + jt_pos[4], jt_pos[5:9], -jt_pos[8] + jt_pos[9]))

def pitch2crank(
    _np,
    jt_pos: jax.Array
):
    """Convert pitch to crank"""
    return _np.hstack((jt_pos[0:4], jt_pos[4] + jt_pos[3], jt_pos[5:9], jt_pos[9] + jt_pos[8]))

##################################
def full2bear(
    _np,
    jt_pos
):
    def by_leg(leg_jt):
        hip_yaw = leg_jt[0]
        hip_diff = leg_jt[7:9]
        knee_pitch = leg_jt[3]
        ankle_crank = leg_jt[6]
        return _np.hstack([hip_yaw, hip_diff, knee_pitch, ankle_crank])
    
    bears = _np.hstack([by_leg(jt_pos[:9]), by_leg(jt_pos[9:])])
    return bears

def bear2full(
    _np,
    bears
):
    def by_leg(leg_bears, leg_pitch, leg_crank):
        ankle_passive = _np.hstack([leg_crank[3] - leg_crank[4], leg_crank[4]])
        return _np.hstack([leg_pitch, ankle_passive, leg_bears[1:3]])
    
    pitch = bear2pitch(_np, bears)
    crank = pitch2crank(_np, pitch)
    full = _np.hstack([
        by_leg(bears[:5], pitch[:5], crank[:5]),
        by_leg(bears[5:], pitch[5:], crank[5:])
    ])
    return full

##################################
def bear2pitch(
    _np,
    bears
):
    return _np.hstack([T_bear2joint @ bears[:5], T_bear2joint @ bears[5:]])

def pitch2bear(
    _np,
    pitch
):
    return _np.hstack([T_joint2bear @ pitch[:5], T_joint2bear @ pitch[5:]])


##################################
# def bear2crank(
#     _np,
#     bears
# ):
#     pitch = bear2pitch(_np, bears)
#     crank = crank2pitch(_np, pitch)
#     return crank

def crank2bear(
    _np,
    crank
):
    pitch = crank2pitch(_np, crank)
    bears = pitch2bear(_np, pitch)
    return bears


##################################
def full2crank(
    _np,
    jt_pos: jax.Array
):
    """Convert full to crank"""
    def by_leg(leg_jt):
        hip_yaw = leg_jt[0]
        hip_diff = leg_jt[7:9]
        knee_pitch = leg_jt[3]
        ankle_crank = leg_jt[6]
        return _np.hstack([hip_yaw, hip_diff, knee_pitch, ankle_crank])
    
    bears = _np.hstack([by_leg(jt_pos[:9]), by_leg(jt_pos[9:])])
    pitch = bear2pitch(_np, bears)
    crank = pitch2crank(_np, pitch)
    return crank

def crank2full(
    _np,
    crank: jax.Array
):
    pitch = crank2pitch(_np, crank)
    bears = _np.hstack([
        T_bear2joint @ pitch[:5],
        T_bear2joint @ pitch[5:]
    ])
    return bears

##################################
def full2pitch(
    _np,
    jt_pos
):
    bears = full2bear(_np, jt_pos)
    pitch = _np.hstack([T_bear2joint @ bears[:5], T_bear2joint @ bears[5:]]) 
    return pitch


def pitch2full(
    _np,
    pitch
):
    bears = pitch2bear(_np, pitch)
    full = bear2full(_np, bears)
    return full

##################################
def full2crank(
    _np,
    jt_pos
):
    pitch = full2pitch(_np, jt_pos)
    crank = pitch2crank(_np, pitch)
    return crank

def crank2full(
    _np,
    crank
):
    pitch = crank2pitch(_np, crank)
    full = pitch2full(_np, pitch)
    return full

def get_raw_contacts(
    _np,
    mj_model: mj.MjModel,
    data: mjx.Data,
    threshold: float,
) -> jax.Array:
    """Return the contact state of the feet."""
    raw_contacts = _np.array([
        mjx_env.get_sensor_data(mj_model, data, sensor)[0]
        for sensor in FEET_SENSORS
    ]) > threshold
    return raw_contacts

def get_ground_contact(_np, raw_contacts: jax.Array) -> jax.Array:
    """Return the ground contact state of the feet."""
    right = _np.any(raw_contacts[:2])
    left = _np.any(raw_contacts[2:])
    return _np.array([right, left])

def get_gravity(_np, accel: jax.Array) -> jax.Array:
    """Return the gravity vector in the world frame."""
    return _np.array([accel @ _np.array([0, 0, -1])])

def get_accelerometer(mj_model: mj.MjModel, data: mjx.Data) -> jax.Array:
    """Return the accelerometer readings in the local frame."""
    return mjx_env.get_sensor_data(mj_model, data, BASE_ACCELEROMETER)

def get_gyro(mj_model: mj.MjModel, data: mjx.Data) -> jax.Array:
    """Return the gyroscope readings in the local frame."""
    return mjx_env.get_sensor_data(mj_model, data, BASE_GYRO)

def get_base_orientation(mj_model: mj.MjModel, data: mjx.Data) -> jax.Array:
    """Return the gyroscope readings in the local frame."""
    return mjx_env.get_sensor_data(mj_model, data, BASE_ORIENTATION)

def get_foot_pos(_np, mj_model: mj.MjModel, data: mjx.Data,) -> jax.Array:
    """Return the accelerometer readings in the local frame."""
    # return mjx_env.get_sensor_data(mj_model, data, )
    foot_height = _np.vstack((mjx_env.get_sensor_data(mj_model, data, HEIGHT_FR), mjx_env.get_sensor_data(mj_model, data, HEIGHT_FL)))
    return foot_height

def get_foot_vel( _np, mj_model: mj.MjModel, data: mjx.Data,) -> jax.Array:
    """Return the accelerometer readings in the local frame."""
    # return mjx_env.get_sensor_data(mj_model, data, )
    foot_vel = _np.vstack((mjx_env.get_sensor_data(mj_model, data, FOOT_VEL_R), mjx_env.get_sensor_data(mj_model, data, FOOT_VEL_L)))
    return foot_vel

def get_left_z_axis( _np, mj_model: mj.MjModel, data: mjx.Data,):
    return mjx_env.get_sensor_data(mj_model, data, LF_ZAXIS)

def get_right_z_axis( _np, mj_model: mj.MjModel, data: mjx.Data,) ->jax.Array:
    return mjx_env.get_sensor_data(mj_model, data, RF_ZAXIS)

def get_body_vel( _np, mj_model: mj.MjModel, data: mjx.Data,) ->jax.Array:
    vel = mjx_env.get_sensor_data(mj_model, data, BASE_VEL)
    return vel
def get_right_z_axis( _np, mj_model: mj.MjModel, data: mjx.Data,) ->jax.Array:
    return mjx_env.get_sensor_data(mj_model, data, RF_ZAXIS)
