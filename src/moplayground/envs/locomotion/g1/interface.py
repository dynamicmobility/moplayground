"""Unitree Go2 Interface"""
# TODO: rename the docstring to "Unitree G1 Interface" once the port is done.
# This file is a copy of go2/interface.py. Each "TODO (G1 port)" below marks a
# value or helper that is still the Go2 version.

import math
from pathlib import Path

# JAX and Mujoco imports
import jax
import mujoco as mj
from mujoco_playground._src.dm_control_suite import common
from mujoco import mjx
from mujoco_playground._src import mjx_env
import pathlib
INTERFACE_PATH = pathlib.Path(__file__).resolve().parent
OFFICIAL_XML = Path(f'{INTERFACE_PATH}') / 'model/scene_mjx.xml'

_interface_model = mj.MjModel.from_xml_path(
    OFFICIAL_XML.as_posix(), common.get_assets()
)

GROUND_GEOM = 'floor'
TORSO       = 'pelvis'
IMU_SITE    = 'imu_in_pelvis'  # pelvis IMU; get_gravity reads its frame

GROUND_GEOM_ID = _interface_model.geom(GROUND_GEOM).id
TORSO_ID       = _interface_model.body(TORSO).id
IMU_SITE_ID    = _interface_model.site(IMU_SITE).id

NDOF = _interface_model.nu  # 29: one position actuator per hinge joint

# Not used: the observation space does not need foot geom or site IDs.
# TODO (G1 port): if needed later, use the G1 names: sites 'left_foot' and
# 'right_foot', and 3 floor-contact capsules per foot
# ('{side}_foot{1,2,3}_collision'). The Go2 joint <-> actuator mapping note
# does not apply: the G1 actuator order equals the qpos order.
# Feet in body order (FL, FR, RL, RR).
# NOTE: qpos/qvel joints go FL, FR, RL, RR, but actuators (ctrl) go FR, FL, RR, RL.
# TODO: add a joint <-> actuator order mapping once the env needs it.
# FEET = ['FL', 'FR', 'RL', 'RR']
# FEET_GEOMS = FEET
# FEET_SITES = [f'{foot}_foot' for foot in FEET]
# FEET_GEOM_IDS = [_interface_model.geom(geom).id for geom in FEET_GEOMS]
# FEET_SITE_IDS = [_interface_model.site(site).id for site in FEET_SITES]

# Real-robot sensors (from g1_mjx.xml, pelvis IMU)
BASE_GYRO          = 'gyro_pelvis'
BASE_ACCELEROMETER = 'accelerometer_pelvis'
BASE_ORIENTATION   = 'orientation_pelvis'
UPVECTOR           = 'upvector_pelvis'

# TODO (G1 port): Go2 noise values. Tune them for the G1 sensors.
# Noise scaling values for sensor data
JOINT_VEL_NOISE = 1.5
JOINT_ANGLE_NOISE = 0.03
GYRO_NOISE = 0.2
ACCEL_NOISE = 0.2
GRAVITY_NOISE = 0.05


# Sim-only sensors (from g1_mjx.xml, pelvis IMU).
# Use only in rewards or privileged obs, never in the policy obs.
# BASE_GLOBAL_POS    = 'global_position'  # not used: no base position sensor
BASE_GLOBAL_LINVEL = 'global_linvel_pelvis'
BASE_GLOBAL_ANGVEL = 'global_angvel_pelvis'
BASE_LOCAL_LINVEL  = 'local_linvel_pelvis'
# Foot sensors on the 'left_foot' / 'right_foot' sites, order left, right.
FEET_GLOBAL_POS    = ['left_foot_global_pos', 'right_foot_global_pos']
FEET_GLOBAL_LINVEL = ['left_foot_global_linvel', 'right_foot_global_linvel']
# Foot touch sensors on the 'left_sole' / 'right_sole' sites, order left, right.
# Each reads the summed normal contact force (N); threshold it for a 1/0 flag.
FEET_CONTACT       = ['left_foot_contact', 'right_foot_contact']

# Menagerie 'home' keyframe (model/scene_mjx.xml). The first 7 qpos values are
# the free joint (position, quaternion); the rest are the joints in qpos order.
# The keyframe's ctrl equals its joint qpos, so actuator order = qpos order.
_HOME_QPOS = _interface_model.keyframe('home').qpos
DEFAULT_FF = _HOME_QPOS[:7].tolist()
DEFAULT_JT = _HOME_QPOS[7:].tolist()

# Fall check: end the episode when get_tilt_angle reaches this (rad).
MAX_TILT_ANGLE = math.radians(60.0)

# TODO: add noise levels (e.g. GYRO_NOISE, QPOS_NOISE) and thresholds
# (e.g. CONTACT_THRESHOLD, MIN_BASE_HEIGHT) once the env needs them.
# Bruce's values do not fit Go2 (e.g. MIN_BASE_HEIGHT = 0.78).
# TODO (G1 port): reword the note above for the G1 (home base height 0.78 m).

##################################
# TODO: add get_raw_contacts / get_ground_contact that read FEET_CONTACT and
# compare against a CONTACT_THRESHOLD.

def get_gravity(_np, data) -> jax.Array:
    """Return the gravity vector in the body frame."""
    return data.site_xmat[IMU_SITE_ID].reshape(3,3).T @ _np.array([0, 0, -1])

def get_accelerometer(mj_model: mj.MjModel, data: mjx.Data) -> jax.Array:
    """Return the accelerometer readings in the local frame."""
    return mjx_env.get_sensor_data(mj_model, data, BASE_ACCELEROMETER)

def get_gyro(mj_model: mj.MjModel, data: mjx.Data) -> jax.Array:
    """Return the gyroscope readings in the local frame."""
    return mjx_env.get_sensor_data(mj_model, data, BASE_GYRO)

def get_base_orientation(mj_model: mj.MjModel, data: mjx.Data) -> jax.Array:
    """Return the base orientation quaternion in the world frame."""
    return mjx_env.get_sensor_data(mj_model, data, BASE_ORIENTATION)

##################################
# Sim-only helpers

# def get_base_global_pos(mj_model: mj.MjModel, data: mjx.Data) -> jax.Array:
#     """Return the base (imu site) position in the world frame."""
#     return mjx_env.get_sensor_data(mj_model, data, BASE_GLOBAL_POS)

def get_base_global_linvel(mj_model: mj.MjModel, data: mjx.Data) -> jax.Array:
    """Return the base (imu site) linear velocity in the world frame."""
    return mjx_env.get_sensor_data(mj_model, data, BASE_GLOBAL_LINVEL)

def get_base_global_angvel(mj_model: mj.MjModel, data: mjx.Data) -> jax.Array:
    """Return the base (imu site) angular velocity in the world frame."""
    return mjx_env.get_sensor_data(mj_model, data, BASE_GLOBAL_ANGVEL)

def get_base_local_linvel(mj_model: mj.MjModel, data: mjx.Data) -> jax.Array:
    """Return the base (imu site) linear velocity in the local frame."""
    return mjx_env.get_sensor_data(mj_model, data, BASE_LOCAL_LINVEL)

def get_foot_pos(_np, mj_model: mj.MjModel, data: mjx.Data) -> jax.Array:
    """Return the foot positions in the world frame, shape (2, 3), order left, right."""
    return _np.vstack([
        mjx_env.get_sensor_data(mj_model, data, sensor)
        for sensor in FEET_GLOBAL_POS
    ])

def get_foot_vel(_np, mj_model: mj.MjModel, data: mjx.Data) -> jax.Array:
    """Return the foot linear velocities in the world frame, shape (2, 3), order left, right."""
    return _np.vstack([
        mjx_env.get_sensor_data(mj_model, data, sensor)
        for sensor in FEET_GLOBAL_LINVEL
    ])

def get_upvector(mj_model: mj.MjModel, data: mjx.Data) -> jax.Array:
    """Return the base (imu site) z-axis in the world frame, shape (3,)."""
    return mjx_env.get_sensor_data(mj_model, data, UPVECTOR)

def get_tilt_angle(_np, mj_model: mj.MjModel, data: mjx.Data) -> jax.Array:
    """Return the angle (rad) between the base z-axis and the world z-axis.

    0 when upright, pi/2 when the base z-axis is parallel to the ground,
    pi when upside down.
    """
    up_z = get_upvector(mj_model, data)[-1]
    return _np.arccos(_np.clip(up_z, -1.0, 1.0))
