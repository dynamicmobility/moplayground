"""Unitree Go2 Interface"""

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
TORSO       = 'base'

GROUND_GEOM_ID = _interface_model.geom(GROUND_GEOM).id
TORSO_ID       = _interface_model.body(TORSO).id

NDOF = 12

# Feet in body order (FL, FR, RL, RR).
# NOTE: qpos/qvel joints go FL, FR, RL, RR, but actuators (ctrl) go FR, FL, RR, RL.
# TODO: add a joint <-> actuator order mapping once the env needs it.
FEET = ['FL', 'FR', 'RL', 'RR']
FEET_GEOMS = FEET
FEET_SITES = [f'{foot}_foot' for foot in FEET]
FEET_GEOM_IDS = [_interface_model.geom(geom).id for geom in FEET_GEOMS]
FEET_SITE_IDS = [_interface_model.site(site).id for site in FEET_SITES]

# Real-robot sensors (from go2_mjx.xml)
BASE_GYRO          = 'gyro'
BASE_ACCELEROMETER = 'accelerometer'
BASE_ORIENTATION   = 'orientation'

# Sim-only sensors (global ones from go2_mjx.xml, the rest from scene_mjx.xml).
# Use only in rewards or privileged obs, never in the policy obs.
BASE_GLOBAL_POS    = 'global_position'
BASE_GLOBAL_LINVEL = 'global_linvel'
BASE_GLOBAL_ANGVEL = 'global_angvel'
BASE_LOCAL_LINVEL  = 'local_linvel'
FEET_GLOBAL_POS    = [f'{foot}_foot_global_pos' for foot in FEET]
FEET_GLOBAL_LINVEL = [f'{foot}_foot_global_linvel' for foot in FEET]

# Menagerie 'home' keyframe, with base height raised from 0.27 so the feet
# (sphere radius 0.0175) start on the floor instead of 0.014 m inside it.
DEFAULT_JT = [0.0, 0.9, -1.8] * 4  # qpos order: FL, FR, RL, RR
DEFAULT_FF = [0.0, 0.0, 0.284, 1.0, 0.0, 0.0, 0.0]

# TODO: add noise levels (e.g. GYRO_NOISE, QPOS_NOISE) and thresholds
# (e.g. CONTACT_THRESHOLD, MIN_BASE_HEIGHT) once the env needs them.
# Bruce's values do not fit Go2 (e.g. MIN_BASE_HEIGHT = 0.2).

##################################
# TODO: add get_raw_contacts / get_ground_contact once a foot touch sensor is
# chosen (see the TODO in model/scene_mjx.xml).

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
    """Return the base orientation quaternion in the world frame."""
    return mjx_env.get_sensor_data(mj_model, data, BASE_ORIENTATION)

##################################
# Sim-only helpers

def get_base_global_pos(mj_model: mj.MjModel, data: mjx.Data) -> jax.Array:
    """Return the base (imu site) position in the world frame."""
    return mjx_env.get_sensor_data(mj_model, data, BASE_GLOBAL_POS)

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
    """Return the foot positions in the world frame, shape (4, 3), order FL, FR, RL, RR."""
    return _np.vstack([
        mjx_env.get_sensor_data(mj_model, data, sensor)
        for sensor in FEET_GLOBAL_POS
    ])

def get_foot_vel(_np, mj_model: mj.MjModel, data: mjx.Data) -> jax.Array:
    """Return the foot linear velocities in the world frame, shape (4, 3), order FL, FR, RL, RR."""
    return _np.vstack([
        mjx_env.get_sensor_data(mj_model, data, sensor)
        for sensor in FEET_GLOBAL_LINVEL
    ])
