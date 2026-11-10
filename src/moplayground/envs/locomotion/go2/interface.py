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
IMU_SITE         = 'imu'

GROUND_GEOM_ID = _interface_model.geom(GROUND_GEOM).id
TORSO_ID       = _interface_model.body(TORSO).id
IMU_SITE_ID    = _interface_model.site(IMU_SITE).id

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
UPVECTOR = 'upvector'

# Noise scaling values for sensor data
JOINT_VEL_NOISE = 1.5
JOINT_ANGLE_NOISE = 0.03
GYRO_NOISE = 0.2
ACCEL_NOISE = 0.2
GRAVITY_NOISE = 0.05


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

def get_base_global_pos(mj_model: mj.MjModel, data: mjx.Data) -> jax.Array:
    """Return the base (imu site) position in the world frame."""
    return mjx_env.get_sensor_data(mj_model, data, BASE_GLOBAL_POS)

def get_ground_reaction_force(_np, mj_model: mj.MjModel, data: mjx.Data) -> jax.Array:
    """Return the net ground reaction force on the robot in the world frame, shape (3,), N.

    From Newton's second law, ``F = m * (a - g)``: the accelerometer reads
    ``a - g`` in the imu frame, which is rotated into the world frame and
    scaled by the total mass. Uses the base's acceleration as the robot's,
    so leg swing adds some error. Standing still gives about ``(0, 0, m*g)``.
    """
    mass = _np.sum(mj_model.body_mass)
    R = data.site_xmat[IMU_SITE_ID].reshape(3, 3)
    return mass * (R @ get_accelerometer(mj_model, data))

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

def get_foot_dist(_np, mj_model: mj.MjModel, data: mjx.Data) -> jax.Array:
    """Return the base-to-foot distances ``||p_foot - p_base||``, shape (4,), order FL, FR, RL, RR.

    The base is the imu site. The distance is the same in any frame.
    """
    foot_pos = get_foot_pos(_np, mj_model, data)
    return _np.linalg.norm(foot_pos - get_base_global_pos(mj_model, data), axis=-1)

def get_feet_contact(_np, data: mjx.Data) -> jax.Array:
    """Return whether each foot touches the floor, shape (4,) bool, order FL, FR, RL, RR.

    Checks ``data.contact`` for a foot-floor contact with ``dist < 0`` (penetrating).
    """
    geom = data.contact.geom                                    # (ncon, 2)
    touching = data.contact.dist < 0                            # (ncon,)
    with_floor = (geom == GROUND_GEOM_ID).any(axis=1)           # (ncon,)
    feet = _np.array(FEET_GEOM_IDS)[:, None]                    # (4, 1)
    has_foot = (geom[None, :, 0] == feet) | (geom[None, :, 1] == feet)  # (4, ncon)
    return (has_foot & (with_floor & touching)[None]).any(axis=1)

def get_upvector(mj_model: mj.MjModel, data: mjx.Data) -> jax.Array:
    """Return the base (imu site) z-axis in the world frame, shape (3,)."""
    return mjx_env.get_sensor_data(mj_model, data, UPVECTOR)

