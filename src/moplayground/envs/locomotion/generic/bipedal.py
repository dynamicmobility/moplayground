from typing import Any, Dict, Optional, Union
from pathlib import Path
import jax
from ml_collections import config_dict
from mujoco_playground._src.dm_control_suite import common
import mujoco as mj
from mujoco import mjx
from mujoco_playground._src import mjx_env
import jax.numpy as jnp
import numpy as np

from moplayground.utils import geometry as geo
# from minimal_mjx.envs.generic.base import SwappableBase
from moplayground.envs.generic.mobase import MultiObjectiveBase

CHILD_ERROR = 'Please implement in child class'    

class BipedalBase(MultiObjectiveBase):
    """Atalante walker task."""

    def __init__(self,
                 xml_path: Path,
                 env_params: config_dict.ConfigDict,
                 backend = 'jnp',
        ):
        """Initializes the base bipedal environment for RL training and 
        evaluation. This class implements core functionalities that are common 
        among various bipedal tasks, such as the Atalante X Exoskeleton, 
        Unitree G1, Westwood robotics BRUCE, and others. Moreover, this class 
        implements a swappable backend for JAX and NumPy, allowing for 
        flexibility when it comes to training and evaluation (i.e. JAX for
        training and NumPy for evaluation).
        
        Args:
            xml_path: Path to the XML file defining the robot and environment.
            config: Configuration dictionary for the environment.
            config_overrides: Optional dictionary to override specific config values.
            backend: Backend to use for computations, either 'jnp' for JAX or 'np' for NumPy.
        """
        super().__init__(
            xml_path    = xml_path,
            env_params  = env_params,
            backend     = backend,
            num_free    = 7
        )

    def periodic_push(
        self, data, info: dict, body_id: int
    ):
        """Assigns random pushes/perturbations that turn on and off based on the
        configuration and the current state of the environment. 
        
        Args:
            state: MJX state for the robot/environment
            body_id: the body to apply the push to
        """
        # Get random keys
        info["rng"], pert_r_rng, pert_theta_rng = self._split(
            info["rng"], 3
        )

        # Compute the perturbation
        pert_on = info['push_on'].astype(self._np.int32)
        pert_t0 = info['push_t0']
        push_durations = self._np.array([info['nopush_duration'], info['push_duration']])
        pert_tau = data.time + self.params.ctrl_dt * 2 - pert_t0
        change_pert_state = (pert_tau % push_durations[pert_on]) < self.params.ctrl_dt
        info['push_t0'] = change_pert_state * data.time + (1 - change_pert_state) * pert_t0
        info['push_on'] = change_pert_state * (1 - pert_on) + (1 - change_pert_state) * pert_on
        
        curriculum = self._np.clip(
            data.time / (1 * self.params.ctrl_dt), 0, 1
        )
        push_config = self.params.push
        r_sample = self._uniform(
            key=pert_r_rng,
            shape=1,
            minval=curriculum * (1 - push_config.annulus_width),
            maxval=curriculum 
        )[0]
        theta_sample = self._uniform(
            key=pert_theta_rng,
            shape=1,
            minval=0.0,
            maxval=2 * self._np.pi
        )[0]
        pertx = push_config.major_lim * r_sample * self._np.cos(theta_sample)
        perty = push_config.minor_lim * r_sample * self._np.sin(theta_sample)
        pertx, perty = self._np.where(
            self._np.all(self._np.abs(info['vdes']) <= self._np.array(self.params.push.max_vdes)),
            self._np.array([pertx, perty]),
            self._np.zeros(2)
        )
        
        cmd_pert = info['push_override']
        pertx = (1 - cmd_pert) * pertx + cmd_pert * info['push_override_xy'][0]
        perty = (1 - cmd_pert) * perty + cmd_pert * info['push_override_xy'][1]
        rel_pert_vec = self._np.array([pertx, perty, 0.0])
        rel_pert_vec = change_pert_state * rel_pert_vec + (1 - change_pert_state) * info['curr_pert']
        rel_pert_vec = info['push_on'] * rel_pert_vec
        info['curr_pert'] = rel_pert_vec
        
        
        rpy = geo.quat2euler(self._np, data.qpos[3:7])
        rot_mat = geo.rotz(self._np, rpy[2])
        global_pert = rot_mat @ rel_pert_vec

        perturbation_vec = self._np.array([
            *global_pert, # force (x, y, z)
            0.0,   0.0,   0.0  # torque (x, y, z)    
        ])

        new_xfrc_applied = self._set_val_fn(
            arr=self._np.copy(data.xfrc_applied),
            val=perturbation_vec,
            min_idx=body_id,
            max_idx=body_id + 1
        )
        data = self._set_xfrc_fn(data, new_xfrc_applied)
        return data, info

    def randomize_velocity(
        self, data_info, override: bool = False
    ):
        data, info = data_info
        info["rng"], v_rng = self._split(
            info["rng"], 2
        )

        tau = data.time - info['vdes_t0'] + self.params.ctrl_dt * 2
        change = (tau % info['vdes_duration']) < self.params.ctrl_dt
        vel_max = self._np.array((self.params.command.lin_vel_x,
                                  self.params.command.lin_vel_y,
                                  self.params.command.ang_vel_z))
        new_v = self._uniform(
            v_rng,
            shape=(3,),
            minval=vel_max[:, 0],
            maxval=vel_max[:, 1]
        )
        info['vdes'] = change * new_v + (1 - change) * info['vdes']
        info['vdes_t0'] = change * data.time + (1 - change) * info['vdes_t0']
        
        override = self._np.array(override, dtype=self._np.bool_)
        info['vdes'] = override * new_v + (1 - override) * info['vdes']
        
        return info

    def reset(
        self,
        rng: jax.Array,
        data: mjx.Data,
        torso_id: int,
        floor_id: int,
        ndof: int,
        num_resets: int = 0,
        njoint: int | None = None,
    ) -> mjx_env.State:
        """Resets the environment to an initial state. Takes in a random key
        (rng) as input and outputs the first state of the environment."""
        # Initialize history buffers
        his_len = self.params.history_length
        qpos_history       = self._np.zeros((his_len, self.nq))
        qvel_history       = self._np.zeros((his_len, self.nv))
        noisy_qpos_history = self._np.zeros((his_len, self.nq))
        noisy_qvel_history = self._np.zeros((his_len, self.nv))
        act_history        = self._np.zeros((his_len, self.action_size))

        # Set a simple info dict...
        info = {
            # Random variable
            'rng':                rng,
            
            # Basic proprioception
            'act_history':        act_history,
            'qpos_history':       qpos_history,
            'noisy_qpos_history': noisy_qpos_history,
            'qvel_history':       qvel_history,
            'noisy_qvel_history': noisy_qvel_history,

            # Push disturbances
            'push_on':            self._np.array(1),
            'push_duration':      self.params.push.push_duration,
            'nopush_duration':    self.params.push.nopush_duration,
            'push_t0':            data.time,
            'curr_pert':          self._np.zeros(3),
            'push_override':      False,
            'push_override_xy':   self._np.zeros(2),

            # Velocity commands
            'vdes':               self._np.array(self.params.initialization.vdes),
            'vdes_t0':            data.time,
            'vdes_duration':      self.params.command.vdes_duration,

            'num_resets':         num_resets + 1
        }

        # Randomize the dynamics of the environment
        dr_params = self.randomize_dynamics(
            rng,
            info     = info,
            torso_id = torso_id,
            floor_id = floor_id,
            ndof     = ndof,
            njoint   = njoint,
        )
        info = info | dr_params
        
        reward, done = self._np.zeros(2)
        obs, metrics = None, {}
        state = self._state_init_fn(data, obs, reward, done, metrics, info)

        return state
    
    def get_bounds(
        self,
        percent_change
    ):
        return 1 + self._np.array([-percent_change, percent_change])
    
    def get_curriculum_level(
        self,
        info
    ):
        if not self.params.curriculum.enabled:
            return self.params.curriculum.scale
        
        completion = self._np.clip(
            info['num_resets'] / self.params.curriculum.scaling_iterations,
            0.0,
            1.0
        )
        return self.params.curriculum.scale * completion
    
    def make_history(self, data, length):
        history = self._np.repeat(
            data[self._np.newaxis, :],
            length,
            axis=0
        )
        
        return history
    
    def update_history(self, history, new_data):
        history = self._np.roll(history, shift=1, axis=0)
        return self._set_val_fn(history, new_data, max_idx=1)
    
    def randomize_dynamics(
        self,
        rng,
        info,
        torso_id,
        floor_id,
        ndof,
        njoint = None
    ) -> dict[str | jax.Array]:
        DR = self.params.domain_randomization 
        curr_level = self.get_curriculum_level(info)
        
        def set_random_params(
            _rng,
            dr_param,
            old_arr,
            rand_shape,
            op=lambda x, y: x * y,
            min_idx=None,
            max_idx=None
        ):
            _rng, key = self._split(_rng)
            bounds = self.get_bounds(dr_param * curr_level)
            rand_val = self._uniform(
                key    = key,
                shape  = rand_shape,
                minval = bounds[0],
                maxval = bounds[1]
            )
            new_arr = self._set_val_fn(
                arr     = old_arr,
                val     = op(old_arr[min_idx:max_idx], rand_val),
                min_idx = min_idx,
                max_idx = max_idx
            )
            
            return _rng, new_arr

        # FLOOR FRICTION
        rng, floor_friction = set_random_params(
            _rng       = rng,
            dr_param   = DR.params.floor_friction,
            old_arr    = self._mjx_model.geom_friction[floor_id],
            rand_shape = (),
            max_idx    = 1
        )
        geom_friction = self._set_val_fn(
            arr     = self._mjx_model.geom_friction,
            val     = floor_friction,
            min_idx = floor_id,
            max_idx = floor_id + 1
        )

        # DOF FRICTION
        rng, dof_frictionloss = set_random_params(
            _rng       = rng,
            dr_param   = DR.params.dof_frictionloss,
            old_arr    = self._mjx_model.dof_frictionloss,
            rand_shape = njoint,
            min_idx    = geo.FREE3D_VEL,
            max_idx    = None
        )
        
        # DOF ARMATURE
        rng, dof_armature = set_random_params(
            _rng       = rng,
            dr_param   = DR.params.dof_armature,
            old_arr    = self._mjx_model.dof_armature,
            rand_shape = njoint,
            min_idx    = geo.FREE3D_VEL,
            max_idx    = None
        )
        
        # DOF DAMPING
        rng, dof_damping = set_random_params(
            _rng       = rng,
            dr_param   = DR.params.dof_damping,
            old_arr    = self._mjx_model.dof_damping,
            rand_shape = njoint,
            min_idx    = geo.FREE3D_VEL,
            max_idx    = None
        )

        # TORSO MASS
        rng, body_mass = set_random_params(
            _rng       = rng,
            dr_param   = DR.params.torso_mass,
            old_arr    = self._mjx_model.body_mass,
            rand_shape = (),
            min_idx    = torso_id,
            max_idx    = torso_id + 1
        )

        # OTHER BODY MASS
        # TODO
        
        # TORSO COM
        rng, key = self._split(rng)
        torso_com_bounds = curr_level * self._np.array([
            -DR.params.torso_com,
            DR.params.torso_com
        ])
        torso_com = self._uniform(
            key,
            shape=(3,),
            minval=torso_com_bounds[0],
            maxval=torso_com_bounds[1]
        )
        # print('Torso CoM:', torso_com)
        body_ipos = self._set_val_fn(
            arr     = self._mjx_model.body_ipos,
            val     = self._mjx_model.body_ipos[torso_id] + torso_com,
            min_idx = torso_id,
            max_idx = torso_id + 1
        )
        
        # OTHER BODIES COM
        rng, key = self._split(rng)
        body_com_bounds = curr_level * self._np.array([
            -DR.params.body_com,
            DR.params.body_com
        ])
        body_com = self._uniform(
            key,
            shape=body_ipos.shape,
            minval=body_com_bounds[0],
            maxval=body_com_bounds[1]
        )
        except_torso = self._np.array(
            self._np.ones(self.mj_model.nbody) == torso_id
        ).astype(int)
        body_ipos = body_ipos + body_com * except_torso[:, self._np.newaxis]

        # GAINS
        rng, key = self._split(rng)
        gain_multiplier_bounds = self.get_bounds(DR.params.gain_multiplier * curr_level)
        gain_multiplier = self._uniform(
            key,
            shape=(self.mj_model.actuator_gainprm.shape[0]),
            minval=gain_multiplier_bounds[0],
            maxval=gain_multiplier_bounds[1]
        )[:, self._np.newaxis]
        actuator_gainprm = gain_multiplier * self._mj_model.actuator_gainprm
        actuator_biasprm = gain_multiplier * self._mj_model.actuator_biasprm

        # TIMECONST
        timeconst_multiplier_bounds = self.get_bounds(DR.params.timeconst * curr_level)
        timeconst = self._uniform(
            key,
            shape=(self.mj_model.actuator_dynprm.shape[0]),
            minval=timeconst_multiplier_bounds[0],
            maxval=timeconst_multiplier_bounds[1]
        )[:, self._np.newaxis]
        actuator_dynprm = timeconst * self._mj_model.actuator_dynprm

        # # FOOT COLLISION ANGLE
        # geom_id = 6
        # foot_angle_bounds = self.get_bounds(DR.params.foot_angle * curr_level)
        # foot_angle_offset = self._uniform(
        #     key,
        #     minval=foot_angle_bounds[0],
        #     maxval=foot_angle_bounds[1]
        # )
        # quat_offset = geo.euler2quat(np, np.array([0, 0, foot_angle_offset]))
        # new_quat = geo.quat_mul(np, quat_offset, model.geom_quat[geom_id])
        
        return {
            'geom_friction'       : geom_friction,
            'dof_frictionloss'    : dof_frictionloss,
            'dof_damping'         : dof_damping,
            'dof_armature'        : dof_armature,
            'body_mass'           : body_mass,
            'body_ipos'           : body_ipos,
            'actuator_gainprm'    : actuator_gainprm,
            'actuator_biasprm'    : actuator_biasprm,
            'actuator_dynprm'     : actuator_dynprm,
        }
    

    def noisy(self, rng, value: jax.Array, lim: jax.Array, curr_level: jax.Array):
        """Generates zero-mean uniform noise in [-lim, lim]. Limits are scaled
        by curr_level (curriculum level)."""
        rng, temp = self._split(rng)
        lim = curr_level * lim
        noise = self._uniform(
            key    = temp,
            shape  = value.shape,
            minval = -lim,
            maxval = lim
        )
        noisy_value = noise + value
        return rng, noisy_value
    
    def handle_env_customization(self, state: mjx_env.State, push_body_id):
        # Assign perturbations if necessary
        def apply_push(local_state): 
            data, info = self.periodic_push(
                local_state.data,
                local_state.info.copy(),
                body_id=push_body_id
            )
            return data, info
        
        def no_push(local_state):
            return local_state.data, local_state.info
        
        data, info = self._cond(
            self.params.push.enabled,
            apply_push,
            no_push,
            state
        )
        state = state.replace(data=data, info=info)
        
        def apply_rand_dynamics(mjx_model):
            """Applies random dynamics to the model."""
            temp_model = self._set_model_params_fn(
                mjx_model,
                geom_friction    = state.info['geom_friction'],
                dof_frictionloss = state.info['dof_frictionloss'],
                dof_armature     = state.info['dof_armature'],
                dof_damping      = state.info['dof_damping'],
                body_mass        = state.info['body_mass'],
                body_ipos        = state.info['body_ipos'],
                actuator_gainprm = state.info['actuator_gainprm'],
                actuator_biasprm = state.info['actuator_biasprm'],
                actuator_dynprm  = state.info['actuator_dynprm']
            )
            return temp_model

        def get_normal_dynamics(mjx_model):
            """Returns the original model without any randomization."""
            return mjx_model
        
        temp_model = self._cond(
            self.params.domain_randomization.enabled,
            apply_rand_dynamics,
            get_normal_dynamics,
            self._mjx_model
        )
        return state, temp_model
    
    def add_random_ff(self, ff_key: jax.Array, qpos: jax.Array) -> jax.Array:
        _, xy_key, yaw_key = self._split(ff_key, 3)
        dxy = self._uniform(xy_key, (2,), minval=-0.5, maxval=0.5)
        # qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        qpos = self._set_val_fn(
            arr     = qpos,
            val     = qpos[0:2] + dxy, 
            min_idx = 0,
            max_idx = 2
        )
        yaw = self._uniform(yaw_key, (1,), minval=-3.14, maxval=3.14)
        # roll = self._uniform(key, (1,), minval=-0.04, maxval=0.04)
        # pitch = self._uniform(key, (1,), minval=-0.1, maxval=0.1)
        # quat = geo.quat_mul(self._np, geo.angle2quat(self._np, self._np.array([1, 0, 0]), roll), quat)
        # quat = geo.quat_mul(self._np, geo.angle2quat(self._np, self._np.array([0, 1, 0]), pitch), quat)
        quat = geo.angle2quat(self._np, self._np.array([0, 0, 1]), yaw)
        new_quat = geo.quat_mul(self._np, quat, qpos[3:7])
        qpos = self._set_val_fn(
            arr     = qpos,
            val     = new_quat,
            min_idx = 3,
            max_idx = 7
        )
        return qpos
    
    def add_random_joint_state(
            self,
            jt_key: jax.Array,
            qpos: jax.Array,
            minval: jax.Array,
            maxval: jax.Array
        ):
        # TODO: This does not appropriately handle joints with no limits
        val = self._uniform(jt_key, qpos[self.qpos_free:].shape[0], minval=minval, maxval=maxval)
        val = self._np.clip(val + qpos[self.qpos_free:], *self._jt_lims)
        qpos = self._set_val_fn(qpos, val, min_idx=geo.FREE3D_POS, max_idx=None)
        return qpos
    
    def get_fall_termination(
            self, data: mjx.Data, base_height_min: jax.Array
        ) -> jax.Array:
        # Terminates if joint limits are exceeded or the robot falls.
        enabled = self._np.array(self.params.termination.enabled)
        fallen = self._np.array(data.qpos[2] < base_height_min)
        return enabled & fallen