"""Joystick gait tracking for Atalante."""

from typing import Any, Dict, Optional, Union

import jax
from ml_collections import config_dict
from mujoco import mjx
import numpy as np
np.set_printoptions(linewidth=300)

from mujoco_playground._src import mjx_env
from mujoco import mjx

from moplayground.envs.locomotion.bruce import interface_westwood as bruce
from moplayground.envs.locomotion.generic.navigait import NaviGait
from moplayground.envs.locomotion.control.bezier import Leg

from minimal_mjx.utils import geometry as geo

class Bruce(NaviGait):
    """
    A class for tracking joystick-controlled gait in a simulated environment.
    """

    def __init__(
        self,
        gaitlib_path: str,
        env_params: config_dict.ConfigDict,
        gait_type='P2',
        backend='jnp',
        idealistic=False,
        animate=False
    ):  
        # Initialize the parent class
        super().__init__(
            xml_path          = bruce.OFFICIAL_XML,
            env_params        = env_params,
            num_states        = bruce.NDOF,
            backend           = backend,
            gait_type         = gait_type,
            gaitlib_path      = gaitlib_path,
            animate           = animate
        )
        if idealistic:
            self.params.domain_randomization.enabled = False
            self.params.domain_randomization.obs_delay.enabled = False
            self.params.domain_randomization.action_delay.enabled = False
            self.params.noise_scale = 0.0
            self.params.curriculum.enabled = False
            self.params.start_stance = 'left'
            # self.params.initialization.strategy = 'manual'
            self.params.initialization.add_random_jt.enabled = False
            self.params.initialization.random_jt_calibration.enabled = False
            self.params.command.enabled = False
            
        self.nq = geo.FREE3D_POS + bruce.NDOF
        self.nv = geo.FREE3D_VEL + bruce.NDOF
        jt_lb_crank = bruce.full2crank(self._np, self._mj_model.jnt_range[1:, 0].T)
        jt_ub_crank = bruce.full2crank(self._np, self._mj_model.jnt_range[1:, 1].T)
        self._jt_lims = self._np.vstack((jt_lb_crank[np.newaxis, :], jt_ub_crank[np.newaxis, :]))
    
    def reset(
        self,
        rng: jax.Array,
        num_resets: int = 0,
    ):
        
        rng, gaitlib, global_hzd_qpos, global_hzd_qvel = self.initialization_randomization(
            rng          = rng, 
            default_qpos = self._np.hstack([bruce.DEFAULT_FF, bruce.DEFAULT_JT]),
            ndof         = bruce.NDOF
        )
        
        rng, ff_key, jt_key = self._split(rng, 3)
        if self.params.initialization.add_random_yaw:
            global_hzd_qpos = self.add_random_ff(ff_key, global_hzd_qpos)
        
        global_hzd_qpos = self._np.where(
            self.params.initialization.add_random_jt.enabled,
            self.add_random_joint_state(
                jt_key = jt_key, 
                qpos   = global_hzd_qpos,
                minval = self.params.initialization.add_random_jt.minval,
                maxval = self.params.initialization.add_random_jt.maxval
            ),
            global_hzd_qpos
        )
        
        motor_targets_linkage = self._np.hstack((
            global_hzd_qpos[geo.FREE3D_POS:],
            global_hzd_qvel[geo.FREE3D_VEL:]
        ))

        parent_state = super().reset(
            rng                = rng, 
            gaitlib            = gaitlib,
            global_qpos        = bruce.ext(self._np, bruce.crank2full, global_hzd_qpos, geo.FREE3D_POS),
            global_qvel        = bruce.ext(self._np, bruce.crank2full, global_hzd_qvel, geo.FREE3D_VEL),
            ctrl               = motor_targets_linkage,
            ndof               = bruce.NDOF,
            torso_id           = bruce.TORSO_ID,
            floor_id           = bruce.GROUND_GEOM_ID,
            num_resets         = num_resets,
            njoint             = bruce.NJOINT
        )
        parent_state.info['rng'], noisy_gyro, noisy_accel, raw_contacts = self.get_sensor_values(
            parent_state.info['rng'],
            parent_state.data,
            curr_level=self.get_curriculum_level(parent_state.info)
        )
        

        his_len = self.params.history_length
        obs_delay = self.params.domain_randomization.obs_delay
        updated_info = self.update_info(
            parent_state      = parent_state,
            global_hzd_qpos   = global_hzd_qpos,
            global_hzd_qvel   = global_hzd_qvel,
            noisy_gyro        = noisy_gyro,
            noisy_accel       = noisy_accel,
            qpos_history_len  = his_len + obs_delay.qpos[1] + 1,
            gyro_history_len  = his_len + obs_delay.gyro[1] + 1,
            accel_history_len = his_len + obs_delay.accel[1] + 1,
        )
        updated_info['rng'], key = self._split(parent_state.info['rng'])
        obs_delay = self.params.domain_randomization.obs_delay
        observation_delay = self._np.round(self._uniform(
            key,
            minval=obs_delay.gyro[0],
            maxval=obs_delay.gyro[1]
        )).astype(self._np.int32)
        updated_info['rng'], key = self._split(parent_state.info['rng'])
        action_delay = self._np.round(self._uniform(
            key,
            minval=self.params.domain_randomization.action_delay.val[0],
            maxval=self.params.domain_randomization.action_delay.val[1]
        )).astype(self._np.int32)
        updated_info['obs_delay'] = observation_delay * self._np.array(obs_delay.enabled).astype(self._np.int32)
        updated_info['act_delay'] = action_delay * self._np.array(self.params.domain_randomization.action_delay.enabled).astype(self._np.int32)
        
        obs = self._get_obs(
            time        = parent_state.data.time,
            info        = updated_info,
            ndof        = bruce.NDOF
        )
        return parent_state.replace(obs=obs, info=updated_info)

    def get_sensor_values(
        self,
        rng,
        data,
        curr_level
    ):
        rng, noisy_gyro = self.noisy(
            rng,
            value      = bruce.get_gyro(self._mj_model, data),
            lim        = self.params.noise_scale * bruce.GYRO_NOISE,
            curr_level = curr_level
        )
        rng, noisy_accel = self.noisy(
            rng,
            value      = bruce.get_accelerometer(self._mj_model, data),
            lim        = self.params.noise_scale * bruce.ACCEL_NOISE,
            curr_level = curr_level
        )

        raw_contacts = bruce.get_raw_contacts(
            self._np,
            self._mj_model,
            data,
            threshold=bruce.CONTACT_THRESHOLD
        )
        return rng, noisy_gyro, noisy_accel, raw_contacts

    def get_gait_qpos_init(
        self,
        vdes_gaitlib,
        random_seed
    ):
        rng = jax.random.PRNGKey(random_seed)
        gaitlib = self.set_gaitlib(
            rng          = rng,
            vdes_gaitlib = vdes_gaitlib,

        )
        
        ff_state = gaitlib.ff_evaluate(0)[:geo.FREE3D_POS]
        jt_state = gaitlib(0)[:bruce.NDOF]
        ff_vel = gaitlib.ff_evaluate(0)[geo.FREE3D_POS:]
        jt_vel = gaitlib(0)[bruce.NDOF:]
        return self._np.hstack((ff_state, jt_state)), self._np.hstack((ff_vel, jt_vel))
    
    def reset_ctrl(
        self,
        initial_vdes: np.ndarray,
        global_hzd_qpos: np.ndarray,
        gyro: np.ndarray,
        accel: np.ndarray,
        random_seed: int,
        policy
    ):
        rng = jax.random.PRNGKey(random_seed)
        gaitlib = self.set_gaitlib(
            rng          = rng,
            vdes_gaitlib = initial_vdes[:2]
        )
        
        ff_state = gaitlib.ff_evaluate(0)[:geo.FREE3D_POS]
        jt_state = gaitlib(0)[:bruce.NDOF]
        desired_qpos = self._np.hstack((ff_state, jt_state))
        mj_qvel = self._np.zeros(self.mjx_model.nv)
        global_hzd_qvel = self._np.zeros(bruce.NDOF + geo.FREE3D_VEL)
        
        initial_ctrl = gaitlib(0)

        if self._np.linalg.norm(desired_qpos - global_hzd_qpos) < 0.1:
            # TODO: Change to warning if this is not always an error
            print('WARNING: gait initial qpos is different from actual')
        
        parent_state = super().reset(
            rng          = rng, 
            gaitlib      = gaitlib,
            global_qpos  = bruce.ext(self._np, bruce.crank2full, global_hzd_qpos, geo.FREE3D_POS),
            global_qvel  = mj_qvel,
            ctrl         = initial_ctrl,
            ndof         = bruce.NDOF,
            njoint       = bruce.NJOINT,
            torso_id     = bruce.TORSO_ID,
            floor_id     = bruce.GROUND_GEOM_ID,
            num_resets   = 0
        )

        his_len = self.params.history_length
        obs_delay = self.params.domain_randomization.obs_delay
        updated_info = self.update_info(
            parent_state    = parent_state,
            global_hzd_qpos = global_hzd_qpos,
            global_hzd_qvel = global_hzd_qvel,
            noisy_gyro      = gyro,
            noisy_accel     = accel,
            qpos_history_len  = his_len + obs_delay.qpos[1] + 1,
            gyro_history_len  = his_len + obs_delay.gyro[1] + 1,
            accel_history_len = his_len + obs_delay.accel[1] + 1
        )
        updated_info['obs_delay'] = 0
        updated_info['act_delay'] = 0

        initial_obs = self._get_obs(
            time         = parent_state.data.time,
            info         = updated_info,
            ndof         = bruce.NDOF
        )
        self.policy = policy

        updated_info['jt_offset'] = self._np.zeros(bruce.NDOF)
        # updated_info['jt_offset'][4] = self.params.ankle_offsets[0]
        # updated_info['jt_offset'][9] = self.params.ankle_offsets[1]
        initial_info = updated_info

        return initial_obs, initial_info        

    def step(
        self, state: mjx_env.State, res_action: jax.Array
    ) -> mjx_env.State:
        # Check if the environment has been reset
        state = self.check_reset(state)
        state, rand_model = self.handle_env_customization(state, bruce.TORSO_ID)

        # Pre-process the action to get the motor targets
        res_jt = res_action[:bruce.NDOF]
        res_vel = res_action[bruce.NDOF:bruce.NDOF+2]
        recurrent_states = res_action[bruce.NDOF+2:]
        motor_targets, info = self.pre_process_action(
            time             = state.data.time,
            info             = state.info,
            res_joints       = res_jt,
            res_vel          = res_vel,
            recurrent_states = recurrent_states,
            ndof             = bruce.NDOF
        )
        # motor_targets = self._splice(info['motor_target_history'], (info['act_delay'], 0), (1, bruce.NDOF*2)).flatten()
        motor_targets_linkage = self._np.hstack([
            bruce.crank2bear(self._np, motor_targets[:bruce.NDOF]),
            bruce.crank2bear(self._np, motor_targets[bruce.NDOF:])
        ])
        # Step the simulation
        data = self._step_fn(state.data, motor_targets_linkage, model=rand_model)
        
        if self.animate:
            jtpos_des = bruce.crank2full(self._np, state.info['gait_history'][0, :bruce.NDOF])
            data      = self.animate_data(state, data, jtpos_des)
        if False:
            data.qpos[2] = 1.0
            data.qvel[:geo.FREE3D_VEL] = 0.0
            data.qacc[:geo.FREE3D_VEL] = 0.0
        

        # Compute and update contact information
        state.info['rng'], noisy_gyro, noisy_accel, raw_contacts = self.get_sensor_values(
            state.info['rng'],
            data,
            curr_level = self.get_curriculum_level(state.info)
        )

        # Update the internal state of the environment
        updated_info = self.update_internal_state(
            time                  = data.time,
            qpos                  = bruce.ext(self._np, bruce.full2crank, data.qpos, geo.FREE3D_POS),
            qvel                  = bruce.ext(self._np, bruce.full2crank, data.qvel, geo.FREE3D_VEL),
            info                  = info,
            res_joints            = res_jt,
            right_ground_contact  = False,
            left_ground_contact   = False,
            gyro                  = noisy_gyro,
            accel                 = noisy_accel,
            ndof                  = bruce.NDOF,
            qpos_noise            = self.params.noise_scale * bruce.QPOS_NOISE,
            qvel_noise            = self.params.noise_scale * bruce.QVEL_NOISE,
            attitude_noise        = self.params.noise_scale * bruce.ATTITUDE_NOISE
        )

        # Randomize training variables
        updated_info = self._cond(
            self.params.command.enabled,
            self.randomize_velocity,
            lambda data_info: data_info[1],
            (data, updated_info.copy())
        )

        # Compute the observation
        obs = self._get_obs(
            info         = updated_info,
            time         = data.time,
            ndof         = bruce.NDOF
        )

        # Check if done (fallen over)
        done = self.get_fall_termination(
            data            = data, 
            base_height_min = bruce.MIN_BASE_HEIGHT,  
        )
        
        # Compute rewards
        reward, metrics = self.get_reward_and_metrics(
            data, updated_info, state.metrics, state.done, res_action
        )
        
        done = done.astype(self._np.float32)
        new_state = self._state_init_fn(data, obs, reward, done, metrics, updated_info)
        return new_state
    
    @property
    def action_size(self) -> int:
        return bruce.NDOF + 2 + self.params.recurrent_states
    
    def get_ctrl(
        self,
        time: float,
        ext_crank_pos: np.ndarray,
        ext_crank_vel: np.ndarray,
        info: dict[str | np.ndarray],
        gyro: np.ndarray,
        accel: np.ndarray,
    ) -> jax.Array:
        """Returns the control signal for the environment."""
        
        # Update the internal state of the environment        
        updated_info: dict = self.update_internal_state(
            time                 = time,
            qpos                 = ext_crank_pos,
            qvel                 = ext_crank_vel,
            info                 = info,
            res_joints           = info['act_history'][0, :bruce.NDOF],
            left_ground_contact  = False,
            right_ground_contact = False,
            gyro                 = gyro,
            accel                = accel,
            ndof                 = bruce.NDOF,
            qpos_noise           = 0.0,
            qvel_noise           = 0.0,
            attitude_noise       = 0.0
        )

        # Compute the observation
        obs = self._get_obs(
            info         = updated_info,
            time         = time,
            ndof         = bruce.NDOF
        )

        # Run an inference and preprocess the outputs
        res, _              = self.policy(obs, None)
        res_joints          = self._np.array(res[:bruce.NDOF])
        res_vel             = self._np.array(res[bruce.NDOF:bruce.NDOF+2])
        recurrent_states    = self._np.array(res[bruce.NDOF+2:])
        
        motor_targets, updated_info = self.pre_process_action(
            time                = time,
            info                = updated_info,
            res_joints          = res_joints,
            res_vel             = res_vel,
            ndof                = bruce.NDOF,
            recurrent_states    = recurrent_states
        )

        return motor_targets, updated_info.copy()

    def reward_function(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        done: jax.Array,
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        sigmas = self.params.reward.sigmas
        base_des = info['base_history'][1]
        gait_des = info['gait_history'][1]
        global_qpos_act = bruce.ext(
            self._np,
            bruce.full2crank,
            data.qpos,
            geo.FREE3D_POS
        )
        # rel_qpos_act = self.get_relative_qpos(
        #     qpos        = global_qpos_act,
        #     base2global = geo.inv_transform(self._np, info['old_base2global'])
        # )
        global_base = geo.apply_transform(
            _np = self._np,
            qpos = base_des,
            offset = info['old_base2global']
        )
        global_init_qpos = geo.apply_transform(
            self._np,
            base_des,
            info['transform_init']
        )
        ground_contact = bruce.get_ground_contact(
            self._np, 
            bruce.get_raw_contacts(
                self._np,
                self.mj_model,
                data,
                threshold=bruce.CONTACT_THRESHOLD
            )
        )
        rewards = {
            'gait_tracking' : self.reward_euclidean_imitation(
                qpos            = global_qpos_act[geo.FREE3D_POS:],
                reference       = gait_des[:bruce.NDOF],
                imitation_sigma = sigmas.gait_tracking
            ),
            'base_xyz_tracking': self.reward_euclidean_imitation(
                qpos            = global_qpos_act[:3],
                reference       = global_base[:3],
                imitation_sigma = sigmas.base_xyz_tracking
            ),
            'base_quat_tracking': self.exp_reward(
                mag2  = geo.quat_dist(
                            _np = self._np,
                            q1  = global_qpos_act[3:geo.FREE3D_POS],
                            q2  = global_base[3:geo.FREE3D_POS]
                )**2,
                sigma = sigmas.base_quat_tracking
            ),
            'base_vel_tracking': self.reward_euclidean_imitation(
                qpos            = data.qvel[:2],
                reference       = base_des[geo.FREE3D_POS:geo.FREE3D_POS + 2],
                imitation_sigma = sigmas.base_vel_tracking
            ),
            'minimize_energy': self.reward_vector_size(
                v            = data.qfrc_actuator[geo.FREE3D_VEL:],
                size_sigma   = sigmas.minimize_energy
            ),
            'vel_residual_size' : self.reward_vector_size(
                v          = self._vel_scale * info['act_history'][0, bruce.NDOF:],
                size_sigma = sigmas.vel_residual_size
            ),
            'vel_residual_rate' : self.reward_residual_vel_rate(
                vdes_res            = self._vel_scale * info['act_history'][0, bruce.NDOF:],
                last_vdes_res       = self._vel_scale * info['act_history'][1, bruce.NDOF:],
                vdes_res_rate_sigma = sigmas.vel_residual_rate
            ),
            'res_jt_rate': self.reward_action_rate(
                act           = self._jt_scale * info['act_history'][0, :bruce.NDOF],
                last_act      = self._jt_scale * info['act_history'][1, :bruce.NDOF],
                last_last_act = self._jt_scale * info['act_history'][2, :bruce.NDOF],
            ),
            'foot_contact': self.reward_foot_contact(
                ground_contact  = ground_contact,
                swing_foot      = info['gaitlib'].swing_leg
            ),
            'foot_impact': self.cost_impact_velocity(
                ground_contact = ground_contact,
                foot_velocity  = bruce.get_foot_vel(
                    self._np,
                    self.mj_model,
                    data
                )
            ),
            'alive': self.reward_alive(),
            'global_xy_tracking': self.reward_euclidean_imitation(
                qpos            = global_qpos_act[:3],
                reference       = global_init_qpos[:3],
                imitation_sigma = sigmas.global_xy_tracking
            ),
            'global_z_tracking': self.exp_reward(
                mag2            = geo.quat_dist(
                    _np = self._np,
                    q1  = global_qpos_act[3:geo.FREE3D_POS],
                    q2  = global_init_qpos[3:geo.FREE3D_POS]
                )**2,
                sigma = sigmas.global_z_tracking
            ),
        }
        return rewards
    
    
    def reward_foot_contact(
        self,
        ground_contact: jax.Array,
        swing_foot: jax.Array
    ) -> jax.Array:
        """Reward for foot contact."""
        feet_des = self._np.array([Leg.LEFT, Leg.RIGHT]) == swing_foot
        # print(f'ground contact: {ground_contact}, feet des: {feet_des}')
        contact_reward = self._np.sum(feet_des == ground_contact.astype(self._np.float32))
        
        return contact_reward / 2.0
    
    def cost_impact_velocity(
        self,
        ground_contact: jax.Array,
        foot_velocity: jax.Array
    ):
        fz = foot_velocity[:, -1]
        return self._np.linalg.norm(fz * ground_contact)