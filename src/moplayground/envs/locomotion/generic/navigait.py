"""Joystick gait tracking for Atalante."""

from typing import Any, Dict, Optional, Union

import jax
from scipy.special import factorial as cpu_factorial
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco import mjx

from moplayground.envs.locomotion.generic.bipedal import BipedalBase

from moplayground.utils import geometry as geo
from moplayground.envs.locomotion.control.gait import GaitLibrary, MIN_SWING_PHASE

class NaviGait(BipedalBase):
    """
    Implements the NaviGait hybrid architecture for bipedal locomotion. Combines
    gait libraries with traditional reinfrocement learning for enhanced 
    locomotion strategies. For more details, read NaviGait, Janwani et. al.
    (2026).

    This class should be generic to any bipedal robot with a suitable gait
    library.
    """

    def __init__(
        self,
        xml_path,
        gaitlib_path: str,
        num_states: int,
        env_params: config_dict.ConfigDict,
        gait_type='P2',
        backend='jnp',
        num_degree=7,
        animate=False,
        track_yaw=False
    ):
        # Initialize the parent (bipedal) class
        super().__init__(
            xml_path         = xml_path,
            env_params       = env_params,
            backend          = backend,
        )
        self.animate = animate
        self.num_states = num_states
        self.num_degrees = num_degree
        self.track_yaw = track_yaw
        factorial = None
        if backend == 'jnp':
            factorial = jax.scipy.special.factorial
        elif backend == 'np':
            factorial = cpu_factorial

        self.gaitlib = GaitLibrary.from_directory(
            path         = gaitlib_path,
            v0           = self._np.array([0.0, 0.0]),
            num_states   = self.num_states,
            num_degree   = num_degree,
            gnp          = self._np,
            fact         = factorial,
            gait_type    = gait_type,
        )

        # Set the configuration
        self.params = env_params
        self._set_model_vars()

    def _set_model_vars(self) -> None:
        
        # compute the initial positions and velocities of gait
        self._lowers = self._mj_model.jnt_range[1:, 0]
        self._uppers = self._mj_model.jnt_range[1:, 1]
        self._jt_scale = self._np.array(self.params.jt_scale)
        self._vel_scale = self.params.vel_scale

    def set_initial_data(
        self,
        default_pose: jax.Array
    ):
        init_q = default_pose.copy()
        init_qd = self._np.zeros(self._mj_model.nv)
        init_ctrl = self._np.zeros(self._mj_model.nu)
        init_xfrc = self._np.zeros((self._mj_model.nbody, 6))

        # initialize data
        data = self._data_init_fn(0.0, init_q, init_qd, init_ctrl, init_xfrc)

        return data
    
    def initialization_randomization(
        self, rng, default_qpos, ndof
    ):
        rng, key = self._split(rng)
        z_bounds = self.get_bounds(self.params.domain_randomization.params.z0)
        rand_z = default_qpos[2] * self._uniform(
            key    = key,
            minval = z_bounds[0],
            maxval = z_bounds[1]
        )
        default_qpos = self._set_val_fn(
            arr     = default_qpos,
            val     = rand_z,
            min_idx = 2,
            max_idx = 3
        )
        if self.params.initialization.strategy == 'random-with-zero':
            rng, vx_rng, vy_rng = self._split(rng, 3)
            lim_vx = self._np.array(self.params.command.lin_vel_x)
            lim_vy = self._np.array(self.params.command.lin_vel_y)
            chosen_vdes = self._np.array([
                self._uniform(vx_rng, minval=lim_vx[0], maxval=lim_vx[1]),
                self._uniform(vy_rng, minval=lim_vy[0], maxval=lim_vy[1])
            ])
            rng, zero_key = self._split(rng)
            chosen_vdes = self._np.where(
                self._uniform(zero_key) > 1 - self.params.initialization.chance_default,
                self._np.zeros(2),
                chosen_vdes
            )
            chosen_vdes = self._np.hstack((chosen_vdes, self._np.array([0.0])))
        else:
            chosen_vdes = self._np.array(self.params.initialization.vdes)
        
        
        gaitlib = self.set_gaitlib(
            rng          = rng,
            vdes_gaitlib = chosen_vdes[:2]
        )
        ff_state = gaitlib.ff_evaluate(0)
        jt_state = gaitlib(0)
        des_qpos = self._np.hstack([ff_state[:self.qpos_free], jt_state[:ndof]])
        vdes_is_zero = self._np.all(chosen_vdes[:2] == self._np.zeros(2))
        qpos = self._np.where(
            vdes_is_zero,
            default_qpos.copy(),
            des_qpos
        )
        qvel = self._np.where(
            vdes_is_zero,
            self._np.zeros(self.nv),
            self._np.hstack([ff_state[self.qpos_free:], jt_state[ndof:]])
        )

        
        if self.params.initialization.strategy == 'manual':
            qpos = des_qpos
            qvel = self._np.hstack([ff_state[self.qpos_free:], jt_state[ndof:]])


        return rng, gaitlib, qpos, qvel

    
    def set_gaitlib(
        self,
        rng: jax.Array,
        vdes_gaitlib: jax.Array,
    ):
        rng, leg_rng = self._split(rng)
        
        swing_leg = self._np.array(1.0) # right
        if self.params.start_stance == 'either':
            swing_leg = self._np.round(self._uniform(leg_rng, minval=0, maxval=1))
        elif self.params.start_stance == 'left':
            swing_leg = self._np.array(0.0)
            
        gaitlib = self.gaitlib.reset_gait(
            vdes      = vdes_gaitlib,
            t         = -0.01,
            swing_leg = swing_leg.astype(self._np.int32)
        )
        return gaitlib
            
        
    def reset(
        self,
        rng: jax.Array,
        gaitlib: GaitLibrary,
        global_qpos: jax.Array,
        global_qvel: jax.Array,
        ctrl: jax.Array,
        ndof: int,
        torso_id: int,
        floor_id: int,
        num_resets: int,
        njoint = None,
    ) -> mjx_env.State:        
        data = self._data_init_fn(
            time         = 0.0,
            qpos         = global_qpos,
            qvel         = global_qvel,
            ctrl         = ctrl,
            xfrc_applied = self._np.zeros((self._mj_model.nbody, 6)),
        )
        state = super().reset(rng, data, torso_id, floor_id, ndof, num_resets, njoint = njoint)
        # Calculate offset transform
        relative_base_des = gaitlib.ff_evaluate(0)
        base2global = geo.solve_transform(
            self._np, 
            relative_base_des[:self.qpos_free], 
            global_qpos[:self.qpos_free],
            reset_yaw=True
        )
        # Initialize history buffers.
        his_len = self.params.history_length
        gait_des = gaitlib(0)
        base_des = gaitlib.ff_evaluate(0)
        gait_history = self._np.repeat(gait_des[self._np.newaxis], his_len, axis=0)
        base_history = self._np.repeat(base_des[self._np.newaxis], his_len, axis=0)

        # Setup joint calibration offsets
        rng, key = self._split(rng)
        rand_cal = self.params.initialization.random_jt_calibration
        jt_offset = self._np.zeros(ndof)
        ankle_offset = self._np.array(self.params.ankle_offsets)
        if rand_cal.enabled:
            ankle_offset = ankle_offset + self._uniform(
                key,
                shape=(2),
                minval=rand_cal.minval,
                maxval=rand_cal.maxval
            )
        jt_offset = self._set_val_fn(
            jt_offset,
            ankle_offset[0],
            min_idx=4,
            max_idx=5
        )
        jt_offset = self._set_val_fn(
            jt_offset,
            ankle_offset[1],
            min_idx=9,
            max_idx=10
        )
        additional_info = {
            "rng":                rng,
            'standing':           0.0,
            'last_res_target':    self._np.zeros(ndof),
            'gait_history':       gait_history, 
            'base_history':       base_history,
            'delta_qdot':         self._np.zeros(ndof),
            'jtdot_desired':      self._np.zeros(ndof),
            'base2global':        base2global,
            'old_base2global':    base2global,
            'gaitlib':            gaitlib,
            'vdes':               self._np.hstack((gaitlib.curr_vdes.copy(), self._np.zeros(1))),
            'vdes_res':           self._np.zeros(3),
            'last_vdes_res':      self._np.zeros(3),
            'vel_target':         self._np.zeros(3),
            'res_target':         self._np.zeros(ndof),
            'last_time':          data.time - self.params.ctrl_dt,
            'jt_offset':          jt_offset
        }
        info = state.info | additional_info

        metrics = {}
        for k in self.params.reward.weights.keys():
            metrics[f"reward/{k}"] = self._np.zeros(())

        info['contact'] = self._np.array([False, False])
        obs = None
        reward, done = self._np.zeros(2)
        state = self._state_init_fn(data, obs, reward, done, metrics, info)
        return state
    
    def update_info(
        self,
        parent_state,
        global_hzd_qpos,
        global_hzd_qvel,
        noisy_gyro,
        noisy_accel,
        qpos_history_len,
        gyro_history_len,
        accel_history_len,
    ):

        base2global = parent_state.info['base2global']
        hzd_relative_qpos = self.get_relative_qpos(global_hzd_qpos, base2global)
        qpos_history = self.make_history(hzd_relative_qpos, qpos_history_len)
        noisy_qpos_history = qpos_history.copy()

        # Don't transform qvel for now...meshing with rpy and quat
        qvel_history = self.make_history(global_hzd_qvel, qpos_history_len)
        noisy_qvel_history = qvel_history.copy()
        
        gyro_history  = self.make_history(noisy_gyro, gyro_history_len)
        accel_history = self.make_history(noisy_accel, accel_history_len)
        
        motor_targets = self._np.hstack([
            global_hzd_qpos[self.qpos_free:],
            global_hzd_qvel[self.qvel_free:]
        ])
        motor_target_history = self.make_history(motor_targets, self.params.domain_randomization.action_delay.val[1] + 1)
        additional_info = {
            'gyro_history'        : gyro_history,
            'accel_history'       : accel_history,
            'qpos_history'        : qpos_history,
            'noisy_qpos_history'  : noisy_qpos_history,
            'qvel_history'        : qvel_history,
            'noisy_qvel_history'  : noisy_qvel_history,
            'motor_target_history': motor_target_history,
        }
        additional_info['transform_init'] = self._np.copy(parent_state.info['base2global'])
        updated_info = parent_state.info | additional_info

        return updated_info
    
    def check_reset(
        self, state: mjx_env.State
    ) -> mjx_env.State:

        def env_reset(rng):
            s = self.reset(rng, num_resets=state.info['num_resets'])
            info = state.info | s.info
            info['num_resets'] = state.info['num_resets']
            metrics = state.metrics | s.metrics
            return self._state_init_fn(
                data=s.data,
                obs=s.obs,
                reward=s.reward,
                done=s.done,
                metrics=metrics,
                info=info
            )
        
        new_state = self._cond(
            state.data.time < state.info['gaitlib'].t0,
            lambda _: env_reset(state.info['rng']),
            lambda _: state,
            None
        )

        return new_state
    
    def animate_data(self, state, data, jtpos_des):
        base_qpos = geo.apply_transform(
            self._np,
            state.info['base_history'][0][:self.qpos_free],
            state.info['base2global']
        )
        data.qpos = self._np.hstack([base_qpos, jtpos_des])
        data.qvel = self._np.zeros(self._mj_model.nv)
        data.qvel[:self.qvel_free] = state.info['base_history'][0][self.qpos_free:]

        return data
        
        
    def pre_process_action(
        self, 
        time: jax.Array,
        info: dict[str | jax.Array],
        res_joints: jax.Array,
        res_vel: jax.Array,
        recurrent_states: jax.Array,
        ndof: int,
    ) -> jax.Array:
        """Pre-processes the residual actions and re-evaluates the gait library
        to compute the desired joint targets."""

        # Run the residual joint commands through lowpass filter
        dt = time - info['last_time']
        scaled_res_action = self._jt_scale * res_joints
        info['res_target'] = info['res_target'] + dt / self.params.filter_tau * (scaled_res_action - info['res_target'])


        # Calculate a running derivative (Euler velocity) of delta q
        scaled_last_action = info['last_res_target']
        info['delta_qdot'] = info['delta_qdot'] + dt / self.params.filter_tau * ((info['res_target'] - scaled_last_action) / dt - info['delta_qdot'])
        info['last_res_target'] = info['res_target'].copy()

        # Run the residual velocity command through a lowpass filter
        input_res_vel = self._np.hstack((self._vel_scale * res_vel, self._np.zeros(1)))
        info['last_vdes_res'] = info['vdes_res'].copy()
        info['vdes_res'] = info['vdes_res'] + dt / self.params.filter_tau * (input_res_vel - info['vdes_res'])
        vel_cmd_limits = self._np.vstack((
            self.params.command.lin_vel_x,
            self.params.command.lin_vel_y,
            self.params.command.ang_vel_z
        )).T
        info['vel_target'] = self._np.clip(info['vdes'] + info['vdes_res'], *vel_cmd_limits)
        # print(round(self._np.linalg.norm(info['vel_target']), 2))
        # print(self._np.round(info['vdes_res'], 2))

        # Pull the gait corresponding to the desired velocity + residual
        # velocity from the gait library
        def set_new_gait(args: tuple) -> GaitLibrary:
            gaitlib, vdes_gaitlib, t = args
            return gaitlib.set_gait(vdes_gaitlib, t)

        def keep_gait(args: tuple) -> GaitLibrary:
            gaitlib, _, _ = args
            return gaitlib
        gaitlib = info['gaitlib'] 

        gaitlib = self._cond(
            gaitlib.get_step_phase(time) < MIN_SWING_PHASE,
            set_new_gait,
            keep_gait,
            (gaitlib, info['vel_target'][:2], time)
        )
        info['gaitlib'] = gaitlib
        
        # Calculate the desired joint positions using the gait library
        new_state_des = gaitlib(gaitlib.get_phase(time))
        jt_desired = new_state_des[:ndof]

        # Combine the delta_qdot with qdot desired from the gaitlib to get
        # desired qvel
        jtdot_desired = new_state_des[ndof:] + info['delta_qdot']
        
        # Compute the final motor targets by adding the residual to desired
        # q_targets = jt_desired + info['res_target'] + info['jt_offset']
        q_targets = info['res_target']

        if self.params.tracking == 'position':
            vel_enabled = self._np.zeros(1)
        elif self.params.tracking == 'position-velocity':
            vel_enabled = self._np.ones(1)
        else:
            raise Exception(f'Tracking mode {self.params.tracking} DNE')

        motor_targets = self._np.hstack([q_targets, vel_enabled * jtdot_desired])
        info['last_time'] = time
    
        info['act_history'] = self.update_history(
            info['act_history'],
            self._np.hstack([res_joints, res_vel, recurrent_states])
        )

        info['motor_target_history'] = self.update_history(
            info['motor_target_history'],
            motor_targets
        )
        return motor_targets, info
    
    def get_relative_qpos(
        self,
        qpos,
        base2global
    ):
        relative_qpos = self._np.hstack([
            geo.apply_transform(self._np, qpos[:self.qpos_free], geo.inv_transform(self._np, base2global)),
            qpos[self.qpos_free:],
        ])
        return relative_qpos
    
    def update_internal_state(
        self,
        time: jax.Array,
        qpos: jax.Array,
        qvel: jax.Array,
        info: dict[str, jax.Array],
        res_joints: jax.Array,
        left_ground_contact: bool,
        right_ground_contact: bool,
        gyro: jax.Array,
        accel: jax.Array,
        ndof: int,
        qpos_noise: float,
        qvel_noise: float,
        attitude_noise: float
    ) -> dict[str | jax.Array]:
        """Updates the internal state of the environment based on the new data.
        Necessary to compute the observations for next policy inference"""

        gaitlib: GaitLibrary = info['gaitlib']
        step_phase = gaitlib.get_step_phase(time)
        phase = gaitlib.get_phase(time)
        
        # check for a step
        swing_contact = self._cond(
            gaitlib.swing_leg == 0,
            lambda _: left_ground_contact,
            lambda _: right_ground_contact,
            None
        )
        ground_contact = (swing_contact & (step_phase > MIN_SWING_PHASE)) \
                         | (phase >= 1.0)
        # ground_contact = phase >= 1.0
        
        # Switch stance and remake gait object if necessary
        switch_stance = lambda x: x.impact_reset(time, self._cond)
        keep_stance = lambda x: x

        new_gaitlib = self._cond(
            ground_contact,
            switch_stance,
            keep_stance,
            gaitlib
        )
        
        # Update the state info with the new gait object and the desired gait targets
        gait_phase = new_gaitlib.get_phase(time)
        # base_pos = state.info['base_des'][:self.qpos_free] + state.info['base_offset']
        new_gait_des = new_gaitlib(gait_phase)
        new_base_des = new_gaitlib.ff_evaluate(gait_phase)
        switched = self._np.astype((ground_contact), self._np.int32)
        heading = geo.extract_yaw(self._np, qpos[3:7])
        euler_yaw = geo.quat2euler(self._np, heading)[2]
        # quit()

        base2global = geo.solve_transform(
            self._np,
            new_base_des[:self.qpos_free],
            qpos[:self.qpos_free],
            cmd_yaw_offset=-1.6 * qpos[1] if self.track_yaw else None #info['vdes'][2]
        )
        info['old_base2global'] = info['base2global'].copy()
        info['base2global'] = switched * base2global + (1 - switched) * info['base2global']
        info['gaitlib'] = new_gaitlib
        
        # Update history.
        relative_qpos = self.get_relative_qpos(qpos, info['base2global'])
        info['rng'], noisy_jt = self.noisy(
            rng        = info['rng'],
            value      = relative_qpos[self.qpos_free:],
            lim        = qpos_noise,
            curr_level = self.get_curriculum_level(info)
        )
        noisy_jt = noisy_jt + info['jt_offset']
        info['rng'], noisy_omega = self.noisy(
            rng        = info['rng'],
            value      = self._np.zeros(3),
            lim        = attitude_noise,
            curr_level = self.get_curriculum_level(info)
        )
        noisy_quat_additional = geo.euler2quat(self._np, noisy_omega)
        noisy_quat = geo.quat_mul(self._np, noisy_quat_additional, relative_qpos[3:self.qpos_free])
        noisy_qpos = self._np.hstack([
            relative_qpos[:3],
            noisy_quat,
            noisy_jt
        ])
        info['rng'], noisy_qvel = self.noisy(
            rng        = info['rng'],
            value      = qvel,
            lim        = qvel_noise,
            curr_level = self.get_curriculum_level(info)
        )
        qpos_history       = self.update_history(info['qpos_history'], relative_qpos)
        noisy_qpos_history = self.update_history(info['noisy_qpos_history'], noisy_qpos)
        qvel_history       = self.update_history(info['qvel_history'], qvel)
        noisy_qvel_history = self.update_history(info['noisy_qvel_history'], noisy_qvel)
        base_history       = self.update_history(info['base_history'], new_base_des)
        gait_history       = self.update_history(info['gait_history'], new_gait_des)
        gyro_history       = self.update_history(info['gyro_history'], gyro)
        accel_history      = self.update_history(info['accel_history'], accel)
        
        info['qpos_history']       = qpos_history
        info['noisy_qpos_history'] = noisy_qpos_history
        info['qvel_history']       = qvel_history
        info['noisy_qvel_history'] = noisy_qvel_history
        info['gait_history']       = gait_history
        info['base_history']       = base_history
        info['gyro_history']       = gyro_history
        info['accel_history']      = accel_history

        return info

    def get_stand_termination(self, info, max_stand_time):
        enabled = self._np.array(self.params.termination.enabled)
        standing = info['standing'] > max_stand_time
        return enabled & standing
        
    
    def _get_obs(
        self,
        info: dict[str, Any],
        time,
        ndof
    ) -> jax.Array:
        
        # info['rng'], key = self._split(info['rng'])
        # obs_delay = self.params.domain_randomization.obs_delay
        # rand_delays = self._np.round(self._uniform(
        #     key,
        #     shape=(3,),
        #     minval=self._np.array([obs_delay.gyro[0], obs_delay.accel[0], obs_delay.qpos[0]]),
        #     maxval=self._np.array([obs_delay.gyro[1], obs_delay.accel[1], obs_delay.qpos[1]])
        # )).astype(self._np.int32)

        # gyro_delay, accel_delay, qpos_delay = self._np.where(
        #     self.params.domain_randomization.obs_delay.enabled,
        #     rand_delays,
        #     self._np.ones(3).astype(self._np.int32)
        # )
        gyro_delay, accel_delay, qpos_delay = info['obs_delay'], info['obs_delay'], info['obs_delay']
        
        his_len = self.params.history_length
        proprioception = self._np.hstack([
            self._splice(info['gyro_history'], (gyro_delay, 0), (his_len, 3)).flatten(),
            self._splice(info['accel_history'], (accel_delay, 0), (his_len, 3)).flatten(),
        ])
        references = self._np.hstack([
            info['gait_history'][0][ndof:],
            info['base_history'][0][self.qpos_free:],
            info['gaitlib'].swing_leg,
            info['gaitlib'].get_phase(time)
        ])

        output_feedback = self._np.hstack([
            info['act_history'].flatten() # includes recurrent states...
        ])

        command = info['vdes'][:2] # only needs xy for observation...

        history = self._np.hstack([
            self._splice(info['noisy_qpos_history'], (qpos_delay, 3), (his_len, self.nq - 3)).flatten(),
            info['gait_history'][:his_len, :ndof].flatten(),
            info['base_history'][:his_len, :self.qpos_free].flatten(),
        ])

        privileged_history = self._np.hstack([
            info['qpos_history'][:his_len].flatten(),
        ])

        disturbance = info['curr_pert']

        obs = self._np.hstack([
            proprioception,
            references,
            output_feedback,
            command,
            history
        ])

        privileged_obs = self._np.hstack([
            obs,
            privileged_history,
            disturbance,
            info['jt_offset']
        ])

        return {
            'state': obs,
            'privileged_state': privileged_obs
        }
    
    def reward_residual_vel_rate(
        self,
        vdes_res: jax.Array,
        last_vdes_res: jax.Array,
        vdes_res_rate_sigma: jax.Array
    ) -> jax.Array:
        # Penalize the size of the residual.
        err = self._np.sum(self._np.square(vdes_res - last_vdes_res))
        return self.exp_reward(err, vdes_res_rate_sigma)