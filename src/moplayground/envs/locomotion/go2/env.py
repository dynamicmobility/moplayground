"""Multi-objective Unitree Go2 quadruped environment.

The Go2 walks on flat ground under position control: each action is an
offset from the default joint pose, scaled by ``action_scale``, in actuator
order (FR, FL, RR, RL).

The policy only sees signals the real robot can measure. Each control step,
joint angles (relative to the default pose), joint velocities, gyro,
accelerometer and body-frame gravity are read with zero-mean uniform noise and
stacked with the last action into one frame. The policy observation is the
last ``history_length`` frames, newest first. The privileged observation used
by the critic appends clean, sim-only quantities to it.

Noise half-widths are the ``*_NOISE`` constants in
``moplayground.envs.locomotion.go2.interface``, scaled by ``noise_scale`` in
``env_config``. Reward terms are named in ``reward_function``; the YAML config
groups them into objectives.
"""

# TODO list (remove items as they are done):
#   [ ] Objectives: choose the tradeoffs for the Pareto front (e.g. vx vs vy,
#       speed vs energy, speed vs yaw rate, speed vs foot clearance). Set
#       `reward.optimization.objectives` / `shared_objectives` in the YAML.
#   [ ] Privileged value obs: training never reads 'privileged_state' yet
#       (TODO in learning/training.py).
#   [ ] Termination: base height threshold, if needed (constant goes in
#       interface.py, see its TODO). See `fall_termination`.
#   [ ] Action: confirm action_scale (Go1 uses 0.5) and actuator order
#       (FR, FL, RR, RL) vs qpos order (FL, FR, RL, RR). See interface.py TODO.
#   [ ] Reset: randomize start pose / domain randomization (later).
#   [ ] minimal_mjx bug: SwappableBase computes joint limits as
#       `jnt_range[num_free:]`, which keeps only 6 of 12 joints when
#       num_free=7. Do not use `add_random_joint_state` until fixed.
#   [ ] Foot contact: choose a touch sensor (TODO in model/scene_mjx.xml),
#       then add contact helpers to interface.py.
#   [ ] Cameras: named cameras for rendering (TODO in model/scene_mjx.xml).

from typing import Any

import jax
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from moplayground.envs.generic.mobase import MultiObjectiveBase
from moplayground.envs.locomotion.go2 import interface as go2


class MOGo2(MultiObjectiveBase):
    """Multi-Objective Unitree Go2 Environment. Objectives: TODO."""

    def __init__(
        self,
        env_params        : config_dict.ConfigDict,
        backend           : str,
    ):
        super().__init__(
            xml_path          = go2.OFFICIAL_XML,
            env_params        = env_params,
            backend           = backend,
            num_free          = 7,
        )
        # Default joint pose; same for every leg, so valid in qpos and actuator order.
        self._default_jt = self._np.array(go2.DEFAULT_JT)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # TODO: randomize start pose (see module TODO on add_random_joint_state).
        rng, sensor_key = self._split(rng)
        frame = self.make_frame(sensors, self._np.zeroes(self.action_size))

        qpos = self._np.hstack([
            go2.DEFAULT_FF,
            go2.DEFAULT_JT
        ])
        qvel = self._np.zeros(self.mj_model.nv)
        ctrl = self._default_jt

        data = self._data_init_fn(
            qpos         = qpos,
            qvel         = qvel,
            ctrl         = ctrl,
            time         = 0.0,
            xfrc_applied = self._np.zeros((self._mj_model.nbody, 6)),
        )
        sensor_key, sensors = self.get_sensor_values(sensor_key, sensors)
        done = self._np.array(0.0)
        info = {
            'last_act':         self._np.zeros(self.action_size),
            'second_last_act':  self._np.zeros(self.action_size),
            'vx':               go2.get_base_local_linvel(self.mj_model, data)[0],
            'posbefore':        0.0,
            'posafter':         0.0,
            'rng': sensor_key,
            'obs_history': self._np.tile(frame, (self.params.history_length, 1))
        }

        rewards = self.reward_function(
            data   = data,
            action = self._np.zeros(self.mj_model.nu),
            info   = info,
            done   = False,
        )
        reward, metrics = self.get_reward_and_metrics(rewards, {})
        obs = self._get_obs(data, info)
        return self._state_init_fn(data, obs, reward, done, metrics, info)

    def noisy(self, rng, value: jax.Array, lim: jax.Array):
            """Add zero-mean uniform noise to a value.

            Each element gets independent noise drawn from ``[-lim, lim]``.

            Args:
                rng: JAX random key.
                value: Clean value to perturb.
                lim: Noise half-width, a scalar or an array broadcastable to
                    ``value``.

            Returns:
                Tuple ``(rng, noisy_value)``: the advanced key, to store back
                for the next draw, and ``value`` plus noise.
            """
            #TODO: implement curriculum learning (see implementation in /generic/bipedal.py)
            rng, temp = self._split(rng)
            noise = self._uniform(
                key    = temp,
                shape  = value.shape,
                minval = -lim,
                maxval = lim
            )
            noisy_value = noise + value
            return rng, noisy_value

    def get_sensor_values(
            self,
            rng,
            data
        ):
            """Read the policy's sensors and add noise to each reading.

            Only signals available on the real robot are read. Each noise
            half-width is ``noise_scale`` times the matching ``*_NOISE``
            constant in the interface.

            Args:
                rng: JAX random key.
                data: MuJoCo data after the physics step.

            Returns:
                Tuple ``(rng, sensors)``: the advanced key and a dict of noisy
                readings with keys ``gyro`` (3), ``accel`` (3),
                ``joint_angles`` (12, offset from the default pose, qpos
                order), ``joint_vel`` (12) and ``gravity`` (3, body frame).
            """
            rng, noisy_gyro = self.noisy(
                rng,
                value      = go2.get_gyro(self._mj_model, data),
                lim        = self.params.noise_scale * go2.GYRO_NOISE,
            )
            rng, noisy_accel = self.noisy(
                rng,
                value      = go2.get_accelerometer(self._mj_model, data),
                lim        = self.params.noise_scale * go2.ACCEL_NOISE,
            )
            rng, noisy_joint_angles = self.noisy(
                            rng,
                            value      = data.qpos[7:] - self._default_jt,
                            lim        = self.params.noise_scale * go2.JOINT_ANGLE_NOISE,
                        )
            rng, noisy_joint_vel = self.noisy(
                                        rng,
                                        value      = data.qvel[6:],
                                        lim        = self.params.noise_scale * go2.JOINT_VEL_NOISE,
                                    )
            rng, noisy_gravity = self.noisy(
                                                    rng,
                                                    value      = go2.get_gravity(self._np, data),
                                                    lim        = self.params.noise_scale * go2.GRAVITY_NOISE,
                                                )
    
            """raw_contacts = go2.get_raw_contacts(
                self._np,
                self._mj_model,
                data,
                threshold=go2.CONTACT_THRESHOLD
            )""" #TODO: implement raw contacts 

            return rng, {
                        "gyro": noisy_gyro, "accel": noisy_accel, "joint_angles": noisy_joint_angles,
                        "joint_vel": noisy_joint_vel, "gravity": noisy_gravity,# raw_contacts
                        }

    def _make_frame(self, sensors, last_act):
        """Concatenate one step's noisy readings and last action into a frame.

        ``reset`` and ``step`` both build frames here, so the observation
        history keeps the same layout throughout the episode.

        Args:
            sensors: Dict of noisy readings from ``get_sensor_values``.
            last_act: The action applied this step, shape ``(nu,)``.

        Returns:
            1-D array: gyro, accel, joint angles, joint velocities, gravity,
            last action.
        """
        return self._np.hstack([
            sensors["gyro"],
            sensors["accel"],
            sensors["joint_angles"],
            sensors["joint_vel"],
            sensors["gravity"],
            last_act,
        ])
    

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        
        # Position control (same as Go1 in Playground): action is an offset from
        # the default pose. Action order = actuator order (FR, FL, RR, RL).
        motor_targets = self._default_jt + self.params.action_scale * action
        state.info['posbefore'] = state.data.qpos[0] 

        data = self._step_fn(state.data, motor_targets)

        state.info["rng"], sensors = self.get_sensor_values(state.info["rng"], data)
        state.info['posafter'] = data.qpos[0]
        state.info['vx'] = go2.get_base_local_linvel(self.mj_model, data)[0]
        
        done = self.fall_termination(data)
        
        rewards = self.reward_function(
            data   = data,
            action = action,
            info   = state.info,
            done   = done
        )

        reward, metrics = self.get_reward_and_metrics(rewards, state.metrics)

        state.info['second_last_act'], state.info['last_act'] = state.info['last_act'], action
        frame = self._make_frame(sensors, state.info['last_act'])
        state.info['obs_history'] = self._np.vstack([frame[None], state.info['obs_history'][:-1]])
        obs = self._get_obs(data, state.info)
        done = done.astype(float)
        return self._state_init_fn(data, obs, reward, done, metrics, state.info)

    def fall_termination(self, data):
        # TODO: consider additional termination conditions (CoM height?)
        flipped_over = go2.get_upvector(self.mj_model, data)[-1] < 0.0
        infinite_state = ~self._np.isfinite(data.qpos).all()
        return flipped_over | infinite_state

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        # TODO: observation history?
        gyro = go2.get_gyro(self.mj_model, data) 
        joint_angles = data.qpos[7:] - self._default_jt
        joint_vel = data.qvel[6:]
        accelerometer = go2.get_accelerometer(self.mj_model, data)
        gravity = go2.get_gravity(self._np, data)

        state = self._np.hstack([
            joint_angles,
            joint_vel,
            gyro,
            gravity,
            accelerometer,
            info["last_act"],
        ])

        base_gyro = go2.get_gyro(self.mj_model, data)
        base_global_angvel = go2.get_base_global_angvel(self.mj_model, data)
        foot_vel = go2.get_foot_vel(self.mj_model, data).flatten()
        foot_pos = go2.get_foot_pos(self.mj_model, data).flatten()

        privileged_state = self._np.hstack([
            state,
            base_gyro,
            base_global_angvel,
            foot_vel,
            foot_pos
        ])
        
        return {
            "state": state, 
            "privileged_state": privileged_state,
        }

    @property
    def action_size(self):
        return self.mj_model.nu

    def reward_function(
        self,
        data,
        action,
        info,
        done
    ):

        rewards = {
           'vel_rate':      self.reward_vel_smoothness(action, info),
           'accel_rate':    self.reward_accel_smoothness(action, info),
           'speed':         self.reward_speed(info),
           'alive':         self.reward_alive(),
           'run':           self.reward_run(info),
           'done':          self.reward_done(done)
        }

        return rewards

    def reward_alive(self):
        return 1.0

    def reward_speed(self, info):
        return info['vx']
        
    def reward_vel_smoothness(self, action, info):
        vel_rate = self._np.sum((info["last_act"] - action)**2)
        return vel_rate
    
    def reward_accel_smoothness(self, action, info):
        accel_rate = self._np.sum(((action - info["last_act"]) - (info["last_act"] - info["second_last_act"]))**2)
        return accel_rate

    def reward_done(self, done):
        return self._np.array(done)

    def reward_run(self, info):
        run = (info['posafter'] - info['posbefore']) / self.dt
        return run