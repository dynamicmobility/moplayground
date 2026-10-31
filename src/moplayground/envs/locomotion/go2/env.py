"""Multi-objective Unitree Go2 environment (skeleton).

TODO list (remove items as they are done):
  [ ] Objectives: choose the tradeoffs for the Pareto front (e.g. vx vs vy,
      speed vs energy, speed vs yaw rate, speed vs foot clearance). Set
      `reward.optimization.objectives` / `shared_objectives` in the YAML.
  [ ] Privileged value obs: training never reads 'privileged_state' yet
      (TODO in learning/training.py).
  [ ] Termination: base height threshold, if needed (constant goes in
      interface.py, see its TODO). See `fall_termination`.
  [ ] Action: confirm action_scale (Go1 uses 0.5) and actuator order
      (FR, FL, RR, RL) vs qpos order (FL, FR, RL, RR). See interface.py TODO.
  [ ] Reset: randomize start pose / domain randomization (later).
  [ ] minimal_mjx bug: SwappableBase computes joint limits as
      `jnt_range[num_free:]`, which keeps only 6 of 12 joints when
      num_free=7. Do not use `add_random_joint_state` until fixed.
  [ ] Foot contact: choose a touch sensor (TODO in model/scene_mjx.xml),
      then add contact helpers to interface.py.
  [ ] Noise: sensor noise levels for sim-to-real (interface.py TODO).
  [ ] Cameras: named cameras for rendering (TODO in model/scene_mjx.xml).
"""

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
        rng, qpos_key, qvel_key = self._split(rng, 3)
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
        done = self._np.array(0.0)
        info = {
            'last_act':         self._np.zeros(self.action_size),
            'second_last_act':  self._np.zeros(self.action_size),
            'vx':               go2.get_base_local_linvel(self.mj_model, data)[0],
            'posbefore':        0.0,
            'posafter':         0.0,
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

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        
        # Position control (same as Go1 in Playground): action is an offset from
        # the default pose. Action order = actuator order (FR, FL, RR, RL).
        motor_targets = self._default_jt + self.params.action_scale * action
        state.info['posbefore'] = state.data.qpos[0] 

        data = self._step_fn(state.data, motor_targets)
        
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

        obs = self._get_obs(data, state.info)
        done = done.astype(float)
        return self._state_init_fn(data, obs, reward, done, metrics, state.info)

    def fall_termination(self, data):
        # TODO: consider additional termination conditions (height?)
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

        base_local_linvel = go2.get_base_local_linvel(self.mj_model, data)
        base_global_linvel = go2.get_base_global_linvel(self.mj_model, data)
        base_global_angvel = go2.get_base_global_angvel(self.mj_model, data)

        #TODO add feet sensor data
        privileged_state = self._np.hstack([
            state,
            base_local_linvel,
            base_global_linvel,
            base_global_angvel
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
        vel_rate = self._np.sum((info[last_act] - action)**2)
        return vel_rate
    
    def reward_accel_smoothness(self, action, info):
        accel_rate = self._np.sum(((action - info[last_act]) - (info[last_act] - info[second_last_act]))**2)
        return accel_rate

    def reward_done(self, done):
        return self._np.array(done)

    def reward_run(self, info):
        run = (info['posafter'] - info['posbefore']) / self.dt
        return run