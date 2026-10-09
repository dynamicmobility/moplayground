"""Multi-objective Unitree Go2 environment (skeleton).

TODO list (remove items as they are done):
  [ ] Objectives: choose the tradeoffs for the Pareto front (e.g. vx vs vy,
      speed vs energy, speed vs yaw rate, speed vs foot clearance). Set
      `reward.optimization.objectives` / `shared_objectives` in the YAML.
  [ ] Rewards: implement the per-key reward terms in `reward_function`.
  [ ] Observations: policy obs = real-robot sensors only; privileged obs =
      policy obs + sim-only sensors. See `_get_obs`.
  [ ] Privileged value obs: training never reads 'privileged_state' yet
      (TODO in learning/training.py).
  [ ] Termination: thresholds for base height / flip (constants go in
      interface.py, see its TODO). See `fall_termination`.
  [ ] Timing: pick sim_dt / ctrl_dt in the YAML (Go1 in Playground uses
      sim_dt 0.004, ctrl_dt 0.02).
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
  [ ] Registration: `case 'MOGo2'` in envs/create.py, 'MOGo2' in
      config.ENVS, entry in envs/__init__.py.
  [ ] Config: config/mogo2.yaml (and config/amor/mogo2.yaml).
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
        parent_state = super().reset(
            rng            = rng,
            data           = data,
            history_length = self.params.history_length
        )
        # TODO: add any info the rewards need (e.g. last action).
        info = {}
        info = parent_state.info | info

        done = self._np.array(0.0)
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
        data = self._step_fn(state.data, motor_targets)

        done = self.fall_termination(data)
        rewards = self.reward_function(
            data   = data,
            action = action,
            info   = state.info,
            done   = done
        )
        reward, metrics = self.get_reward_and_metrics(rewards, state.metrics)
        obs = self._get_obs(data, state.info)
        done = done.astype(float)
        return self._state_init_fn(data, obs, reward, done, metrics, state.info)

    def fall_termination(self, data):
        # TODO: add base height and flip checks (thresholds go in interface.py).
        infinite_state = ~self._np.isfinite(data.qpos).all()
        return infinite_state

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        # TODO: policy obs from real-robot sensors only (joint pos/vel, gyro,
        # accelerometer/gravity, last action, history?).
        # TODO: privileged obs = policy obs + sim-only sensors (go2.get_base_*,
        # go2.get_foot_pos, go2.get_foot_vel).
        raise NotImplementedError('MOGo2._get_obs')

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
        # TODO: one entry per reward key used in the YAML weights /
        # objectives (e.g. alive, energy, vx, vy, upright, base height,
        # action rate, foot slip).
        raise NotImplementedError('MOGo2.reward_function')
