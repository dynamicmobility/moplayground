import jax
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from moplayground.envs.generic.mobase import MultiObjectiveBase
from moplayground.envs.dmcontrol.interface import HopperInterface

class MOHopper(MultiObjectiveBase):
    def __init__(
        self,
        env_params        : config_dict.ConfigDict,
        backend           : str,
    ):
        super().__init__(
            xml_path          = HopperInterface.XML,
            env_params        = env_params,
            backend           = backend,
            num_free          = 3
        )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._np.hstack([
            HopperInterface.DEFAULT_FF,
            HopperInterface.DEFAULT_JT
        ])
        qvel = self._np.zeros(self.mj_model.nv)
        ctrl = qpos[self.qpos_free:].copy()

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
        info = {}
        info['posbefore'] = 0.0
        info['posafter']  = 0.01
        info['height']    = 0.0
        info['ang']       = 0.0
        info = parent_state.info | info

        done = self._np.array(0.0).astype(self._np.float32)
        rewards = self.reward_function(
            data   = data,
            action = ctrl,
            info   = info,
            done   = False,
        )
        reward, metrics = self.get_reward_and_metrics(rewards, {})
        
        obs = self._get_obs(data, parent_state.info)
        return self._state_init_fn(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = self._np.array(self.params.action_scale) * action
        state.info['posbefore'] = state.data.qpos[0]
        data = self._step_fn(state.data, action)
        state.info['posafter'], state.info['height'], state.info['ang'] = data.qpos[0:3]
        
        done = self.fall_termination(data)
        rewards = self.reward_function(
            data   = data,
            action = action,
            info   = state.info,
            done   = done
        )
        reward, metrics = self.get_reward_and_metrics(rewards, state.metrics)
        done = done.astype(float)
        obs = self._get_obs(
            data,
            state.info
        )
        return self._state_init_fn(data, obs, reward, done, metrics, state.info)
    
    def fall_termination(
        self,  
        data: mjx.Data
    ):
        height = data.qpos[1] < 0.2
        base_angle  = self._np.abs(data.qpos[2]) > self._np.deg2rad(270)
        return height | base_angle

    def _get_obs(self, data: mjx.Data, info: dict) -> jax.Array:
        obs = self._np.concatenate([
            data.qpos[1:],
            self._np.clip(data.qvel, -10.0, 10.0),
        ])
        return {
            'state': obs,
            'privileged_state': obs
        }
    
    @property
    def action_size(self):
        return 4

    def reward_function(
        self,
        data,
        action,
        info,
        done
    ):
        rewards = {
            'alive'     : self.reward_alive(),
            'ctrl_cost' : self.reward_ctrl_cost(action),
            'run'       : self.reward_run(info),
            'jump'      : self.reward_jump(info),
            'upright'   : self.reward_upright(info),
            'done'      : self.reward_done(done)
        }
        return rewards
    
    def reward_upright(self, info):
        return self._np.clip(
            info['height'] - 0.4,
            0.2 - 0.4,
            0.0,
        )
    
    def reward_alive(self):
        return 1.0
    
    def reward_ctrl_cost(self, action):
        return self._np.square(action).sum()
    
    def reward_run(self, info):
        return (info['posafter'] - info['posbefore']) / self.dt
    
    def reward_jump(self, info):
        return self._np.clip(
            info['height'] - HopperInterface.DEFAULT_FF[1] + 0.2,
            0.0,
            self._np.inf,
        )
        
    def reward_done(self, done):
        return self._np.array(done)
    
    def reward_energy(self, action):
        return 4.0 - 1.0 * self._np.square(action).sum()