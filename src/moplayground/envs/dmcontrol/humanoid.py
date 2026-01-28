import jax
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from moplayground.dmcontrol.interface import HumanoidInterface
from minimal_mjx.envs.generic.base import MultiObjectiveBase

class MOHumanoid(MultiObjectiveBase):
    """Humanoid environment."""

    def __init__(
        self,
        env_params: config_dict.ConfigDict,
        backend:    str,
    ):
        super().__init__(
            xml_path          = HumanoidInterface.XML,
            env_params        = env_params,
            backend           = backend,
            num_free          = 7
        )
        self.qfrc_max = self.params.action_scale * self._np.sum(self.mj_model.actuator_gear[:, 0]) 

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._np.hstack([
            HumanoidInterface.DEFAULT_FF,
            HumanoidInterface.DEFAULT_JT
        ])
        qvel = self._np.zeros(self.mj_model.nv)
        ctrl = qpos[self.num_free:].copy()

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
        info = parent_state.info | info

        done = self._np.array(0.0)
        rewards = self.reward_function(
            data   = data,
            action = ctrl,
            info   = info,
            done   = False,
        )
        reward, metrics = self.get_reward_and_metrics(rewards, {})
        
        obs = self._get_obs(data, parent_state.info)
        return self._state_init_fn(data, obs, reward, done, metrics, info)
    
    def mass_center(self, data):
        mass = self._np.expand_dims(self._mj_model.body_mass, 1)
        xpos = data.xipos
        return (self._np.sum(mass * xpos, 0) / self._np.sum(mass))[0]

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = self._np.ones(self.action_size)
        action = self.params.action_scale * action
        state.info['posbefore'] = self.mass_center(state.data)
        data = self._step_fn(state.data, action)
        state.info['posafter'] = self.mass_center(data)
        
        done = self.fall_termination(data)
        rewards = self.reward_function(
            data   = data,
            action = action,
            info   = state.info,
            done   = done
        )
        reward, metrics = self.get_reward_and_metrics(rewards, state.metrics)
        obs = self._get_obs(
            data,
            state.info
        )
        done = done.astype(float)
        return self._state_init_fn(data, obs, reward, done, metrics, state.info)
    
    def fall_termination(self, data):
        return data.qpos[2] < 0.6

    def _get_obs(self, data: mjx.Data, info: dict) -> jax.Array:
        del info  # Unused.
        return self._np.concatenate([
            self.joint_angles(data),
            self.head_height(data).reshape(1),
            self.extremities(data).ravel(),
            self.torso_vertical_orientation(data),
            HumanoidInterface.com_vel(data),
            data.qvel,
        ])
    
    @property
    def action_size(self):
        return self._mj_model.nu

    def reward_function(
        self,
        data,
        action,
        info,
        done
    ):
        rewards = {
            'alive'     : self.reward_alive(),
            'energy'    : self.reward_energy(data, action),
            'run'       : self.reward_run(info),
            'done'      : self.reward_done(done)
        }
        return rewards
    
    def reward_alive(self):
        return 1.0
    
    def reward_run(self, info):
        return (info['posafter'] - info['posbefore']) / self.dt
    
    def reward_done(self, done):
        return self._np.array(done)
    
    def reward_energy(self, data, action):
        # norm_actuation = self._np.sum(self._np.abs(data.ctrl)) * 1 / self.params.action_scale
        # energy = (self.mj_model.nu - norm_actuation)
        # return energy
        # print(self._np.sum(self.params.action_scale * self.mj_model.actuator_gear))
        # quit()
        # actuation_lvl = self._np.sum((data.qfrc_actuator))
        # return actuation_lvl

        energy = self._np.max(
            self._np.array(
                [3.0 - self._np.sum(self._np.square(action / self.params.action_scale)),
                0.0]
            )
        )
        return energy
        

    def joint_angles(self, data: mjx.Data) -> jax.Array:
        """Returns the state without global orientation or position."""
        return data.qpos[self.num_free:]

    def torso_vertical_orientation(self, data: mjx.Data) -> jax.Array:
        """Returns the z-projection of the torso orientation matrix."""
        return data.xmat[HumanoidInterface.TORSO_ID].reshape((3,3))[2]

    def com_position(self, data: mjx.Data) -> jax.Array:
        """Returns the position of the center of mass in global coordinates."""
        return data.subtree_com[HumanoidInterface.TORSO_ID]

    def head_height(self, data: mjx.Data) -> jax.Array:
        """Returns the height of the torso."""
        return data.xpos[HumanoidInterface.HEAD_ID, -1]

    def torso_upright(self, data: mjx.Data) -> jax.Array:
        """Returns projection from z-axes of torso to the z-axes of world."""
        return data.xmat[HumanoidInterface.TORSO_ID, 2, 2]

    def extremities(self, data: mjx.Data) -> jax.Array:
        """Returns end effector positions in the egocentric frame."""
        torso_frame   = data.xmat[HumanoidInterface.TORSO_ID].reshape((3,3))
        torso_pos     = data.xpos[HumanoidInterface.TORSO_ID]
        torso_to_limb = data.xpos[HumanoidInterface.TORSO_ID] - torso_pos
        return (torso_to_limb @ torso_frame).ravel()