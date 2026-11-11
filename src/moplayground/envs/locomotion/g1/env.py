"""Multi-objective Unitree G1 humanoid environment.

Matches the Go2 environment (``moplayground.envs.locomotion.go2.env``),
except where the body geometry differs. It reads all model constants and
helpers from ``moplayground.envs.locomotion.g1.interface``.

The G1 walks on flat ground under position control: each action is an
offset from the default joint pose, scaled by ``action_scale``, in the
actuator order of the G1 model.

The policy only sees signals the real robot can measure. Each control step,
gyro, accelerometer, joint angles (relative to the default pose), joint
velocities and body-frame gravity are read with zero-mean uniform noise and
stacked with the last action into one frame (see ``_make_frame``). The policy
observation is the last ``history_length`` frames, newest first. The
privileged observation used by the critic appends clean, sim-only quantities
and the per-foot gait state to it (see ``_get_obs``).

Noise half-widths are the ``*_NOISE`` constants in
``moplayground.envs.locomotion.g1.interface``, scaled by ``noise_scale`` in
``env_config``. Reward terms are named in ``reward_function``; the YAML config
groups them into objectives.
"""

# TODO list (remove items as they are done):
#   [ ] Termination: a humanoid falls well before it flips over. The base
#       height check (termination.base_height = 0.3 m in config/g1.yaml) is
#       a first guess; tune it and consider a tilt limit.
#   [ ] Objectives: choose the tradeoffs for the Pareto front (e.g. vx vs vy,
#       speed vs energy, speed vs yaw rate, speed vs foot clearance). Set
#       `reward.optimization.objectives` / `shared_objectives` in the YAML.
#   [ ] Privileged value obs: training never reads 'privileged_state' yet
#       (TODO in learning/training.py).
#   [ ] Reset: randomize start pose / domain randomization (later).
#   [ ] minimal_mjx bug: SwappableBase computes joint limits as
#       `jnt_range[num_free:]`, which drops joints when num_free=7. Do not
#       use `add_random_joint_state` until fixed.
#   [ ] Foot contact: touch sensors exist (FEET_CONTACT in g1/interface.py).
#       Add get_raw_contacts / CONTACT_THRESHOLD when a reward or obs needs it.

from typing import Any

import jax
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from moplayground.envs.generic.mobase import MultiObjectiveBase
from moplayground.envs.locomotion.g1 import interface as g1


class MOG1(MultiObjectiveBase):
    """Multi-objective Unitree G1 environment.

    Emits one reward per objective. Which reward terms make up each objective
    is set by ``reward.optimization.objectives`` and ``shared_objectives`` in
    the YAML config; the available terms are the keys of
    ``reward_function``.

    Args:
        env_params: ``env_config`` from the YAML. Reads ``ctrl_dt``,
            ``sim_dt``, ``action_scale``, ``history_length``, ``noise_scale``,
            ``swing_retraction``, ``min_air_time``, ``max_stance_time`` and
            ``reward``.
        backend: ``'jnp'`` for JAX (training) or ``'np'`` for NumPy (eval).
    """

    def __init__(
        self,
        env_params        : config_dict.ConfigDict,
        backend           : str,
    ):
        super().__init__(
            xml_path          = g1.OFFICIAL_XML,
            env_params        = env_params,
            backend           = backend,
            num_free          = 7,
        )
        # Default joint pose. The G1 actuator order equals the qpos order, so it
        # is valid for both.
        self._default_jt = self._np.array(g1.DEFAULT_JT)
        standing = mujoco.MjData(self.mj_model)
        standing.qpos[:] = np.hstack([g1.DEFAULT_FF, g1.DEFAULT_JT])
        mujoco.mj_forward(self.mj_model, standing)
        self._ideal_foot_dist = self._np.array(
            g1.get_foot_dist(np, self.mj_model, standing) - self.params.swing_retraction
        )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Start an episode standing still in the default pose.

        The observation history is filled with copies of the first frame,
        whose last action is zero.

        Args:
            rng: JAX random key.

        Returns:
            The initial ``mjx_env.State``.
        """
        # TODO: randomize start pose (see module TODO on add_random_joint_state).
        rng, sensor_key = self._split(rng)

        qpos = self._np.hstack([
            g1.DEFAULT_FF,
            g1.DEFAULT_JT
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
        sensor_key, sensors = self.get_sensor_values(sensor_key, data)
        frame = self._make_frame(sensors, self._np.zeros(self.action_size))

        n_feet = len(g1.FEET)
        done = self._np.array(0.0)
        info = {
            'last_act':         self._np.zeros(self.action_size),
            'second_last_act':  self._np.zeros(self.action_size),
            'vx':               g1.get_base_global_linvel(self.mj_model, data)[0],
            'rng': sensor_key,
            'obs_history': self._np.tile(frame, (self.params.history_length, 1)),
            'feet_air_time':  self._np.zeros(n_feet),
            'stance_time':    self._np.zeros(n_feet),
            'swing_min_dist': g1.get_foot_dist(self._np, self.mj_model, data),
            'last_contact':   self._np.zeros(n_feet, dtype=bool),
            'touchdown':      self._np.zeros(n_feet, dtype=bool),
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
        # TODO: noise curriculum (see the implementation in generic/bipedal.py).
        rng, temp = self._split(rng)
        noise = self._uniform(
            key    = temp,
            shape  = value.shape,
            minval = -lim,
            maxval = lim,
        )
        return rng, value + noise

    def get_sensor_values(self, rng, data):
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
            ``joint_angles`` (one per joint, offset from the default pose,
            qpos order), ``joint_vel`` (one per joint) and ``gravity``
            (3, body frame).
        """
        scale = self.params.noise_scale
        rng, noisy_gyro = self.noisy(
            rng,
            value = g1.get_gyro(self._mj_model, data),
            lim   = scale * g1.GYRO_NOISE,
        )
        rng, noisy_accel = self.noisy(
            rng,
            value = g1.get_accelerometer(self._mj_model, data),
            lim   = scale * g1.ACCEL_NOISE,
        )
        rng, noisy_joint_angles = self.noisy(
            rng,
            value = data.qpos[7:] - self._default_jt,
            lim   = scale * g1.JOINT_ANGLE_NOISE,
        )
        rng, noisy_joint_vel = self.noisy(
            rng,
            value = data.qvel[6:],
            lim   = scale * g1.JOINT_VEL_NOISE,
        )
        rng, noisy_gravity = self.noisy(
            rng,
            value = g1.get_gravity(self._np, data),
            lim   = scale * g1.GRAVITY_NOISE,
        )
        # TODO: add foot contacts if the policy obs needs them, e.g.
        #   raw_contacts = g1.get_raw_contacts(
        #       self._np, self._mj_model, data, threshold=g1.CONTACT_THRESHOLD)

        return rng, {
            "gyro":         noisy_gyro,
            "accel":        noisy_accel,
            "joint_angles": noisy_joint_angles,
            "joint_vel":    noisy_joint_vel,
            "gravity":      noisy_gravity,
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
        """Apply one action, step the physics and compute rewards.

        Args:
            state: Current state.
            action: Joint-position offsets from the default pose, in the
                G1 actuator order, scaled by ``action_scale``.

        Returns:
            The next ``mjx_env.State``. ``done`` is 1.0 if
            ``fall_termination`` fired.
        """
        # Position control (same as Go1 in Playground): action is an offset from
        # the default pose. Action order = actuator order of the G1 model.
        motor_targets = self._default_jt + self.params.action_scale * action

        data = self._step_fn(state.data, motor_targets)

        state.info["rng"], sensors = self.get_sensor_values(state.info["rng"], data)
        state.info['vx'] = g1.get_base_global_linvel(self.mj_model, data)[0]

        contact = g1.get_feet_contact(self._np, data)
        foot_dist = g1.get_foot_dist(self._np, self.mj_model, data)
        state.info['touchdown'] = (state.info['feet_air_time'] > 0) & contact
        state.info['feet_air_time'] += self.params.ctrl_dt
        state.info['stance_time'] += self.params.ctrl_dt
        state.info['swing_min_dist'] = self._np.minimum(state.info['swing_min_dist'], foot_dist)

        done, blowup = self.fall_termination(data)

        rewards = self.reward_function(
            data   = data,
            action = action,
            info   = state.info,
            done   = done
        )

        reward, metrics = self.get_reward_and_metrics(rewards, state.metrics)
        reward = self._np.where(blowup, self._np.zeros_like(reward), reward)

        state.info['feet_air_time'] *= ~contact
        state.info['stance_time'] *= contact
        state.info['swing_min_dist'] = self._np.where(contact, foot_dist, state.info['swing_min_dist'])
        state.info['last_contact'] = contact

        state.info['second_last_act'], state.info['last_act'] = state.info['last_act'], action
        frame = self._make_frame(sensors, state.info['last_act'])
        new_history = self._np.vstack([frame[None], state.info['obs_history'][:-1]])
        state.info['obs_history'] = self._np.where(blowup, state.info['obs_history'], new_history)
        obs = self._get_obs(data, state.info)
        obs = {k: self._np.where(blowup, state.obs[k], v) for k, v in obs.items()}
        done = done.astype(float)
        return self._state_init_fn(data, obs, reward, done, metrics, state.info)

    def fall_termination(self, data):
        """Return whether the episode should end.

        Ends when the base is tilted ``MAX_TILT_ANGLE`` (60 degrees) or
        more from upright, the sim has blown up (``qpos`` or ``qvel`` is
        non-finite), or, if ``termination.enabled``, the base is below
        ``termination.base_height``.

        Args:
            data: MuJoCo data after the physics step.

        Returns:
            Tuple ``(done, blowup)`` of boolean scalars. ``done`` is True
            for any termination. ``blowup`` is True only for a non-finite
            state.
        """
        tilted = g1.get_tilt_angle(self._np, self.mj_model, data) >= g1.MAX_TILT_ANGLE
        blowup = (
            ~self._np.isfinite(data.qpos).all()
            | ~self._np.isfinite(data.qvel).all()
        )
        done = tilted | blowup

        term = self.params.termination
        if term.enabled:
            too_low = data.qpos[2] < term.base_height
            done = done | too_low

        return done, blowup

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
        """Build the policy and privileged observations.

        Args:
            data: MuJoCo data after the physics step.
            info: State info; reads ``obs_history``, ``feet_air_time``,
                ``stance_time``, ``swing_min_dist`` and ``last_contact``.

        Returns:
            Dict with ``state``, the flattened noisy observation history,
            and ``privileged_state``, which appends clean gyro (3),
            body-frame base linear velocity (3), body-frame foot velocities
            (2 x 3), body-frame base-minus-foot positions (2 x 3), and the
            gait state: air time (2), stance time (2), swing minimum
            base-to-foot distance (2) and last contact (2). Feet in order
            left, right. The base is the pelvis imu site.
        """
        state = info['obs_history'].flatten()

        base_gyro = g1.get_gyro(self.mj_model, data)
        base_local_linvel = g1.get_base_local_linvel(self.mj_model, data)

        foot_vel = g1.get_foot_vel(self._np, self.mj_model, data)
        base_pos = g1.get_base_global_pos(self.mj_model, data)
        foot_pos = g1.get_foot_pos(self._np, self.mj_model, data)
        # Row vectors times R rotate world-frame vectors into the body frame.
        R = data.site_xmat[g1.IMU_SITE_ID].reshape(3, 3)
        local_foot_pos = ((base_pos - foot_pos) @ R).flatten()
        local_foot_vel = (foot_vel @ R).flatten()

        privileged_state = self._np.hstack([
            state,
            base_gyro,
            base_local_linvel,
            local_foot_vel,
            local_foot_pos,
            info['feet_air_time'],
            info['stance_time'],
            info['swing_min_dist'],
            info['last_contact'].astype(float),
        ])

        return {
            "state": state,
            "privileged_state": privileged_state,
        }

    @property
    def action_size(self):
        """Number of actuators in the G1 model."""
        return self.mj_model.nu

    def reward_function(self, data, action, info, done):
        """Compute every reward term for one step.

        Terms are unweighted; ``reward.weights`` in the YAML sets their
        weight and sign (penalties such as ``vel_rate`` are returned
        positive and given negative weights).

        Args:
            data: MuJoCo data after the physics step.
            action: The action applied this step.
            info: State info, read before ``last_act`` is updated.
            done: Whether the episode terminated this step.

        Returns:
            Dict mapping each term name to a scalar.
        """
        return {
            'vel_rate':   self.reward_vel_smoothness(action, info),
            'accel_rate': self.reward_accel_smoothness(action, info),
            'alive':      self.reward_alive(),
            'run':        self.reward_run(info),
            'done':       self.reward_done(done),
            'energy':     self.reward_energy(data, action, info),
            'softness':   self.reward_softness(data, action, info),
            'heading':    self.reward_heading(data, action, info),
            'lin_vel_z':  self.cost_lin_vel_z(data),
            'ang_vel_xy': self.cost_ang_vel_xy(data),
            'orientation': self.cost_orientation(data),
            'feet_slip':  self.cost_feet_slip(data),
            'torque':     self.cost_torque(data),
            'feet_clearance': self.cost_feet_clearance(data),
            'feet_height':    self.cost_feet_height(info),
            'feet_air_time':  self.reward_feet_air_time(info),
            'stance_time':    self.cost_stance_time(info),
        }

    def cost_stance_time(self, info):
        """Stance time beyond ``max_stance_time``, summed over feet.

        Per foot: ``max(0, stance_time - max_stance_time)``. Grows while a
        foot stays planted, so it penalizes a robot that stops stepping.
        """
        excess = info['stance_time'] - self.params.max_stance_time
        return self._np.sum(self._np.maximum(excess, 0.0))

    def cost_feet_slip(self, data):
        """Squared x/y foot speed, summed over feet in floor contact."""
        contact = g1.get_feet_contact(self._np, data)
        vel_xy = g1.get_foot_vel(self._np, self.mj_model, data)[:, :2]
        return self._np.sum(self._np.sum(vel_xy**2, axis=-1) * contact)

    def cost_torque(self, data):
        """L2 norm of the actuator torques, ``sqrt(sum(tau^2))``.
        """
        return self._np.sqrt(self._np.sum(data.actuator_force**2))

    def cost_feet_clearance(self, data):
        """Swing distance error, gated by horizontal foot speed, summed over feet.

        Per foot: ``|dist - ideal| * sqrt(||v_xy||)``. ``dist`` is the
        pelvis-to-foot distance; ``ideal`` is the standing distance minus
        ``swing_retraction``. ``v_xy`` is the world-frame x/y foot velocity,
        so stance feet, which do not slide, cost about 0.
        """
        dist = g1.get_foot_dist(self._np, self.mj_model, data)
        vel_xy = g1.get_foot_vel(self._np, self.mj_model, data)[:, :2]
        speed = self._np.sqrt(self._np.linalg.norm(vel_xy, axis=-1))
        return self._np.sum(self._np.abs(dist - self._ideal_foot_dist) * speed)

    def cost_feet_height(self, info):
        """Squared relative error of each swing's closest pelvis-to-foot distance, at touchdown.

        Per foot: ``(swing_min_dist / ideal - 1)^2`` on the step the foot
        lands, else 0. Summed over feet.
        """
        error = info['swing_min_dist'] / self._ideal_foot_dist - 1.0
        return self._np.sum(error**2 * info['touchdown'])

    def reward_feet_air_time(self, info):
        """``air_time - min_air_time`` per foot on the step it lands, summed over feet."""
        return self._np.sum((info['feet_air_time'] - self.params.min_air_time) * info['touchdown'])

    def cost_lin_vel_z(self, data):
        """Squared world-frame z base linear velocity."""
        return g1.get_base_global_linvel(self.mj_model, data)[2]**2

    def cost_ang_vel_xy(self, data):
        """Squared world-frame x/y base angular velocity, summed."""
        return self._np.sum(g1.get_base_global_angvel(self.mj_model, data)[:2]**2)

    def cost_orientation(self, data):
        """Squared x/y components of the torso up axis, summed (0 when upright)."""
        return self._np.sum(g1.get_torso_upvector(self.mj_model, data)[:2]**2)

    def reward_alive(self):
        """Constant 1 per step."""
        return 1.0

    def reward_vel_smoothness(self, action, info):
        """Squared first difference of actions, summed over joints."""
        return self._np.sum((action - info["last_act"])**2)

    def reward_accel_smoothness(self, action, info):
        """Squared second difference of actions, summed over joints."""
        first_diff = action - info["last_act"]
        prev_diff = info["last_act"] - info["second_last_act"]
        return self._np.sum((first_diff - prev_diff)**2)

    def reward_done(self, done):
        """1 on the terminating step, else 0."""
        return self._np.array(done)

    def reward_run(self, info):
        """Gaussian tracking of ``ideal_base_vel`` (world-frame vx)."""
        err = info['vx'] - self.params.ideal_base_vel
        return self._np.exp(-err**2 / (2 * self.params.reward.sigmas.run**2))

    def reward_energy(self, data, action, info):
        """``sum(|force * velocity|)`` over actuators."""
        return self._np.sum(self._np.abs(data.actuator_force * data.actuator_velocity))

    def reward_softness(self, data, action, info):
        """Squared deviation of the vertical ground reaction force from body weight.

        ``((F_z - m*g) / (m*g))^2``, with ``F`` from
        ``g1.get_ground_reaction_force``. The time-average of ``F_z`` is
        ``m*g`` whatever the gait, so the penalty targets the spikes from hard
        landings rather than ``F_z`` itself. 0 when the body is not
        accelerating vertically; 1 for a spike of one extra body weight.
        A penalty: returned positive, given a negative weight.
        """
        weight = self._np.sum(self.mj_model.body_mass) * 9.81
        f_z = g1.get_ground_reaction_force(self._np, self.mj_model, data)[2]
        return ((f_z - weight) / weight)**2

    def reward_heading(self, data, action, info):
        """Gaussian tracking of ``ideal_yaw_rate`` (world-frame wz)."""
        yaw_rate = g1.get_base_global_angvel(self.mj_model, data)[2]
        err = yaw_rate - self.params.ideal_yaw_rate
        return self._np.exp(-err**2 / (2 * self.params.reward.sigmas.heading**2))
