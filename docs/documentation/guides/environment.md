# Writing an environment

A multi-objective environment subclasses
[`MultiObjectiveBase`][moplayground.envs.generic.mobase.MultiObjectiveBase], which
extends minimal_mjx's `SwappableBase`. The same class then runs on both backends:
`jnp` (JAX/MJX) for training and `np` (NumPy and the MuJoCo C bindings) for fast,
debuggable rollouts. This page walks through
[`MOCheetah`][moplayground.envs.dmcontrol.cheetah.MOCheetah].

!!! info "Backend handles"
    Never call `np`, `jnp`, `mjx` or `jax.random` directly inside an environment. Use
    the handles `SwappableBase` provides (`self._np`, `self._step_fn`,
    `self._data_init_fn`, `self._state_init_fn`, `self._split`, ...), which point at
    the right library for the backend. See minimal_mjx's
    [Writing an environment](https://dynamicmobility.github.io/minimal-mjx/guides/environment/)
    for the full table.

## How the reward vector is formed

The environment returns a dict of **named reward terms**. The config groups them into
objectives, and
[`get_reward_and_metrics`][moplayground.envs.generic.mobase.MultiObjectiveBase.get_reward_and_metrics]
builds the reward vector:

```text
reward[i] = Σ_{k ∈ objectives[i]} weights[k] · r[k]  +  Σ_{k ∈ shared_objectives} weights[k] · r[k]
```

The reward vector has one entry per objective, and
each weighted term is also logged in `metrics`. The environment never decides what
the objectives are; it only defines the terms.

## What to implement

### `__init__`

Pass the MuJoCo XML, the `env_params` from the config, and the backend:

```python
class MOCheetah(MultiObjectiveBase):
    def __init__(self, env_params, backend, xml_path=CheetahInterface.XML):
        super().__init__(
            xml_path   = xml_path,
            env_params = env_params,
            backend    = backend,
        )
```

`MultiObjectiveBase` reads `reward.optimization.objectives` and `shared_objectives`
from `env_params`, then `SwappableBase` compiles the model and builds the backend
handles. `num_free` (default 3) is the number of `qpos` entries of the floating base.
Keep model paths and constants (XML path, default joint positions, body ids) in an
interface class, as in [`dmcontrol/interface.py`][moplayground.envs.dmcontrol.interface].

### `reset(rng)`

Build the initial `data`, compute the first observation and reward, and return a
state:

```python
def reset(self, rng):
    rng, qpos_key, qvel_key = self._split(rng, 3)
    qpos = self._np.hstack([
        self._np.array(CheetahInterface.DEFAULT_FF),
        self._np.array(CheetahInterface.DEFAULT_JT)
    ])
    qvel = self._np.zeros(self.mj_model.nv)
    ctrl = self._np.zeros(self.mj_model.nu)
    data = self._data_init_fn(qpos=qpos, qvel=qvel, ctrl=ctrl, time=0.0,
                              xfrc_applied=self._np.zeros((self._mj_model.nbody, 6)))
    info = {'xposbefore': 0.0, 'xposafter': 0.01, 'ang': 0.0}

    done = self._np.array(0.0)
    rewards = self.reward_function(data=data, action=ctrl, info=info, done=False)
    reward, metrics = self.get_reward_and_metrics(rewards, {})
    obs = self._get_obs(data, info)
    return self._state_init_fn(data, obs, reward, done, metrics, info)
```

Under JAX the state's structure is fixed after `reset`: every key that `step` writes
to `info` or `metrics` must already exist.

### `step(state, action)`

Apply the action, advance the physics, and recompute the reward:

```python
def step(self, state, action):
    state.info['xposbefore'] = state.data.qpos[0]
    action = self._np.clip(self.params.action_scale * action, -1.0, 1.0)
    data = self._step_fn(state.data, action)
    state.info['xposafter'] = data.qpos[0]
    state.info['ang']       = data.qpos[2]

    done = self.fall_termination(state.info)
    rewards = self.reward_function(data=data, action=action, info=state.info, done=done)
    reward, metrics = self.get_reward_and_metrics(rewards, state.metrics)
    obs = self._get_obs(data, state.info)
    return self._state_init_fn(data, obs, reward, done.astype(float), metrics, state.info)
```

### `reward_function(data, action, info, done)`

Return every reward term that any objective might use:

```python
def reward_function(self, data, action, info, done):
    return {
        'alive'  : self.reward_alive(),
        'energy' : self.reward_energy(action),
        'height' : self.reward_height(data),
        'run'    : self.reward_run(info),
        'done'   : self.reward_done(done),
    }
```

Every returned name needs an entry in `reward.weights`. A term with weight `0`, or
that appears in no objective, is still logged but does not affect training.

### `_get_obs(data, info)` and `action_size`

```python
def _get_obs(self, data, info):
    obs = self._np.concatenate([data.qpos[1:], self._np.clip(data.qvel, -10.0, 10.0)])
    return {'state': obs, 'privileged_state': obs}

@property
def action_size(self):
    return 6
```

The policy and value networks read `obs['state']`.

## Registering the environment

Add a case to [`create_environment`][moplayground.envs.create.create_environment] in
`src/moplayground/envs/create.py`:

```python
match env_name:
    ...
    case 'MyRobot':
        from moplayground.envs.my_robot import MyRobot
        env = MyRobot(**common_kwargs)
```

Optionally expose the class at the top level by adding it to `_EXPORTS` in
`src/moplayground/__init__.py`.

## Writing the config

Copy a config from `config/`, set `env: MyRobot`, and define the reward terms and
objectives:

```yaml
env: MyRobot
env_config:
  ctrl_dt: 0.02
  sim_dt: 0.004
  reward:
    weights:      {forward: 1.0, energy: 0.1, alive: 1.0}
    optimization:
      objectives:            [['forward'], ['energy']]
      shared_objectives:     ['alive']
      labels:                ['Forward Speed', 'Energy Efficiency']
      default_scalarization: [1, 1]
```

See [The config file](configuration.md) for every key.

## Checking the environment

Before training, roll the environment out with a dummy controller on both backends
(see [Rolling out a policy](../examples/rollout.md)):

```bash
python -m scripts.rollout config/myrobot.yaml --inference_fn zero     # backend: np
python -m scripts.rollout config/myrobot.yaml --inference_fn random   # then backend: jnp
```

The `np` run checks dynamics, rewards and termination with ordinary Python debugging;
the `jnp` run checks that the code traces under JAX.

The test suite checks every config in `config/`. After adding an environment, add its
name to `ENVS` in `tests/unit/test_configs_dir.py` and run
`pytest tests/unit/test_configs_dir.py`.
