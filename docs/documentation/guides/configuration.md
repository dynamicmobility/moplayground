# The config file

Every run is described by one YAML file. Scripts read it through
[`MOConfig`][moplayground.utils.config.MOConfig], which exposes each section as a
property and hands plain values to library functions. Library functions never
receive the config itself. Here is `config/mocheetah.yaml`:

??? example "config/mocheetah.yaml"

    ```yaml
    --8<-- "config/mocheetah.yaml"
    ```

## Top-level keys

| Key | Read as | Meaning |
| --- | --- | --- |
| `name` | `cfg.name` | Run name. The run directory is `save_dir/name`. A run named `test` may be overwritten and records no git hash. |
| `save_dir` | `cfg.save_dir` | Parent directory for runs made from this config |
| `algorithm` | `cfg.algorithm` | `morlax`, `amor` or `ppo` |
| `env` | `cfg.env_name` | Environment name, mapped to a class by [`create_environment`][moplayground.envs.create.create_environment] |
| `backend` | `cfg.backend` | `np` or `jnp` for rollouts. Training always uses `jnp`. |
| `gaitlib_path` | `cfg.gaitlib_path` | Gait library directory (`NaviGait` only) |

## `env_config`

Everything under `env_config` is passed to the environment as `env_params`
(`cfg.env_params`) and read inside it as `self.params`. Besides `ctrl_dt` and `sim_dt`
(read by minimal_mjx), the multi-objective layer reads the `reward` section:

```yaml
reward:
  weights:                                   # one weight per named reward term
    run: 1.0
    height: 10.0
    energy: 1.0
    alive: 1.0
    done: 0
  optimization:
    objectives:        [['run'], ['energy'], ['height']]
    shared_objectives: ['alive']
    labels:            ['Running Speed', 'Energy Efficiency', 'Height']
    default_scalarization: [1, 1, 1]
```

| Key | Read as | Meaning |
| --- | --- | --- |
| `weights` | | Weight of each reward term returned by the environment's `reward_function` |
| `objectives` | `cfg.objectives`, `cfg.num_objectives` | One list of reward terms per objective. Objective *i* is the weighted sum of the terms in list *i*. |
| `shared_objectives` | | Weighted terms added to *every* objective, such as an alive bonus or a control cost |
| `labels` | `cfg.objective_labels` | Display names for plots |
| `default_scalarization` | `cfg.default_scalarization` | Fixed objective weights used when `algorithm: ppo` |

The objective grouping is the only place objectives are defined: the environment
returns named terms, and the config decides how they form the reward vector. To train
on a different set of objectives, edit `objectives` and `weights`; no code changes are
needed.

!!! warning "The config is locked"
    MuJoCo Playground locks `env_params` when the environment is built. Reading a key
    that is missing from the YAML raises an error, so add every key the environment
    reads to the config.

## `learning_params`

| Section | Read as | Used by |
| --- | --- | --- |
| `ppo_params` | `cfg.ppo_params` | All algorithms. PPO settings such as `num_timesteps`, `num_envs`, `num_evals`, `episode_length`, `learning_rate`. |
| `sampling_params` | `cfg.sampling_params` | `morlax`, `amor`. Tradeoff sampling: `sampling`, `alpha`, `k`, `warmup_frac` (see [Training a policy](../examples/training.md#5-train)). |
| `morlax_network_params` | `cfg.morlax_network_params` | `morlax`. `hypertype` (`single` or `dual`), `hypersize`, `num_features`, and the target `policy_hidden_layer_sizes` / `value_hidden_layer_sizes`. |
| `amor_network_params` | `cfg.amor_network_params` | `amor`. `policy_hidden_layer_sizes`, `value_hidden_layer_sizes`. |
| `ppo_network_params` | `cfg.ppo_network_params` | `ppo`. `policy_hidden_layer_sizes`, `value_hidden_layer_sizes`. |

For MORLAX, `hypertype: single` uses one shared feature MLP with separate heads for
the policy and value weights ([`ActorCriticHypernet`][moplayground.moppo.networks.ActorCriticHypernet]);
`dual` uses a separate feature MLP per head ([`DualA2CHypernet`][moplayground.moppo.networks.DualA2CHypernet]).
`hypersize` is the hidden layer sizes of the feature MLP, and `num_features` its
output width.

## Adding a key

`MOConfig` is the only code that knows key names. To add a setting, add a property to
`MOConfig` and read the property in the script, rather than indexing the YAML
elsewhere:

```python
class MOConfig(mm.Config):
    @property
    def my_algo_network_params(self) -> dict:
        """Network settings for my algorithm."""
        return self.cfg.learning_params['my_algo_network_params'].to_dict()
```
