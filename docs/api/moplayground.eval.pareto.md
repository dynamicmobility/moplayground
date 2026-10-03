---
layout: default
title: "moplayground.eval.pareto"
parent: "moplayground.eval"
grand_parent: API Reference
---

<!-- markdownlint-disable -->

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/eval/pareto.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `moplayground.eval.pareto`





---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/eval/pareto.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_pareto_rollout`

```python
get_pareto_rollout(env, N_STEPS, make_policy)
```

Build a jit/vmapped rollout fn that sweeps a batch of (key, directive, params). 

The same scaffold works for any algorithm — algorithm-specific details live inside ``make_policy``. 

Parameters 
---------- env : the brax/mjx environment to roll out in. N_STEPS : int, number of env steps per rollout. make_policy : callable with signature  ``(params, deterministic, directive) -> policy(obs, key) -> (action, extras)``. 

Returns 
------- run_rollouts : jit-compiled fn ``(keys, directives, params) -> ((final_state, returns), None)``  vmapped over the leading axis of ``keys`` and ``directives``. ``returns`` is  the per-objective episodic return, shape ``(NUM_ENVS, num_objs)``. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/eval/pareto.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_fronts`

```python
compute_fronts(
    configs,
    rng,
    env,
    N_STEPS,
    NUM_ENVS,
    make_policy,
    load_params_fn,
    model_files_per_config,
    save_results
)
```

Run rollouts for every checkpoint in every run over a fixed batch of randomly sampled directives, and return the resulting Pareto fronts. 

Algorithm-agnostic: callers supply ``make_policy`` and ``load_params_fn`` to plug in MORLAX, AMOR, or anything else with the right shapes. The rollout JIT is built once (from the first config's network) and reused across every config in ``configs`` — so multiple runs that share an environment and network architecture amortize the compile cost. 

Parameters 
---------- configs : list of training configs, all sharing the same environment. The  first config is used to read the number of objectives; ``save_results``  uses each file's owning config to choose the output directory. rng : PRNGKey used both to sample the (fixed) batch of directives and to  seed the per-rollout env reset keys. env : environment to roll out in. N_STEPS : int, env steps per rollout. NUM_ENVS : int, number of directives sampled (== number of parallel rollouts  per checkpoint). make_policy : callable with signature  ``(params, deterministic, directive) -> policy(obs, key) -> (action, extras)``,  passed through to :func:`get_pareto_rollout`. load_params_fn : callable ``file -> params`` mapping a checkpoint path to the  params pytree expected by ``make_policy``. model_files_per_config : list of lists of checkpoint paths, aligned with  ``configs``. ``model_files_per_config[i]`` are the checkpoints for  ``configs[i]``. save_results : if True, dump per-checkpoint returns to  ``{cfg.save_dir}/{cfg.name}/obj{j}.txt`` (where ``cfg`` is the file's  owning config and ``j`` is its index within that config's checkpoints). 

Returns 
------- rewards_over_iters : ndarray, shape ``(total_checkpoints, NUM_ENVS, num_objs)``,  per-objective episodic return. Files are concatenated in config order. tradeoffs_over_iters : ndarray, same shape, the directive used for each  rollout. Fixed across checkpoints (broadcast along axis 0). 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/eval/pareto.py#L171"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_morlax_fronts`

```python
get_morlax_fronts(
    config,
    rng,
    env,
    N_STEPS,
    NUM_ENVS,
    save_results=False,
    only_final=False
)
```

Compute Pareto fronts for one or more MORLAX (hypernetwork) runs. 

Parameters 
---------- config : training config dict, or list of training config dicts. When a  list is passed, every config must deploy the same environment (same  ``env`` field and ``env_config`` contents); the rollout JIT is built  once from the first config and reused across all of them. ``ValueError``  is raised if envs disagree. rng : PRNGKey used both to sample the (fixed) batch of directives and to  seed the per-rollout env reset keys. env : environment to roll out in. N_STEPS : int, env steps per rollout. NUM_ENVS : int, number of directives sampled (== number of parallel rollouts  per checkpoint). save_results : if True, dump per-checkpoint returns to  ``{cfg.save_dir}/{cfg.name}/obj{j}.txt`` for each owning config. only_final : if True, evaluate only the final checkpoint of each config. 

Returns 
------- rewards_over_iters : ndarray, shape ``(total_checkpoints, NUM_ENVS, num_objs)``,  per-objective episodic return. Files are concatenated in config order. tradeoffs_over_iters : ndarray, same shape, the directive used for each  rollout. Fixed across checkpoints (broadcast along axis 0). 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/eval/pareto.py#L222"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_amor_fronts`

```python
get_amor_fronts(
    config,
    rng,
    env,
    N_STEPS,
    NUM_ENVS,
    save_results=False,
    only_final=False
)
```

Compute Pareto fronts for one or more AMOR (tradeoff-conditioned) runs. 

Parameters 
---------- config : training config dict, or list of training config dicts. When a  list is passed, every config must deploy the same environment (same  ``env`` field and ``env_config`` contents); the rollout JIT is built  once from the first config and reused across all of them. ``ValueError``  is raised if envs disagree. rng : PRNGKey used both to sample the (fixed) batch of directives and to  seed the per-rollout env reset keys. env : environment to roll out in. N_STEPS : int, env steps per rollout. NUM_ENVS : int, number of directives sampled (== number of parallel rollouts  per checkpoint). save_results : if True, dump per-checkpoint returns to  ``{cfg.save_dir}/{cfg.name}/obj{j}.txt`` for each owning config. only_final : if True, evaluate only the final checkpoint of each config. 

Returns 
------- rewards_over_iters : ndarray, shape ``(total_checkpoints, NUM_ENVS, num_objs)``,  per-objective episodic return. Files are concatenated in config order. tradeoffs_over_iters : ndarray, same shape, the directive used for each  rollout. Fixed across checkpoints (broadcast along axis 0). 


