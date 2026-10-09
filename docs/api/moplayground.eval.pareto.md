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

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/eval/pareto.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/eval/pareto.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_fronts`

```python
compute_fronts(
    run_dirs,
    num_objectives,
    rng,
    env,
    N_STEPS,
    NUM_ENVS,
    make_policy,
    load_params_fn,
    model_files_per_run,
    save_results
)
```

Run rollouts for every checkpoint in every run over a fixed batch of randomly sampled directives, and return the resulting Pareto fronts. 

Algorithm-agnostic: callers supply ``make_policy`` and ``load_params_fn`` to plug in MORLAX, AMOR, or anything else with the right shapes. The rollout JIT is built once and reused across every run in ``run_dirs`` — so multiple runs that share an environment and network architecture amortize the compile cost. 

Parameters 
---------- run_dirs : list of run directories, all sharing the same environment.  ``save_results`` writes into each file's owning run directory. num_objectives : int, number of objectives. rng : PRNGKey used both to sample the (fixed) batch of directives and to  seed the per-rollout env reset keys. env : environment to roll out in. N_STEPS : int, env steps per rollout. NUM_ENVS : int, number of directives sampled (== number of parallel rollouts  per checkpoint). make_policy : callable with signature  ``(params, deterministic, directive) -> policy(obs, key) -> (action, extras)``,  passed through to :func:`get_pareto_rollout`. load_params_fn : callable ``file -> params`` mapping a checkpoint path to the  params pytree expected by ``make_policy``. model_files_per_run : list of lists of checkpoint paths, aligned with  ``run_dirs``. ``model_files_per_run[i]`` are the checkpoints for  ``run_dirs[i]``. save_results : if True, dump per-checkpoint returns to  ``{run_dir}/obj{j}.txt`` (where ``run_dir`` is the file's owning run  directory and ``j`` is its index within that run's checkpoints). 

Returns 
------- rewards_over_iters : ndarray, shape ``(total_checkpoints, NUM_ENVS, num_objs)``,  per-objective episodic return. Files are concatenated in run order. tradeoffs_over_iters : ndarray, same shape, the directive used for each  rollout. Fixed across checkpoints (broadcast along axis 0). 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/eval/pareto.py#L138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_morlax_fronts`

```python
get_morlax_fronts(
    run_dirs,
    network_params,
    num_objectives,
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
---------- run_dirs : run directory, or list of run directories. Every run must deploy  the same environment and network architecture; the rollout JIT is  built once and reused across all of them. Check this before calling  (see ``moplayground.config.check_same_env``). network_params : dict, network-factory keyword arguments shared by all runs. num_objectives : int, number of objectives. rng : PRNGKey used both to sample the (fixed) batch of directives and to  seed the per-rollout env reset keys. env : environment to roll out in. N_STEPS : int, env steps per rollout. NUM_ENVS : int, number of directives sampled (== number of parallel rollouts  per checkpoint). save_results : if True, dump per-checkpoint returns to  ``{run_dir}/obj{j}.txt`` for each owning run. only_final : if True, evaluate only the final checkpoint of each run. 

Returns 
------- rewards_over_iters : ndarray, shape ``(total_checkpoints, NUM_ENVS, num_objs)``,  per-objective episodic return. Files are concatenated in run order. tradeoffs_over_iters : ndarray, same shape, the directive used for each  rollout. Fixed across checkpoints (broadcast along axis 0). 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/eval/pareto.py#L194"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_amor_fronts`

```python
get_amor_fronts(
    run_dirs,
    network_params,
    num_objectives,
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
---------- run_dirs : run directory, or list of run directories. Every run must deploy  the same environment and network architecture; the rollout JIT is  built once and reused across all of them. Check this before calling  (see ``moplayground.config.check_same_env``). network_params : dict, network-factory keyword arguments shared by all runs. num_objectives : int, number of objectives. rng : PRNGKey used both to sample the (fixed) batch of directives and to  seed the per-rollout env reset keys. env : environment to roll out in. N_STEPS : int, env steps per rollout. NUM_ENVS : int, number of directives sampled (== number of parallel rollouts  per checkpoint). save_results : if True, dump per-checkpoint returns to  ``{run_dir}/obj{j}.txt`` for each owning run. only_final : if True, evaluate only the final checkpoint of each run. 

Returns 
------- rewards_over_iters : ndarray, shape ``(total_checkpoints, NUM_ENVS, num_objs)``,  per-objective episodic return. Files are concatenated in run order. tradeoffs_over_iters : ndarray, same shape, the directive used for each  rollout. Fixed across checkpoints (broadcast along axis 0). 


