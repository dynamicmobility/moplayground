---
layout: default
title: "moplayground.learning.training"
parent: "moplayground.learning"
grand_parent: API Reference
---

<!-- markdownlint-disable -->

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/training.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `moplayground.learning.training`





---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/training.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `setup_morlax`

```python
setup_morlax(config)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/training.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `setup_amor`

```python
setup_amor(config)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/training.py#L65"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_training_directory`

```python
create_training_directory(config, warn_github_changes=True)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/training.py#L85"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `train_policy`

```python
train_policy(
    config,
    env,
    eval_env,
    run=None,
    handle_params=None,
    warn_github_changes=False,
    progress_fn=None
)
```

Train a policy on the given environment. 

Sets up the GPU, builds MOPPO network parameters from ``config``, saves the resolved config alongside the run, and dispatches to either the standard single-objective trainer (when ``config.mo2so.enabled`` is True — wrapping ``env``/``eval_env`` with ``Multi2SingleObjective``) or the multi-objective ``mo_train`` loop. 



**Args:**
 
 - <b>`config`</b>:  Training config (ConfigDict). Must include ``save_dir``,  ``name``, ``mo2so`` (with ``enabled`` and, if enabled,  ``weighting``), and ``learning_params``. 
 - <b>`env`</b>:  Training environment. 
 - <b>`eval_env`</b>:  Evaluation environment used for periodic rollouts. 
 - <b>`run`</b>:  (optional) Experiment-tracking handle (e.g. a wandb run) forwarded to the  multi-objective trainer; ignored on the single-objective path. 
 - <b>`handle_params`</b>:  (optional) Callable ``config -> (train_fn, network_factory)``.  Defaults to the handler registered for ``config.algorithm`` in  ``_ALGO_HANDLERS``. 
 - <b>`warn_github_changes`</b>:  (optional) If True, warn about uncommitted git  changes when creating the training directory. Defaults to False. 
 - <b>`progress_fn`</b>:  (optional) Callback invoked each eval step as  ``progress_fn(run, num_steps, metrics, save_dir, training_data)``  to log/plot training progress. Defaults to  ``mop.utils.plotting.plot_mo_progress``. 



**Returns:**
 Tuple ``(make_inference_fn, params)`` — a factory that builds an inference function and the trained policy parameters. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/training.py#L184"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `mo_wrapper`

```python
mo_wrapper(
    env: mujoco_playground._src.mjx_env.MjxEnv,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn=None
) → Wrapper
```

Multi-Objective Wrapper 


