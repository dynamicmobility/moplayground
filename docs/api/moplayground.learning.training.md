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

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/training.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `setup_morlax`

```python
setup_morlax(ppo_params, sampling_params, network_params)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/training.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `setup_amor`

```python
setup_amor(ppo_params, sampling_params, network_params)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/training.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `train_policy`

```python
train_policy(
    algorithm,
    ppo_params,
    sampling_params,
    network_params,
    run_dir,
    env,
    eval_env,
    run=None,
    handle_params=None,
    progress_fn=None
)
```

Train a multi-objective policy on the given environment. 

Sets up the GPU, builds the train function and network factory for ``algorithm``, and runs training. Progress plots and checkpoints are written to ``run_dir``. The caller creates ``run_dir`` and writes the run's ``config.yaml`` before calling this function (see ``moplayground.config.save``). 



**Args:**
 
 - <b>`algorithm`</b>:  ``'morlax'`` or ``'amor'``. 
 - <b>`ppo_params`</b>:  PPO keyword arguments shared by both algorithms. 
 - <b>`sampling_params`</b>:  Preference-sampling keyword arguments  (``alpha``, ``k``, ``sampling``, ``warmup_frac``). 
 - <b>`network_params`</b>:  Keyword arguments for the algorithm's network factory. 
 - <b>`run_dir`</b>:  Existing run directory for checkpoints and progress plots. 
 - <b>`env`</b>:  Training environment. 
 - <b>`eval_env`</b>:  Evaluation environment used for periodic rollouts. 
 - <b>`run`</b>:  (optional) Experiment-tracking handle (e.g. a wandb run) used to  log progress and checkpoints. 
 - <b>`handle_params`</b>:  (optional) Callable  ``(ppo_params, sampling_params, network_params) -> (train_fn, network_factory)``.  Defaults to the handler registered for ``algorithm`` in  ``_ALGO_HANDLERS``. 
 - <b>`progress_fn`</b>:  (optional) Callback invoked each eval step as  ``progress_fn(run, num_steps, metrics, save_dir, training_data)``  to log/plot training progress. Defaults to  ``mop.utils.plotting.plot_mo_progress``. 



**Returns:**
 Tuple ``(make_inference_fn, params)`` — a factory that builds an inference function and the trained policy parameters. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/training.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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


