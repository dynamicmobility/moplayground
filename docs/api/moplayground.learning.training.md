---
layout: default
title: "moplayground.learning.training"
parent: API Reference
---

<!-- markdownlint-disable -->

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/training.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `moplayground.learning.training`





---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/training.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_mo_progress`

```python
plot_mo_progress(
    num_steps: int,
    metrics: dict,
    training_data: moplayground.learning.training.MOTrainingInfo,
    save_dir: pathlib.Path,
    run: wandb.sdk.wandb_run.Run = None
)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/training.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/training.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `mo_train`

```python
mo_train(
    config_yaml: dict,
    output_dir: pathlib.Path,
    env: moplayground.envs.generic.mobase.MultiObjectiveBase,
    eval_env: moplayground.envs.generic.mobase.MultiObjectiveBase,
    moppo_params,
    network_params,
    policy_init_params,
    run: wandb.sdk.wandb_run.Run
)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/training.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MOTrainingInfo`
MOTrainingInfo(start_time: float, times: list = <factory>, iterations: list = <factory>, paretos: list = <factory>, directives: list = <factory>, labels: list = <factory>) 

<a href="https://github.com/dynamicmobility/moplayground/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MOTrainingInfo.__init__`

```python
__init__(
    start_time: float,
    times: list = <factory>,
    iterations: list = <factory>,
    paretos: list = <factory>,
    directives: list = <factory>,
    labels: list = <factory>
) → None
```








---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/training.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MOTrainingInfo.save`

```python
save(save_dir, create_time=True)
```






