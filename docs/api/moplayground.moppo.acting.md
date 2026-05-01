---
layout: default
title: "moplayground.moppo.acting"
parent: API Reference
---

<!-- markdownlint-disable -->

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/acting.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `moplayground.moppo.acting`
Brax training acting functions. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/acting.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `actor_step`

```python
actor_step(
    env: brax.envs.base.Env,
    env_state: brax.envs.base.State,
    policy: brax.training.types.Policy,
    directive,
    key: jax.Array,
    extra_fields: Sequence[str] = ()
) → Tuple[brax.envs.base.State, moplayground.moppo.acting.MultiObjectiveTransition]
```

Collect data. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/acting.py#L72"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `generate_unroll`

```python
generate_unroll(
    env: brax.envs.base.Env,
    env_state: brax.envs.base.State,
    policy: brax.training.types.Policy,
    directive: jax.Array,
    key: jax.Array,
    unroll_length: int,
    extra_fields: Sequence[str] = ()
) → Tuple[brax.envs.base.State, moplayground.moppo.acting.MultiObjectiveTransition]
```

Collect trajectories of given unroll_length. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/acting.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultiObjectiveTransition`
Container for a transition. 





---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/acting.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Evaluator`
Class to run evaluations. 

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/acting.py#L102"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Evaluator.__init__`

```python
__init__(
    eval_env: brax.envs.base.Env,
    num_objs: int,
    eval_policy_fn,
    num_eval_envs: int,
    episode_length: int,
    action_repeat: int,
    key: jax.Array
)
```








---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/acting.py#L136"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Evaluator.run_evaluation`

```python
run_evaluation(
    policy_params,
    training_metrics: Mapping[str, jax.Array],
    aggregate_episodes: bool = True
) → Mapping[str, jax.Array]
```

Run one epoch of evaluation. 


