---
layout: default
title: "moplayground.learning.wrappers"
parent: API Reference
---

<!-- markdownlint-disable -->

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/wrappers.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `moplayground.learning.wrappers`






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/wrappers.py#L6"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultiObjectiveEpisodeWrapper`
Maintains episode step count and sets done at episode end. 

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/wrappers.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MultiObjectiveEpisodeWrapper.__init__`

```python
__init__(env: brax.envs.base.Env, episode_length: int, action_repeat: int)
```






---

#### <kbd>property</kbd> MultiObjectiveEpisodeWrapper.action_size





---

#### <kbd>property</kbd> MultiObjectiveEpisodeWrapper.backend





---

#### <kbd>property</kbd> MultiObjectiveEpisodeWrapper.observation_size





---

#### <kbd>property</kbd> MultiObjectiveEpisodeWrapper.unwrapped







---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/wrappers.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MultiObjectiveEpisodeWrapper.reset`

```python
reset(rng: jax.Array) → State
```





---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/wrappers.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MultiObjectiveEpisodeWrapper.step`

```python
step(state: brax.envs.base.State, action: jax.Array) → State
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/wrappers.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultiObjectiveEvalWrapper`
Brax env with eval metrics. 


---

#### <kbd>property</kbd> MultiObjectiveEvalWrapper.action_size





---

#### <kbd>property</kbd> MultiObjectiveEvalWrapper.backend





---

#### <kbd>property</kbd> MultiObjectiveEvalWrapper.observation_size





---

#### <kbd>property</kbd> MultiObjectiveEvalWrapper.unwrapped







---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/wrappers.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MultiObjectiveEvalWrapper.reset`

```python
reset(rng: jax.Array) → State
```





---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/wrappers.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MultiObjectiveEvalWrapper.step`

```python
step(state: brax.envs.base.State, action: jax.Array) → State
```






