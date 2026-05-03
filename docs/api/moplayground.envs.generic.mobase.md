---
layout: default
title: "moplayground.envs.generic.mobase"
parent: "moplayground.envs"
grand_parent: API Reference
---

<!-- markdownlint-disable -->

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/envs/generic/mobase.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `moplayground.envs.generic.mobase`






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/envs/generic/mobase.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultiObjectiveBase`
Base environment that emits a vector-valued reward. 

Extends ``minimal_mjx.envs.generic.base.SwappableBase`` so concrete environments can be constructed against either NumPy or JAX backends. Per-step reward components are bucketed into objective groups defined by ``env_params.reward.optimization.objectives`` (one entry per output dimension of the reward vector) and an optional set of ``shared_objectives`` that are added to every dimension. 



**Args:**
 
 - <b>`xml_path`</b>:  Path to the MuJoCo XML model. 
 - <b>`env_params`</b>:  ConfigDict with at least ``reward.weights``,  ``reward.optimization.objectives`` (list of lists of reward keys,  one list per objective dimension), and  ``reward.optimization.shared_objectives`` (list of reward keys  added to every objective). 
 - <b>`backend`</b>:  ``'jnp'`` for JAX (training), ``'np'`` for NumPy (eval). 
 - <b>`num_free`</b>:  Number of free joints in the model. Forwarded to  ``SwappableBase``. 

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/envs/generic/mobase.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MultiObjectiveBase.__init__`

```python
__init__(
    xml_path: pathlib.Path,
    env_params: ml_collections.config_dict.config_dict.ConfigDict,
    backend: str = 'jnp',
    num_free: int = 3
)
```






---

#### <kbd>property</kbd> MultiObjectiveBase.action_size

Required action size for the environment 

---

#### <kbd>property</kbd> MultiObjectiveBase.dt

Control timestep for the environment. 

---

#### <kbd>property</kbd> MultiObjectiveBase.mj_model





---

#### <kbd>property</kbd> MultiObjectiveBase.mjx_model





---

#### <kbd>property</kbd> MultiObjectiveBase.n_substeps

Number of sim steps per control step. 

---

#### <kbd>property</kbd> MultiObjectiveBase.observation_size





---

#### <kbd>property</kbd> MultiObjectiveBase.sim_dt

Simulation timestep for the environment. 

---

#### <kbd>property</kbd> MultiObjectiveBase.unwrapped





---

#### <kbd>property</kbd> MultiObjectiveBase.xml_path







---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/envs/generic/mobase.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MultiObjectiveBase.get_reward_and_metrics`

```python
get_reward_and_metrics(
    rewards: jax.Array,
    metrics: dict
) → tuple[jax.Array, dict[str, jax.Array]]
```

Combine per-key rewards into a vector reward plus updated metrics. 

Each entry of ``self.objectives`` maps to one component of the returned reward vector, computed as a weighted sum of the listed per-key rewards. Shared objectives are then added to every component. 



**Args:**
 
 - <b>`rewards`</b>:  Mapping from reward key to scalar reward for the  current step. 
 - <b>`metrics`</b>:  Existing metrics dict to extend. 



**Returns:**
 ``(reward, metrics)`` where ``reward`` has shape ``(len(self.objectives),)`` and ``metrics`` is the updated metrics dict. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/envs/generic/mobase.py#L81"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Multi2SingleObjective`
Wrap a multi-objective env to expose a scalar reward. 

Replaces the vector reward from the wrapped environment with the inner product ``reward · weighting``, so the wrapped env can be plugged into standard single-objective PPO. All other attributes/methods are delegated to the underlying ``env`` via ``__getattr__``. 



**Args:**
 
 - <b>`env`</b>:  A ``MultiObjectiveBase`` (or compatible) environment whose  ``reset``/``step`` return states with vector rewards. 
 - <b>`weighting`</b>:  Per-objective weights, length must match the env's  reward dimension. 

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/envs/generic/mobase.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Multi2SingleObjective.__init__`

```python
__init__(env, weighting)
```








---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/envs/generic/mobase.py#L100"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Multi2SingleObjective.reset`

```python
reset(rng)
```





---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/envs/generic/mobase.py#L104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Multi2SingleObjective.step`

```python
step(state, action)
```






