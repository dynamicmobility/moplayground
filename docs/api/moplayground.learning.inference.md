---
layout: default
title: "moplayground.learning.inference"
parent: "moplayground.learning"
grand_parent: API Reference
---

<!-- markdownlint-disable -->

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/inference.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `moplayground.learning.inference`





---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/inference.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_mo_policy`

```python
load_mo_policy(
    config,
    tradeoff: numpy.ndarray,
    network_factory=<function make_moppo_networks at 0x797cd2801b20>,
    deterministic: bool = True
)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/inference.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_moppo_network`

```python
load_moppo_network(
    config,
    network_factory=<function make_moppo_networks at 0x797cd2801b20>,
    path=None
) → tuple[moplayground.moppo.factory.MOPPONetworks, dict]
```

Loads the MOPPO object 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/inference.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_hypernetwork`

```python
load_hypernetwork(
    config,
    network_factory=<function make_moppo_networks at 0x797cd2801b20>,
    path=None
)
```

Loads policy inference function from PPO checkpoint. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/inference.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rollout_policy`

```python
rollout_policy(
    env,
    config,
    tradeoff=None,
    T=10.0,
    camera='track',
    width=1080,
    height=720
)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/inference.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_num_objectives`

```python
get_num_objectives(config)
```






