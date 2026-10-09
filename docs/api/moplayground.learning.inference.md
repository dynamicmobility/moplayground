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

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/inference.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_all_models`

```python
get_all_models(run_dir) → list[pathlib.Path]
```

Checkpoint directories in ``run_dir``, sorted by training step. 

A checkpoint directory is a subdirectory whose name is an integer (the training step), e.g. ``000050032640``. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/inference.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_last_model`

```python
get_last_model(run_dir) → Path
```

Most recent checkpoint directory in ``run_dir``. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/inference.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_mo_policy`

```python
load_mo_policy(
    algorithm: str,
    network_params: dict,
    num_objectives: int,
    run_dir,
    tradeoff: numpy.ndarray,
    network_factory=None,
    deterministic: bool = True
)
```

Load the latest multi-objective policy in ``run_dir``. 

Dispatches on ``algorithm`` (``'morlax'`` or ``'amor'``). Returns a 2-arg callable ``policy(obs, key) -> (action, extras)`` with ``tradeoff`` baked in, so it is compatible with ``mm.eval.rollout_policy`` and other consumers. 

For AMOR specifically, the underlying inference function natively accepts a directive at call time (so the tradeoff can change per step). To get that 3-arg form, call :func:`load_amor_inference_fn` directly instead of this function. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/inference.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_hypernetworks`

```python
load_hypernetworks(
    network_params: dict,
    num_objectives: int,
    run_dir=None,
    path=None,
    network_factory=<function make_morlax_networks at 0x7ffa3c96b600>,
    quiet=True
) → tuple[moplayground.moppo.factory.MORLAXNetworks, dict]
```

Load MORLAX networks and hypernetwork params from a checkpoint. 



**Args:**
 
 - <b>`network_params`</b>:  MORLAX network-factory keyword arguments. 
 - <b>`num_objectives`</b>:  Number of objectives. 
 - <b>`run_dir`</b>:  Run directory; the latest checkpoint in it is used when  ``path`` is not given. 
 - <b>`path`</b>:  Explicit checkpoint directory. Overrides ``run_dir``. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/inference.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_hypernetwork_inference_fn`

```python
load_hypernetwork_inference_fn(
    network_params: dict,
    num_objectives: int,
    run_dir=None,
    path=None,
    network_factory=<function make_morlax_networks at 0x7ffa3c96b600>
)
```

Loads policy inference function from PPO checkpoint. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/inference.py#L144"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_amor_networks`

```python
load_amor_networks(
    network_params: dict,
    num_objectives: int,
    run_dir=None,
    path=None,
    network_factory=<function make_amor_networks at 0x7ffa3c980220>,
    quiet=True
) → tuple
```

Load AMOR networks + saved (normalizer, policy, value) params. 



**Args:**
 
 - <b>`network_params`</b>:  AMOR network-factory keyword arguments. 
 - <b>`num_objectives`</b>:  Number of objectives. 
 - <b>`run_dir`</b>:  Run directory; the latest checkpoint in it is used when  ``path`` is not given. 
 - <b>`path`</b>:  Explicit checkpoint directory. Overrides ``run_dir``. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/inference.py#L178"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_make_amor_inference_fn`

```python
load_make_amor_inference_fn(
    network_params: dict,
    num_objectives: int,
    run_dir=None,
    path=None,
    network_factory=<function make_amor_networks at 0x7ffa3c980220>
)
```

Load the (call-time-directive) AMOR inference function from a checkpoint. 

Returns ``(amor_inference_fn, saved_params)`` where ``amor_inference_fn`` has signature ``(params, deterministic) -> policy(obs, directive, key)`` and ``saved_params`` is the ``(normalizer, policy, value)`` 3-tuple from the checkpoint. 


