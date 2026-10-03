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

## <kbd>function</kbd> `load_mo_policy`

```python
load_mo_policy(
    config,
    tradeoff: numpy.ndarray,
    network_factory=None,
    deterministic: bool = True
)
```

Load a multi-objective policy for the configured algorithm. 

Dispatches on ``config.algorithm``. Returns a 2-arg callable ``policy(obs, key) -> (action, extras)`` with ``tradeoff`` baked in, so it is compatible with ``mm.eval.rollout_policy`` and other consumers. 

For AMOR specifically, the underlying inference function natively accepts a directive at call time (so the tradeoff can change per step). To get that 3-arg form, call :func:`load_amor_inference_fn` directly instead of this function. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/inference.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_hypernetworks`

```python
load_hypernetworks(
    config,
    network_factory=<function make_morlax_networks at 0x7ffa3c976340>,
    path=None,
    quiet=True
) → tuple[moplayground.moppo.factory.MORLAXNetworks, dict]
```

Loads the MOPPO object 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/inference.py#L95"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_hypernetwork_inference_fn`

```python
load_hypernetwork_inference_fn(
    config,
    network_factory=<function make_morlax_networks at 0x7ffa3c976340>,
    path=None
)
```

Loads policy inference function from PPO checkpoint. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/inference.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_amor_networks`

```python
load_amor_networks(
    config,
    network_factory=<function make_amor_networks at 0x7ffa3c976f20>,
    path=None,
    quiet=True
) → tuple
```

Load AMOR networks + saved (normalizer, policy, value) params. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/inference.py#L130"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_make_amor_inference_fn`

```python
load_make_amor_inference_fn(
    config,
    network_factory=<function make_amor_networks at 0x7ffa3c976f20>,
    path=None
)
```

Load the (call-time-directive) AMOR inference function from a checkpoint. 

Returns ``(amor_inference_fn, saved_params)`` where ``amor_inference_fn`` has signature ``(params, deterministic) -> policy(obs, directive, key)`` and ``saved_params`` is the ``(normalizer, policy, value)`` 3-tuple from the checkpoint. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/inference.py#L146"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_num_objectives`

```python
get_num_objectives(config)
```






