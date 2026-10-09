---
layout: default
title: "moplayground.envs.create"
parent: "moplayground.envs"
grand_parent: API Reference
---

<!-- markdownlint-disable -->

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/envs/create.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `moplayground.envs.create`





---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/envs/create.py#L1"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_environment`

```python
create_environment(
    env_name,
    env_params,
    backend='jnp',
    gaitlib_path=None,
    **env_kwargs
)
```

Instantiate a MO-Playground environment. 

Constructs one of the registered multi-objective environments (``MOCheetah``, ``MOHopper``, ``MOAnt``, ``MOWalker``, ``MOHumanoid``, ``NaviGait``). 



**Args:**
 
 - <b>`env_name`</b>:  Registered environment name, e.g. ``'MOCheetah'``. 
 - <b>`env_params`</b>:  ``ConfigDict`` of environment parameters (the  ``env_config`` section of a run config). 
 - <b>`backend`</b>:  ``'jnp'`` (JAX, for training) or ``'np'`` (NumPy, for  evaluation/rollout). 
 - <b>`gaitlib_path`</b>:  Gait library path. Required for ``NaviGait``. 
 - <b>`**env_kwargs`</b>:  Extra keyword arguments forwarded to the environment  constructor. Currently only consumed by ``NaviGait`` (Bruce). 



**Returns:**
 Tuple ``(env, env_params)`` where ``env`` is the constructed environment instance and ``env_params`` is the ``ConfigDict`` of environment parameters passed in. 



**Raises:**
 
 - <b>`Exception`</b>:  If ``env_name`` does not match a registered environment. 


