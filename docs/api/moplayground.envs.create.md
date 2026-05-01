---
layout: default
title: "moplayground.envs.create"
parent: API Reference
---

<!-- markdownlint-disable -->

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/envs/create.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `moplayground.envs.create`





---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/envs/create.py#L3"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_environment`

```python
create_environment(config, for_training=False, **env_kwargs)
```

Instantiate a MO-Playground environment from a config. 

Dispatches on ``config['env']`` to construct one of the registered multi-objective environments (``MOCheetah``, ``MOHopper``, ``MOAnt``, ``MOWalker``, ``MOHumanoid``, ``NaviGait``). 



**Args:**
 
 - <b>`config`</b>:  Config dict (typically loaded from a YAML file in ``config/``)  with at least ``env`` (str) and ``env_config`` (dict) entries.  ``backend`` is read when ``for_training`` is False. 
 - <b>`for_training`</b>:  If True, force the JAX (``'jnp'``) backend regardless of  ``config['backend']``. Set this when building the environment for  GPU training; leave False for evaluation/rollout. 
 - <b>`**env_kwargs`</b>:  Extra keyword arguments forwarded to the environment  constructor. Currently only consumed by ``NaviGait`` (Bruce). 



**Returns:**
 Tuple ``(env, env_params)`` where ``env`` is the constructed environment instance and ``env_params`` is the resolved ``ConfigDict`` of environment parameters. 



**Raises:**
 
 - <b>`Exception`</b>:  If ``config['env']`` does not match a registered  environment name. 


