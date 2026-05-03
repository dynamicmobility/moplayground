---
layout: default
title: "moplayground.learning.startup"
parent: "moplayground.learning"
grand_parent: API Reference
---

<!-- markdownlint-disable -->

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/startup.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `moplayground.learning.startup`





---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/learning/startup.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `train_policy`

```python
train_policy(config, env, eval_env, run)
```

Train a policy on the given environment. 

Sets up the GPU, builds PPO/network parameters from ``config``, saves the resolved config alongside the run, and dispatches to either the standard single-objective trainer (when ``config.mo2so.enabled`` is True — wrapping ``env``/``eval_env`` with ``Multi2SingleObjective``) or the multi-objective ``mo_train`` loop. 



**Args:**
 
 - <b>`config`</b>:  Training config (ConfigDict). Must include ``save_dir``,  ``name``, ``mo2so`` (with ``enabled`` and, if enabled,  ``weighting``), and ``learning_params``. 
 - <b>`env`</b>:  Training environment. 
 - <b>`eval_env`</b>:  Evaluation environment used for periodic rollouts. 
 - <b>`run`</b>:  Experiment-tracking handle (e.g. a wandb run) forwarded to the  multi-objective trainer; ignored on the single-objective path. 



**Returns:**
 Tuple ``(make_inference_fn, params)`` — a factory that builds an inference function and the trained policy parameters. 


