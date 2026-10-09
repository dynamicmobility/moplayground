---
layout: default
title: "moplayground.config"
parent: API Reference
has_children: true
---

<!-- markdownlint-disable -->

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/config.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `moplayground.config`
Single access point between YAML config files and the ``moplayground`` package. 

This is the only module that knows YAML key names. Scripts call :func:`load` to read a config, then use the getter functions to pull plain values out of it and pass those values to package functions. Package code never indexes a config directly. 

Canonical layout:
``` 

     name, save_dir, description, algorithm, env, backend, gaitlib_path      env_config: {...}      learning_params:        ppo_params:           {learning_rate, num_envs, ...}        sampling_params:      {alpha, k, sampling, warmup_frac}        morlax_params:        {hypertype, hypersize, num_features,                               policy_hidden_layer_sizes, value_hidden_layer_sizes}        amor_params:          {policy_hidden_layer_sizes, value_hidden_layer_sizes}        morlax_warmup_params: {enabled, policy} 

```
:func:`load` renames keys from older layouts to the canonical layout, so saved ``config.yaml`` files from old runs still load. 

**Global Variables**
---------------
- **ALGORITHMS**
- **ENVS**
- **SAMPLING_MODES**
- **HYPERTYPES**

---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/config.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load`

```python
load(path) → ConfigDict
```

Read a YAML config, convert it to the canonical layout, and validate it. 



**Args:**
 
 - <b>`path`</b>:  Path to a YAML config file. Either a file in ``config/`` or a  ``config.yaml`` saved in a run directory. 



**Returns:**
 ``ConfigDict`` in the canonical layout. 



**Raises:**
 
 - <b>`ValueError`</b>:  If the config fails validation. The message lists every  problem found. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/config.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `save`

```python
save(cfg, run_dir, warn_github_changes=True) → Path
```

Create ``run_dir`` and write ``config.yaml`` into it. 

A run named ``test`` may reuse an existing ``run_dir`` and does not record a git hash. Any other name raises if ``run_dir`` already exists, and records the current git hash as ``cfg.git_hash``. 



**Args:**
 
 - <b>`cfg`</b>:  Config returned by :func:`load`. Modified in place (``git_hash``). 
 - <b>`run_dir`</b>:  Run directory, normally :func:`run_dir` of ``cfg``. 
 - <b>`warn_github_changes`</b>:  Forwarded to the git-hash lookup. 



**Returns:**
 Path to the written ``config.yaml``. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/config.py#L98"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `algorithm`

```python
algorithm(cfg) → str
```

``'morlax'`` or ``'amor'``. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/config.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `env_name`

```python
env_name(cfg) → str
```

Environment class name, e.g. ``'MOCheetah'``. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/config.py#L108"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `backend`

```python
backend(cfg, for_training=False) → str
```

``'jnp'`` when building an env for training, else the configured backend. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/config.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `env_params`

```python
env_params(cfg) → ConfigDict
```

The ``env_config`` section, given to env classes as ``env_params``. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/config.py#L118"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gaitlib_path`

```python
gaitlib_path(cfg)
```

Gait library path (NaviGait only), or ``None``. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/config.py#L123"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `ppo_params`

```python
ppo_params(cfg) → dict
```

PPO settings shared by both algorithms. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/config.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `sampling_params`

```python
sampling_params(cfg) → dict
```

Preference-sampling settings: ``alpha``, ``k``, ``sampling``, ``warmup_frac``. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/config.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `network_params`

```python
network_params(cfg) → dict
```

Network settings for the configured algorithm. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/config.py#L138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `normalize_observations`

```python
normalize_observations(cfg) → bool
```

Whether training keeps a running observation normalizer. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/config.py#L143"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `objectives`

```python
objectives(cfg) → list
```

Reward keys per objective, one list per reward dimension. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/config.py#L148"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `num_objectives`

```python
num_objectives(cfg) → int
```

Number of objectives (length of the reward vector). 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/config.py#L153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `objective_labels`

```python
objective_labels(cfg)
```

Display names for the objectives, or ``None`` if not set. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/config.py#L159"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `run_name`

```python
run_name(cfg) → str
```

Run name. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/config.py#L164"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `run_dir`

```python
run_dir(cfg) → Path
```

Run directory: ``save_dir / name``. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/config.py#L169"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_same_env`

```python
check_same_env(cfgs)
```

Raise ``ValueError`` unless every config deploys the same environment. 

Compares ``env`` and the full ``env_config`` section. Use this before evaluating several runs together (e.g. ``eval.pareto.get_morlax_fronts`` with a list of run directories). 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/config.py#L197"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `set_name`

```python
set_name(cfg, value)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/config.py#L201"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `set_hypertype`

```python
set_hypertype(cfg, value)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/config.py#L205"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `set_sampling`

```python
set_sampling(cfg, value)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/config.py#L209"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `set_k`

```python
set_k(cfg, value)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/config.py#L213"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `apply_overrides`

```python
apply_overrides(cfg, overrides, where='overrides')
```

Write each value in a nested dict into ``cfg`` at the same key path. 

``overrides`` mirrors the config layout, e.g. ``{'learning_params': {'ppo_params': {'learning_rate': 3e-4}}}``. A wandb sweep with nested ``parameters`` gives its sampled values in this form. 



**Args:**
 
 - <b>`cfg`</b>:  Config returned by :func:`load`. Modified in place. 
 - <b>`overrides`</b>:  Nested dict of values to write. 
 - <b>`where`</b>:  Name for ``overrides`` in error messages. 



**Raises:**
 
 - <b>`ValueError`</b>:  If a key path in ``overrides`` does not exist in ``cfg``  (e.g. a typo in a sweep file), or if the result fails validation. 


