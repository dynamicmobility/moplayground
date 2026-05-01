---
layout: default
title: "moplayground.eval.rollout"
parent: API Reference
---

<!-- markdownlint-disable -->

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/eval/rollout.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `moplayground.eval.rollout`





---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/eval/rollout.py#L6"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

Roll out a trained policy on ``env`` and return rendered frames. 

Loads the policy specified by ``config`` (single-objective if ``config.mo2so.enabled`` else multi-objective with the given ``tradeoff``), JIT-compiles the inference function, and delegates the actual stepping/rendering to ``minimal_mjx.eval.rollout_policy``. 



**Args:**
 
 - <b>`env`</b>:  The environment to roll out in (typically built with  ``create_environment``). 
 - <b>`config`</b>:  Run config (ConfigDict) for the trained policy. 
 - <b>`tradeoff`</b>:  Per-objective weighting used to condition the  multi-objective policy. Defaults to all-ones with length equal  to the number of objectives in ``config``. Ignored when  ``config.mo2so.enabled``. 
 - <b>`T`</b>:  Rollout duration in seconds. 
 - <b>`camera`</b>:  MuJoCo camera name to render from. 
 - <b>`width`</b>:  Frame width in pixels. 
 - <b>`height`</b>:  Frame height in pixels. 



**Returns:**
 Whatever ``minimal_mjx.eval.rollout_policy`` returns — at the time of writing a 4-tuple ``(frames, reward_plotter, ...)``. 


