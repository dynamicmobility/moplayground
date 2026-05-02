---
layout: default
title: "moplayground.utils.pareto"
parent: "moplayground.utils"
grand_parent: API Reference
---

<!-- markdownlint-disable -->

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/pareto.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `moplayground.utils.pareto`





---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/pareto.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_nondominated`

```python
get_nondominated(F, epsilon=None)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/pareto.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `hypervolume_from_nondominated`

```python
hypervolume_from_nondominated(F_min)
```

Compute the hypervolume of a non-dominated front in minimization space. 

Uses the origin as the reference point, so callers must pass an already-non-dominated, already-negated front (see ``get_pareto_statistics`` for the typical pipeline). 



**Args:**
 
 - <b>`F_min`</b>:  ``(n_points, n_objectives)`` array of points in  minimization space (i.e. negated objective values). 



**Returns:**
 Hypervolume as a float, computed by ``pymoo.indicators.hv.HV``. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/pareto.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `sparsity_from_normalized_nondominated`

```python
sparsity_from_normalized_nondominated(F_min_norm)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/pareto.py#L37"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_pareto_statistics`

```python
get_pareto_statistics(F)
```

Compute hypervolume and sparsity for a set of objective vectors. 

Extracts the non-dominated front of ``F`` (treated as a maximization problem), normalizes it, converts to minimization space, and returns the resulting hypervolume and spacing-based sparsity. When the front contains only one point, it is duplicated so the spacing indicator has at least two points to work with. 



**Args:**
 
 - <b>`F`</b>:  ``(n_points, n_objectives)`` array of objective vectors  (higher is better). 



**Returns:**
 Tuple ``(hypervolume, sparsity)``. 


