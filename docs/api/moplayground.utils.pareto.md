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

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/pareto.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `project_to_simplex`

```python
project_to_simplex(v)
```

Euclidean projection of ``v`` onto the probability simplex.  




---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/pareto.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `closest_points`

```python
closest_points(points, targets)
```

Index into ``points`` of the nearest (Euclidean) neighbor of each target. 



**Args:**
 
 - <b>`points`</b>:  ``(n_points, dim)`` array searched over. 
 - <b>`targets`</b>:  ``(dim,)`` or ``(n_targets, dim)`` query point(s). 



**Returns:**
 ``(n_targets,)`` integer indices into ``points``. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/pareto.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `corner_tradeoffs`

```python
corner_tradeoffs(tradeoffs)
```

Indices of the tradeoffs nearest each one-hot corner of the simplex.  




---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/pareto.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_nondominated`

```python
get_nondominated(F, epsilon=None)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/pareto.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `hypervolume_from_nondominated`

```python
hypervolume_from_nondominated(F_min, ref_point=None)
```

Compute the hypervolume of a non-dominated front in minimization space. 



**Args:**
 
 - <b>`F_min`</b>:  ``(n_points, n_objectives)`` array of points in  minimization space (i.e. negated objective values). 
 - <b>`ref_point`</b>:  ``(n_objectives,)`` upper bound in that same minimization  space; a point not strictly below it on every axis contributes  nothing. Defaults to the origin. 



**Returns:**
 Hypervolume as a float, computed by ``pymoo.indicators.hv.HV``. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/pareto.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `sparsity_from_normalized_nondominated`

```python
sparsity_from_normalized_nondominated(F_min_norm)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/pareto.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_pareto_statistics`

```python
get_pareto_statistics(F, ref_point=None)
```

Compute hypervolume and sparsity for a set of objective vectors. 



**Args:**
 
 - <b>`F`</b>:  ``(n_points, n_objectives)`` array of objective vectors  (higher is better). 
 - <b>`ref_point`</b>:  ``(n_objectives,)`` reference in the same maximization space  as ``F``; an objective at or below it contributes no hypervolume.  Defaults to the origin. 



**Returns:**
 Tuple ``(hypervolume, sparsity)``. 


