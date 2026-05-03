---
layout: default
title: "moplayground.utils.plotting"
parent: "moplayground.utils"
grand_parent: API Reference
---

<!-- markdownlint-disable -->

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/plotting.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `moplayground.utils.plotting`





---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/plotting.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_pareto`

```python
plot_pareto(
    ax,
    pareto: numpy.ndarray,
    directive: numpy.ndarray = None,
    objective: list[str] = None,
    nondominated=None,
    alpha=1.0
)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/plotting.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_sequential_paretos`

```python
plot_sequential_paretos(
    ax_titles: list[str],
    paretos: numpy.ndarray,
    directives: numpy.ndarray = None,
    objectives: list[str] = None
)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/plotting.py#L126"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_sequential_hypervolume`

```python
plot_sequential_hypervolume(
    iterations: list[int] | numpy.ndarray,
    paretos: numpy.ndarray
)
```






