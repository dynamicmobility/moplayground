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

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/plotting.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_mo_progress`

```python
plot_mo_progress(
    num_steps: int,
    metrics: dict,
    training_data: moplayground.utils.plotting.MOTrainingPlottingInfo,
    save_dir: pathlib.Path,
    run: wandb.sdk.wandb_run.Run = None
)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/plotting.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `default_coloring`

```python
default_coloring(tradeoff)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/plotting.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_pareto`

```python
plot_pareto(
    ax: matplotlib.axes._axes.Axes,
    pareto: numpy.ndarray,
    colors: str | numpy.ndarray = None,
    objective: list[str] = None,
    connect: bool = False,
    show_dominated: bool = True,
    nondominated_alpha: float = 1.0,
    dominated_alpha: float = 1.0,
    nondominated_s: int = 20,
    dominated_s: int = 8,
    outline_nondominated: float = 0.0,
    label: str = None,
    set_lims: bool = True,
    label_fontsize: int = 16,
    special_idxs: numpy.ndarray = None,
    special_marker: str = '*',
    connect_line_color: str | numpy.ndarray = 'black',
    **plot_kwargs
)
```

Plot a pareto frontier. 

``colors`` is a single matplotlib color or per-point color. RGB(A) ``label`` names the front in the legend;  ``set_lims`` zooms to the nondominated front. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/plotting.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/plotting.py#L235"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_sequential_hypervolume`

```python
plot_sequential_hypervolume(
    iterations: list[int] | numpy.ndarray,
    paretos: numpy.ndarray
)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/plotting.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MOTrainingPlottingInfo`
MOTrainingPlottingInfo(start_time: float, times: list = <factory>, iterations: list = <factory>, paretos: list = <factory>, directives: list = <factory>, labels: list = <factory>, hypervolumes: list = <factory>, sparsities: list = <factory>) 

<a href="https://github.com/dynamicmobility/moplayground/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MOTrainingPlottingInfo.__init__`

```python
__init__(
    start_time: float,
    times: list = <factory>,
    iterations: list = <factory>,
    paretos: list = <factory>,
    directives: list = <factory>,
    labels: list = <factory>,
    hypervolumes: list = <factory>,
    sparsities: list = <factory>
) → None
```








---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/plotting.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MOTrainingPlottingInfo.save`

```python
save(save_dir, create_time=True)
```






