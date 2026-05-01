---
layout: default
title: "moplayground.moppo.networks"
parent: API Reference
---

<!-- markdownlint-disable -->

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/networks.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `moplayground.moppo.networks`





---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/networks.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `flatten_model`

```python
flatten_model(target_params)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/networks.py#L410"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `count_params`

```python
count_params(params)
```

Count the total number of parameters in a Flax model. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/networks.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MLP`
MLP(obs_dim: int, hidden_sizes: Sequence[int], action_dim: int, parent: Union[flax.linen.module.Module, flax.core.scope.Scope, flax.linen.module._Sentinel, NoneType] = <flax.linen.module._Sentinel object at 0x7f5cfe3d6a20>, name: Optional[str] = None) 

<a href="https://github.com/dynamicmobility/moplayground/blob/main/moplayground/moppo/networks/__init__"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MLP.__init__`

```python
__init__(
    obs_dim: int,
    hidden_sizes: Sequence[int],
    action_dim: int,
    parent: Optional[flax.linen.module.Module, flax.core.scope.Scope, flax.linen.module._Sentinel] = <flax.linen.module._Sentinel object at 0x7f5cfe3d6a20>,
    name: Optional[str] = None
) → None
```






---

#### <kbd>property</kbd> MLP.path

Get the path of this Module. Top-level root modules have an empty path ``()``. Note that this method can only be used on bound modules that have a valid scope. 

Example usage:
``` 

   >>> import flax.linen as nn    >>> import jax, jax.numpy as jnp 

   >>> class SubModel(nn.Module):    ...   @nn.compact    ...   def __call__(self, x):    ...     print(f'SubModel path: {self.path}')    ...     return x 

   >>> class Model(nn.Module):    ...   @nn.compact    ...   def __call__(self, x):    ...     print(f'Model path: {self.path}')    ...     return SubModel()(x) 

   >>> model = Model()    >>> variables = model.init(jax.random.key(0), jnp.ones((1, 2)))    Model path: ()    SubModel path: ('SubModel_0',) 

---

#### <kbd>property</kbd> MLP.variables

Returns the variables in this module. 




---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/networks.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Hypernet`
Hypernet(target_model_dict: flax.linen.module.Module, num_objs: int, obs_dim: int, hypersize: tuple, num_features: int = 8, W_variance: float = 0.0, parent: Union[flax.linen.module.Module, flax.core.scope.Scope, flax.linen.module._Sentinel, NoneType] = <flax.linen.module._Sentinel object at 0x7f5cfe3d6a20>, name: Optional[str] = None) 

<a href="https://github.com/dynamicmobility/moplayground/blob/main/moplayground/moppo/networks/__init__"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Hypernet.__init__`

```python
__init__(
    target_model_dict: flax.linen.module.Module,
    num_objs: int,
    obs_dim: int,
    hypersize: tuple,
    num_features: int = 8,
    W_variance: float = 0.0,
    parent: Optional[flax.linen.module.Module, flax.core.scope.Scope, flax.linen.module._Sentinel] = <flax.linen.module._Sentinel object at 0x7f5cfe3d6a20>,
    name: Optional[str] = None
) → None
```






---

#### <kbd>property</kbd> Hypernet.path

Get the path of this Module. Top-level root modules have an empty path ``()``. Note that this method can only be used on bound modules that have a valid scope. 

Example usage:
``` 

   >>> import flax.linen as nn    >>> import jax, jax.numpy as jnp 

   >>> class SubModel(nn.Module):    ...   @nn.compact    ...   def __call__(self, x):    ...     print(f'SubModel path: {self.path}')    ...     return x 

   >>> class Model(nn.Module):    ...   @nn.compact    ...   def __call__(self, x):    ...     print(f'Model path: {self.path}')    ...     return SubModel()(x) 

   >>> model = Model()    >>> variables = model.init(jax.random.key(0), jnp.ones((1, 2)))    Model path: ()    SubModel path: ('SubModel_0',) 

---

#### <kbd>property</kbd> Hypernet.variables

Returns the variables in this module. 



---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/moplayground/moppo/networks/setup#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Hypernet.setup`

```python
setup()
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/networks.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `HypernetMLP`
HypernetMLP(target_model_dict: dict, num_objs: int, obs_dim: int, hypersize: tuple, parent: Union[flax.linen.module.Module, flax.core.scope.Scope, flax.linen.module._Sentinel, NoneType] = <flax.linen.module._Sentinel object at 0x7f5cfe3d6a20>, name: Optional[str] = None) 

<a href="https://github.com/dynamicmobility/moplayground/blob/main/moplayground/moppo/networks/__init__"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `HypernetMLP.__init__`

```python
__init__(
    target_model_dict: dict,
    num_objs: int,
    obs_dim: int,
    hypersize: tuple,
    parent: Optional[flax.linen.module.Module, flax.core.scope.Scope, flax.linen.module._Sentinel] = <flax.linen.module._Sentinel object at 0x7f5cfe3d6a20>,
    name: Optional[str] = None
) → None
```






---

#### <kbd>property</kbd> HypernetMLP.path

Get the path of this Module. Top-level root modules have an empty path ``()``. Note that this method can only be used on bound modules that have a valid scope. 

Example usage:
``` 

   >>> import flax.linen as nn    >>> import jax, jax.numpy as jnp 

   >>> class SubModel(nn.Module):    ...   @nn.compact    ...   def __call__(self, x):    ...     print(f'SubModel path: {self.path}')    ...     return x 

   >>> class Model(nn.Module):    ...   @nn.compact    ...   def __call__(self, x):    ...     print(f'Model path: {self.path}')    ...     return SubModel()(x) 

   >>> model = Model()    >>> variables = model.init(jax.random.key(0), jnp.ones((1, 2)))    Model path: ()    SubModel path: ('SubModel_0',) 

---

#### <kbd>property</kbd> HypernetMLP.variables

Returns the variables in this module. 



---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/moplayground/moppo/networks/setup#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `HypernetMLP.setup`

```python
setup()
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/networks.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FakeHypernet`
FakeHypernet(target_model_dict: dict, num_objs: int, obs_dim: int, hypersize: tuple, parent: Union[flax.linen.module.Module, flax.core.scope.Scope, flax.linen.module._Sentinel, NoneType] = <flax.linen.module._Sentinel object at 0x7f5cfe3d6a20>, name: Optional[str] = None) 

<a href="https://github.com/dynamicmobility/moplayground/blob/main/moplayground/moppo/networks/__init__"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FakeHypernet.__init__`

```python
__init__(
    target_model_dict: dict,
    num_objs: int,
    obs_dim: int,
    hypersize: tuple,
    parent: Optional[flax.linen.module.Module, flax.core.scope.Scope, flax.linen.module._Sentinel] = <flax.linen.module._Sentinel object at 0x7f5cfe3d6a20>,
    name: Optional[str] = None
) → None
```






---

#### <kbd>property</kbd> FakeHypernet.path

Get the path of this Module. Top-level root modules have an empty path ``()``. Note that this method can only be used on bound modules that have a valid scope. 

Example usage:
``` 

   >>> import flax.linen as nn    >>> import jax, jax.numpy as jnp 

   >>> class SubModel(nn.Module):    ...   @nn.compact    ...   def __call__(self, x):    ...     print(f'SubModel path: {self.path}')    ...     return x 

   >>> class Model(nn.Module):    ...   @nn.compact    ...   def __call__(self, x):    ...     print(f'Model path: {self.path}')    ...     return SubModel()(x) 

   >>> model = Model()    >>> variables = model.init(jax.random.key(0), jnp.ones((1, 2)))    Model path: ()    SubModel path: ('SubModel_0',) 

---

#### <kbd>property</kbd> FakeHypernet.variables

Returns the variables in this module. 



---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/moplayground/moppo/networks/setup#L179"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FakeHypernet.setup`

```python
setup()
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/networks.py#L222"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ActorCriticHypernet`
ActorCriticHypernet(target_policy_dict: dict, target_value_dict: dict, num_objs: int, obs_dim: int, hypersize: tuple = (128, 128), num_features: int = 8, W_variance: float = 0.0, parent: Union[flax.linen.module.Module, flax.core.scope.Scope, flax.linen.module._Sentinel, NoneType] = <flax.linen.module._Sentinel object at 0x7f5cfe3d6a20>, name: Optional[str] = None) 

<a href="https://github.com/dynamicmobility/moplayground/blob/main/moplayground/moppo/networks/__init__"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ActorCriticHypernet.__init__`

```python
__init__(
    target_policy_dict: dict,
    target_value_dict: dict,
    num_objs: int,
    obs_dim: int,
    hypersize: tuple = (128, 128),
    num_features: int = 8,
    W_variance: float = 0.0,
    parent: Optional[flax.linen.module.Module, flax.core.scope.Scope, flax.linen.module._Sentinel] = <flax.linen.module._Sentinel object at 0x7f5cfe3d6a20>,
    name: Optional[str] = None
) → None
```






---

#### <kbd>property</kbd> ActorCriticHypernet.path

Get the path of this Module. Top-level root modules have an empty path ``()``. Note that this method can only be used on bound modules that have a valid scope. 

Example usage:
``` 

   >>> import flax.linen as nn    >>> import jax, jax.numpy as jnp 

   >>> class SubModel(nn.Module):    ...   @nn.compact    ...   def __call__(self, x):    ...     print(f'SubModel path: {self.path}')    ...     return x 

   >>> class Model(nn.Module):    ...   @nn.compact    ...   def __call__(self, x):    ...     print(f'Model path: {self.path}')    ...     return SubModel()(x) 

   >>> model = Model()    >>> variables = model.init(jax.random.key(0), jnp.ones((1, 2)))    Model path: ()    SubModel path: ('SubModel_0',) 

---

#### <kbd>property</kbd> ActorCriticHypernet.variables

Returns the variables in this module. 



---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/moplayground/moppo/networks/setup#L231"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ActorCriticHypernet.setup`

```python
setup()
```





---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/moplayground/moppo/networks/unflatten_params#L269"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ActorCriticHypernet.unflatten_params`

```python
unflatten_params(flat_params, target_network, single)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/networks.py#L307"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DualA2CHypernet`
DualA2CHypernet(target_policy_dict: dict, target_value_dict: dict, num_objs: int, obs_dim: int, hypersize: tuple = (128, 128), num_features: int = 8, W_variance: float = 0.0, parent: Union[flax.linen.module.Module, flax.core.scope.Scope, flax.linen.module._Sentinel, NoneType] = <flax.linen.module._Sentinel object at 0x7f5cfe3d6a20>, name: Optional[str] = None) 

<a href="https://github.com/dynamicmobility/moplayground/blob/main/moplayground/moppo/networks/__init__"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DualA2CHypernet.__init__`

```python
__init__(
    target_policy_dict: dict,
    target_value_dict: dict,
    num_objs: int,
    obs_dim: int,
    hypersize: tuple = (128, 128),
    num_features: int = 8,
    W_variance: float = 0.0,
    parent: Optional[flax.linen.module.Module, flax.core.scope.Scope, flax.linen.module._Sentinel] = <flax.linen.module._Sentinel object at 0x7f5cfe3d6a20>,
    name: Optional[str] = None
) → None
```






---

#### <kbd>property</kbd> DualA2CHypernet.path

Get the path of this Module. Top-level root modules have an empty path ``()``. Note that this method can only be used on bound modules that have a valid scope. 

Example usage:
``` 

   >>> import flax.linen as nn    >>> import jax, jax.numpy as jnp 

   >>> class SubModel(nn.Module):    ...   @nn.compact    ...   def __call__(self, x):    ...     print(f'SubModel path: {self.path}')    ...     return x 

   >>> class Model(nn.Module):    ...   @nn.compact    ...   def __call__(self, x):    ...     print(f'Model path: {self.path}')    ...     return SubModel()(x) 

   >>> model = Model()    >>> variables = model.init(jax.random.key(0), jnp.ones((1, 2)))    Model path: ()    SubModel path: ('SubModel_0',) 

---

#### <kbd>property</kbd> DualA2CHypernet.variables

Returns the variables in this module. 



---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/moplayground/moppo/networks/setup#L316"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DualA2CHypernet.setup`

```python
setup()
```





---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/moplayground/moppo/networks/unflatten_params#L371"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DualA2CHypernet.unflatten_params`

```python
unflatten_params(flat_params, target_network, single)
```






