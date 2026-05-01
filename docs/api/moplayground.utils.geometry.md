---
layout: default
title: "moplayground.utils.geometry"
parent: API Reference
---

<!-- markdownlint-disable -->

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/geometry.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `moplayground.utils.geometry`




**Global Variables**
---------------
- **FREE3D_POS**
- **FREE3D_VEL**

---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/geometry.py#L4"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `euler2quat`

```python
euler2quat(_np, euler_angles)
```

Convert euler angles to quaternion representation 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/geometry.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `quat2euler`

```python
quat2euler(_np, q)
```

Convert quaternion to euler angles 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/geometry.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rotx`

```python
rotx(_np, theta)
```

Rotation matrix around x axis 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/geometry.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `roty`

```python
roty(_np, theta)
```

Rotation matrix around y axis 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/geometry.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rotz`

```python
rotz(_np, theta)
```

Rotation matrix around z axis 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/geometry.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rotmat`

```python
rotmat(_np, angles)
```

Rotation matrix from euler angles 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/geometry.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `quat_mul`

```python
quat_mul(_np, u, v)
```

Multiplies two quaternions. 



**Args:**
 
 - <b>`u`</b>:  (4,) quaternion (w,x,y,z) 
 - <b>`v`</b>:  (4,) quaternion (w,x,y,z) 



**Returns:**
 A quaternion u * v. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/geometry.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `angle2quat`

```python
angle2quat(_np, axis, angle)
```

Provides a quaternion that describes rotating around axis by angle. 



**Args:**
 
 - <b>`axis`</b>:  (3,) axis (x,y,z) 
 - <b>`angle`</b>:  () float angle to rotate by 



**Returns:**
 A quaternion that rotates around axis by angle 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/geometry.py#L104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `quat_conjugate`

```python
quat_conjugate(_np, q)
```

Returns the conjugate of a quaternion. 



**Args:**
 
 - <b>`q`</b>:  (4,) quaternion (w,x,y,z) 



**Returns:**
 A quaternion that is the conjugate of q. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/geometry.py#L116"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `extract_yaw`

```python
extract_yaw(_np, quat)
```

Extracts the yaw angle from a quaternion. 



**Args:**
 
 - <b>`quat`</b>:  (4,) quaternion (w,x,y,z) 



**Returns:**
 The yaw angle in radians. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/geometry.py#L139"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `solve_transform`

```python
solve_transform(_np, qpos_des, qpos_act, reset_yaw=False, cmd_yaw_offset=None)
```

Solves the transformation, T, between two qpos for the form qpos_des = qpos_act @ T 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/geometry.py#L182"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `apply_transform`

```python
apply_transform(_np, qpos, offset)
```

Applies the offset transform to the qpos 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/geometry.py#L195"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `quat_rotate_vector`

```python
quat_rotate_vector(np, q, v)
```

Rotate vector v by quaternion q. q: array-like, shape (4,) [w, x, y, z] v: array-like, shape (3,) Returns rotated vector as np.ndarray, shape (3,) 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/geometry.py#L216"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `inv_transform`

```python
inv_transform(_np, offset)
```

Computes the inverse of the offset transform to the qpos 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/geometry.py#L223"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `quat_dist`

```python
quat_dist(_np, q1, q2)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/geometry.py#L231"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `quat_rotate`

```python
quat_rotate(_np, q, v)
```

Rotate vector v (3,) using unit quaternion q (w,x,y,z). 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/utils/geometry.py#L236"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `decide_quat`

```python
decide_quat(_np, quat)
```






