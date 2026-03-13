FREE3D_POS = 7 # number of dims a free joint takes (position x,y,z) (quat x,y,z,w)
FREE3D_VEL = 6 # number of dims a free joint takes (position x,y,z) (quat x,y,z,w)

def euler2quat(_np, euler_angles):
    """Convert euler angles to quaternion representation"""
    roll = euler_angles[0]
    pitch = euler_angles[1]
    yaw = euler_angles[2]

    cy = _np.cos(yaw * 0.5)
    sy = _np.sin(yaw * 0.5)
    cp = _np.cos(pitch * 0.5)
    sp = _np.sin(pitch * 0.5)
    cr = _np.cos(roll * 0.5)
    sr = _np.sin(roll * 0.5)

    w = cr * cp * cy - sr * sp * sy
    x = sr * cp * cy + cr * sp * sy
    y = cr * sp * cy - sr * cp * sy
    z = cr * cp * sy + sr * sp * cy
    q = _np.array([w, x, y, z]) 
    q = decide_quat(_np, q)
    return q
    
def quat2euler(_np, q):
    """Convert quaternion to euler angles"""
    w, x, y, z = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = _np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = _np.clip(t2, -1.0, 1.0)
    Y =  _np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z =  _np.arctan2(t3, t4)

    return _np.array([X, Y, Z])

def rotx(_np, theta):
    """Rotation matrix around x axis"""
    c = _np.cos(theta)
    s = _np.sin(theta)
    return _np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])

def roty(_np, theta):
    """Rotation matrix around y axis"""
    c = _np.cos(theta)
    s = _np.sin(theta)
    return _np.array([[c, 0, s],
                      [0, 1, 0],
                      [-s, 0, c]])

def rotz(_np, theta):
    """Rotation matrix around z axis"""
    c = _np.cos(theta)
    s = _np.sin(theta)
    return _np.array([[c, -s, 0],
                      [s, c, 0],
                      [0, 0, 1]])
        

def rotmat(_np, angles):
    """Rotation matrix from euler angles"""
    return rotz(_np, angles[2]) @ roty(_np, angles[1]) @ rotx(_np, angles[0])

def quat_mul(_np, u, v):
    """Multiplies two quaternions.

    Args:
      u: (4,) quaternion (w,x,y,z)
      v: (4,) quaternion (w,x,y,z)

    Returns:
      A quaternion u * v.
    """
    return _np.array([
        u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
        u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
        u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
        u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
    ])


def angle2quat(_np, axis, angle):
    """Provides a quaternion that describes rotating around axis by angle.

    Args:
      axis: (3,) axis (x,y,z)
      angle: () float angle to rotate by

    Returns:
      A quaternion that rotates around axis by angle
    """
    s, c = _np.sin(angle * 0.5), _np.cos(angle * 0.5)
    quat = _np.insert(axis * s, 0, c)
    quat = decide_quat(_np, quat)
    return quat

def quat_conjugate(_np, q):
    """Returns the conjugate of a quaternion.

    Args:
      q: (4,) quaternion (w,x,y,z)

    Returns:
      A quaternion that is the conjugate of q.
    """
    conj = _np.array([q[0], -q[1], -q[2], -q[3]])
    return decide_quat(_np, conj)

def extract_yaw(_np, quat):
    """Extracts the yaw angle from a quaternion.

    Args:
      quat: (4,) quaternion (w,x,y,z)

    Returns:
      The yaw angle in radians.
    """
    w, x, y, z = quat
    yaw = _np.arctan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y**2 + z**2))
    
    # Reconstruct yaw-only quaternion
    half_yaw = yaw / 2.0
    cy = _np.cos(half_yaw)
    sy = _np.sin(half_yaw)

    # Pure yaw quaternion (rotation around z axis)
    q_yaw = _np.array([cy, 0.0, 0.0, sy])  # [w, x, y, z]
    q_yaw = decide_quat(_np, q_yaw)
    return q_yaw
    

def solve_transform(_np, qpos_des, qpos_act, reset_yaw=False, cmd_yaw_offset=0.0):
    """Solves the transformation, T, between two qpos for the form
    qpos_des = qpos_act @ T"""
    xyz_des  = qpos_des[:3]
    xyz_act  = qpos_act[:3]
    quat_act = qpos_act[3:FREE3D_POS]
    quat_des = qpos_des[3:FREE3D_POS]

    if reset_yaw:
        quat_diff = quat_mul(_np, quat_conjugate(_np, quat_des), quat_act)
        yaw_quat_act = extract_yaw(_np, quat_diff)
    else:
        yaw_quat_act = angle2quat(_np, _np.array([0.0, 0.0, 1.0]), cmd_yaw_offset)
        # quat_des = quat_mul(_np, yaw_quat_act, quat_des)
        # quat_diff = quat_mul(_np, quat_conjugate(_np, quat_des), quat_act)
        # yaw_quat_act = extract_yaw(_np, quat_diff)
        # yaw_quat_act = angle2quat(_np, _np.array([0.0, 0.0, 1.0]), cmd_yaw_offset)
        # yaw_quat_act = _np.array([1.0, 0.0, 0.0, 0.0])
    offset_quat = yaw_quat_act
    # R = quat2rot(_np, offset_quat)
    xyz_des_rot = quat_rotate_vector(_np, offset_quat, xyz_des)
    offset_xyz = _np.hstack((xyz_act[:2] - xyz_des_rot[:2], [0]))
    offset_quat = decide_quat(_np, offset_quat)
    return _np.hstack((offset_xyz, offset_quat))

# def solve_transform(_np, qpos_des, qpos_act, reset_yaw=False, cmd_yaw_offset=0.0):
#     """Solves the transformation, T, between two qpos for the form
#     qpos_des = qpos_act @ T"""
#     xyz_des  = qpos_des[:3]
#     xyz_act  = qpos_act[:3]
#     quat_act = qpos_act[3:FREE3D_POS]

#     yaw_quat_act = extract_yaw(_np, quat_act)
#     # yaw_quat_act = _np.array([1, 0, 0, 0]) #extract_yaw(_np, quat_act)
#     offset_quat = yaw_quat_act
#     # R = quat2rot(_np, offset_quat)
#     xyz_des_rot = quat_rotate_vector(_np, offset_quat, xyz_des)
#     offset_xyz = _np.hstack((xyz_act[:2] - xyz_des_rot[:2], [0]))
#     offset_quat = decide_quat(_np, offset_quat)
#     return _np.hstack((offset_xyz, offset_quat))


def apply_transform(_np, qpos, offset):
    """Applies the offset transform to the qpos"""
    # quit()
    # new_qpos = quat_rotate_vector(_np, quat_conjugate(_np, offset[3:FREE3D_POS]), qpos[:3])
    new_qpos = quat_rotate_vector(_np, offset[3:FREE3D_POS], qpos[:3])
    # new_xyz = new_qpos + offset[:3]
    # print(offset[:3])
    # input()
    new_xyz = new_qpos[:3] + offset[:3]
    new_quat = quat_mul(_np, offset[3:FREE3D_POS], qpos[3:FREE3D_POS])
    new_quat = decide_quat(_np, new_quat)
    return _np.hstack((new_xyz, new_quat))

def quat_rotate_vector(np, q, v):
    """
    Rotate vector v by quaternion q.
    q: array-like, shape (4,) [w, x, y, z]
    v: array-like, shape (3,)
    Returns rotated vector as np.ndarray, shape (3,)
    """
    q = np.array(q, dtype=float)
    v = np.array(v, dtype=float)

    # Normalize quaternion (optional but recommended)
    q = q / np.linalg.norm(q)

    w, x, y, z = q
    q_vec = np.array([x, y, z])

    # Quaternion * vector formula (optimized)
    uv = np.cross(q_vec, v)
    uuv = np.cross(q_vec, uv)
    return v + 2 * (w * uv + uuv)

def inv_transform(_np, offset):
    """Computes the inverse of the offset transform to the qpos"""
    new_xyz = -offset[:3]
    new_quat = quat_conjugate(_np, offset[3:FREE3D_POS])
    new_quat = decide_quat(_np, new_quat)
    return _np.hstack((new_xyz, new_quat))

def quat_dist(_np, q1, q2):
    q1 = _np.array(q1)
    q2 = _np.array(q2)
    dot = _np.abs(_np.dot(q1, q2))
    dot = _np.clip(dot, -1.0, 1.0)  # numerical safety
    angle = 2 * _np.arccos(dot)
    return angle  # in radians

def quat_rotate(_np, q, v):
    """Rotate vector v (3,) using unit quaternion q (w,x,y,z)."""
    q_v = _np.array([0, *v])
    return quat_mul(_np, quat_mul(_np, q, q_v), quat_conjugate(_np, q))[1:]

def decide_quat(_np, quat):
    cond = quat[0] >= 0
    return _np.where(cond, quat, -quat)