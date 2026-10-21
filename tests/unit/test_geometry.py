import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import moplayground as mop

BACKENDS = pytest.mark.parametrize('_np', [np, jnp], ids=['np', 'jnp'])
ANGLES = [np.array([0.1, -0.2, 0.3]), np.array([0.0, 0.0, 1.2]), np.array([-0.5, 0.4, -2.0])]


def wxyz(rotation):
    q = np.roll(rotation.as_quat(), 1)
    return q if q[0] >= 0 else -q


def ref_quat(euler):
    """(w, x, y, z) for roll/pitch/yaw composed as R = Rx @ Ry @ Rz (the ``euler2quat`` convention)."""
    return wxyz(Rotation.from_euler('XYZ', euler))  # intrinsic XYZ == Rx Ry Rz


@BACKENDS
@pytest.mark.parametrize('euler', ANGLES)
def test_quat2euler(_np, euler):
    np.testing.assert_allclose(mop.quat2euler(_np, _np.array(ref_quat(euler))), euler, atol=1e-5)


@BACKENDS
@pytest.mark.parametrize('euler', ANGLES)
def test_euler2quat(_np, euler):
    q = mop.euler2quat(_np, _np.array(euler))
    np.testing.assert_allclose(q, ref_quat(euler), atol=1e-6)
    np.testing.assert_allclose(mop.quat2euler(_np, q), euler, atol=1e-5)


@BACKENDS
@pytest.mark.parametrize('euler', ANGLES)
def test_rotation_matrices(_np, euler):
    for rot in (mop.rotx, mop.roty, mop.rotz):
        R = np.asarray(rot(_np, euler[0]))
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(R), 1.0)
    R = np.asarray(mop.rotmat(_np, _np.array(euler)))
    np.testing.assert_allclose(R, Rotation.from_euler('XYZ', euler).as_matrix(), atol=1e-6)


@BACKENDS
@pytest.mark.parametrize('euler', ANGLES)
def test_quat_rotation_matches_rotmat(_np, euler):
    q = _np.array(ref_quat(euler))
    v = _np.array([0.3, -1.0, 2.0])
    expected = np.asarray(mop.rotmat(_np, _np.array(euler))) @ np.asarray(v)
    np.testing.assert_allclose(mop.quat_rotate(_np, q, v), expected, atol=1e-5)
    np.testing.assert_allclose(mop.quat_rotate_vector(_np, q, v), expected, atol=1e-5)


@BACKENDS
def test_quat_conjugate_is_inverse(_np):
    q = _np.array(ref_quat(ANGLES[0]))
    ident = mop.quat_mul(_np, q, mop.quat_conjugate(_np, q))
    np.testing.assert_allclose(np.abs(ident), [1, 0, 0, 0], atol=1e-6)


@BACKENDS
def test_angle2quat_and_yaw(_np):
    q = mop.angle2quat(_np, _np.array([0.0, 0.0, 1.0]), 0.7)
    np.testing.assert_allclose(mop.quat2euler(_np, q), [0, 0, 0.7], atol=1e-6)
    # extract_yaw keeps the heading: yaw applied first (intrinsic z-y-x), roll/pitch dropped
    q_full = _np.array(wxyz(Rotation.from_euler('ZYX', [0.7, -0.1, 0.2])))
    np.testing.assert_allclose(mop.extract_yaw(_np, q_full), ref_quat([0, 0, 0.7]), atol=1e-5)


def test_quat_dist_zero_for_same():
    q = ref_quat(ANGLES[2])
    assert np.isclose(mop.quat_dist(np, q, q), 0.0, atol=1e-6)
