"""
rotations.py — Thin wrapper around scipy.spatial.transform.Rotation

No custom Rodrigues, no custom Exp/Log. scipy handles everything,
including batched operations and edge cases.

Representations:
    'rotvec'  : axis-angle ∈ ℝ³  (= Lie algebra of SO(3))
    'euler'   : xyz Euler angles ∈ ℝ³
    'quat'    : quaternion ∈ ℝ⁴ (scalar-last, scipy convention)
"""

import numpy as np
from scipy.spatial.transform import Rotation

REPR_DIM = {'rotvec': 3, 'euler': 3, 'quat': 4}


def random_rotation(rng: np.random.Generator = None) -> Rotation:
    """Uniform random rotation."""
    if rng is None:
        rng = np.random.default_rng()
    return Rotation.random(random_state=rng)


def rotmat_to_repr(R: np.ndarray, repr_type: str) -> np.ndarray:
    """Rotation matrix (3,3) or (B,3,3) → chosen representation vector."""
    r = Rotation.from_matrix(R)
    if repr_type == 'rotvec':
        return r.as_rotvec().astype(np.float32)
    elif repr_type == 'euler':
        return r.as_euler('xyz').astype(np.float32)
    elif repr_type == 'quat':
        return r.as_quat().astype(np.float32)  # scalar-last
    raise ValueError(f"Unknown repr: {repr_type}")


def repr_to_rotation(vec: np.ndarray, repr_type: str) -> Rotation:
    """Representation vector (dim,) or (B, dim) → scipy Rotation."""
    if repr_type == 'rotvec':
        return Rotation.from_rotvec(vec)
    elif repr_type == 'euler':
        return Rotation.from_euler('xyz', vec)
    elif repr_type == 'quat':
        return Rotation.from_quat(vec)
    raise ValueError(f"Unknown repr: {repr_type}")


def geodesic_distance_batch(a: np.ndarray, b: np.ndarray, repr_type: str) -> np.ndarray:
    """Batch geodesic distance between two sets of orientations.

    Args:
        a, b: orientation vectors, shape (B, dim) or (dim,)
        repr_type: 'rotvec', 'euler', or 'quat'

    Returns:
        distances in radians, shape (B,) or scalar
    """
    r_a = repr_to_rotation(a, repr_type)
    r_b = repr_to_rotation(b, repr_type)
    # r_a.inv() * r_b = relative rotation, magnitude() = geodesic distance
    return (r_a.inv() * r_b).magnitude().astype(np.float32)
