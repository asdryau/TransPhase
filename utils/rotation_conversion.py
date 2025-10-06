# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Check PYTORCH3D_LICENCE before use

import functools
from typing import Optional

import torch
import torch.nn.functional as F


"""
The transformation matrices returned from the functions in this file assume
the points on which the transformation will be applied are column vectors.
i.e. the R matrix is structured as

    R = [
            [Rxx, Rxy, Rxz],
            [Ryx, Ryy, Ryz],
            [Rzx, Rzy, Rzz],
        ]  # (3, 3)

This matrix can be applied to column vectors by post multiplication
by the points e.g.

    points = [[0], [1], [2]]  # (3 x 1) xyz coordinates of a point
    transformed_points = R * points

To apply the same matrix to points which are row vectors, the R matrix
can be transposed and pre multiplied by the points:

e.g.
    points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
    transformed_points = points * R.transpose(1, 0)
"""


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _sqrt_positive_part(x):
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)


def _axis_angle_rotation(axis: str, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles, convention: str):
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
    return functools.reduce(torch.matmul, matrices)


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2


def matrix_to_euler_angles(matrix, convention: str):
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


def random_quaternions(
    n: int, dtype: Optional[torch.dtype] = None, device=None, requires_grad=False
):
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.
        requires_grad: Whether the resulting tensor should have the gradient
            flag set.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    o = torch.randn((n, 4), dtype=dtype, device=device, requires_grad=requires_grad)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(
    n: int, dtype: Optional[torch.dtype] = None, device=None, requires_grad=False
):
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.
        requires_grad: Whether the resulting tensor should have the gradient
            flag set.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(
        n, dtype=dtype, device=device, requires_grad=requires_grad
    )
    return quaternion_to_matrix(quaternions)


def random_rotation(
    dtype: Optional[torch.dtype] = None, device=None, requires_grad=False
):
    """
    Generate a single random 3x3 rotation matrix.

    Args:
        dtype: Type to return
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type
        requires_grad: Whether the resulting tensor should have the gradient
            flag set

    Returns:
        Rotation matrix as tensor of shape (3, 3).
    """
    return random_rotations(1, dtype, device, requires_grad)[0]


def standardize_quaternion(quaternions):
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a, b):
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(a, b):
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def quaternion_invert(quaternion):
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    return quaternion * quaternion.new_tensor([1, -1, -1, -1])


def quaternion_apply(quaternion, point):
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, f{point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix):
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)


#####################################################################################
class Quaternions:
    """
    Quaternions is a wrapper around a numpy ndarray
    that allows it to act as if it were an narray of
    a quaternion data type.
    
    Therefore addition, subtraction, multiplication,
    division, negation, absolute, are all defined
    in terms of quaternion operations such as quaternion
    multiplication.
    
    This allows for much neater code and many routines
    which conceptually do the same thing to be written
    in the same way for point data and for rotation data.
    
    The Quaternions class has been desgined such that it
    should support broadcasting and slicing in all of the
    usual ways.
    """
    
    def __init__(self, qs):
        if isinstance(qs, np.ndarray):
        
            if len(qs.shape) == 1: qs = np.array([qs])
            self.qs = qs
            return
            
        if isinstance(qs, Quaternions):
            self.qs = qs.qs
            return
            
        raise TypeError('Quaternions must be constructed from iterable, numpy array, or Quaternions, not %s' % type(qs))
    
    def __str__(self): return "Quaternions("+ str(self.qs) + ")"
    def __repr__(self): return "Quaternions("+ repr(self.qs) + ")"
    
    """ Helper Methods for Broadcasting and Data extraction """
    
    @classmethod
    def _broadcast(cls, sqs, oqs, scalar=False):
        
        if isinstance(oqs, float): return sqs, oqs * np.ones(sqs.shape[:-1])
        
        ss = np.array(sqs.shape) if not scalar else np.array(sqs.shape[:-1])
        os = np.array(oqs.shape)
        
        if len(ss) != len(os):
            raise TypeError('Quaternions cannot broadcast together shapes %s and %s' % (sqs.shape, oqs.shape))
            
        if np.all(ss == os): return sqs, oqs
        
        if not np.all((ss == os) | (os == np.ones(len(os))) | (ss == np.ones(len(ss)))):
            raise TypeError('Quaternions cannot broadcast together shapes %s and %s' % (sqs.shape, oqs.shape))
            
        sqsn, oqsn = sqs.copy(), oqs.copy()
        
        for a in np.where(ss == 1)[0]: sqsn = sqsn.repeat(os[a], axis=a)
        for a in np.where(os == 1)[0]: oqsn = oqsn.repeat(ss[a], axis=a)
        
        return sqsn, oqsn
        
    """ Adding Quaterions is just Defined as Multiplication """
    
    def __add__(self, other): return self * other
    def __sub__(self, other): return self / other
    
    """ Quaterion Multiplication """
    
    def __mul__(self, other):
        """
        Quaternion multiplication has three main methods.
        
        When multiplying a Quaternions array by Quaternions
        normal quaternion multiplication is performed.
        
        When multiplying a Quaternions array by a vector
        array of the same shape, where the last axis is 3,
        it is assumed to be a Quaternion by 3D-Vector 
        multiplication and the 3D-Vectors are rotated
        in space by the Quaternions.
        
        When multipplying a Quaternions array by a scalar
        or vector of different shape it is assumed to be
        a Quaternions by Scalars multiplication and the
        Quaternions are scaled using Slerp and the identity
        quaternions.
        """
        
        """ If Quaternions type do Quaternions * Quaternions """
        if isinstance(other, Quaternions):
            
            sqs, oqs = Quaternions._broadcast(self.qs, other.qs)
            
            q0 = sqs[...,0]; q1 = sqs[...,1]; 
            q2 = sqs[...,2]; q3 = sqs[...,3]; 
            r0 = oqs[...,0]; r1 = oqs[...,1]; 
            r2 = oqs[...,2]; r3 = oqs[...,3]; 
            
            qs = np.empty(sqs.shape)
            qs[...,0] = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
            qs[...,1] = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
            qs[...,2] = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
            qs[...,3] = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0
            
            return Quaternions(qs)
        
        """ If array type do Quaternions * Vectors """
        if isinstance(other, np.ndarray) and other.shape[-1] == 3:
            vs = Quaternions(np.concatenate([np.zeros(other.shape[:-1] + (1,)), other], axis=-1))
            return (self * (vs * -self)).imaginaries
        
        """ If float do Quaternions * Scalars """
        if isinstance(other, np.ndarray) or isinstance(other, float):
            return Quaternions.slerp(Quaternions.id_like(self), self, other)
        
        raise TypeError('Cannot multiply/add Quaternions with type %s' % str(type(other)))
        
    def __div__(self, other):
        """
        When a Quaternion type is supplied, division is defined
        as multiplication by the inverse of that Quaternion.
        
        When a scalar or vector is supplied it is defined
        as multiplicaion of one over the supplied value.
        Essentially a scaling.
        """
        
        if isinstance(other, Quaternions): return self * (-other)
        if isinstance(other, np.ndarray): return self * (1.0 / other)
        if isinstance(other, float): return self * (1.0 / other)
        raise TypeError('Cannot divide/subtract Quaternions with type %s' + str(type(other)))
        
    def __eq__(self, other): return self.qs == other.qs
    def __ne__(self, other): return self.qs != other.qs
    
    def __neg__(self):
        """ Invert Quaternions """
        return Quaternions(self.qs * np.array([[1, -1, -1, -1]]))
    
    def __abs__(self):
        """ Unify Quaternions To Single Pole """
        qabs = self.normalized().copy()
        top = np.sum(( qabs.qs) * np.array([1,0,0,0]), axis=-1)
        bot = np.sum((-qabs.qs) * np.array([1,0,0,0]), axis=-1)
        qabs.qs[top < bot] = -qabs.qs[top <  bot]
        return qabs
    
    def __iter__(self): return iter(self.qs)
    def __len__(self): return len(self.qs)
    
    def __getitem__(self, k):    return Quaternions(self.qs[k]) 
    def __setitem__(self, k, v): self.qs[k] = v.qs
        
    @property
    def lengths(self):
        return np.sum(self.qs**2.0, axis=-1)**0.5
    
    @property
    def reals(self):
        return self.qs[...,0]
        
    @property
    def imaginaries(self):
        return self.qs[...,1:4]
    
    @property
    def shape(self): return self.qs.shape[:-1]
    
    def repeat(self, n, **kwargs):
        return Quaternions(self.qs.repeat(n, **kwargs))
    
    def normalized(self):
        return Quaternions(self.qs / self.lengths[...,np.newaxis])
    
    def log(self):
        norm = abs(self.normalized())
        imgs = norm.imaginaries
        lens = np.sqrt(np.sum(imgs**2, axis=-1))
        lens = np.arctan2(lens, norm.reals) / (lens + 1e-10)
        return imgs * lens[...,np.newaxis]
    
    def constrained(self, axis):
        
        rl = self.reals
        im = np.sum(axis * self.imaginaries, axis=-1)
        
        t1 = -2 * np.arctan2(rl, im) + np.pi
        t2 = -2 * np.arctan2(rl, im) - np.pi
        
        top = Quaternions.exp(axis[np.newaxis] * (t1[:,np.newaxis] / 2.0))
        bot = Quaternions.exp(axis[np.newaxis] * (t2[:,np.newaxis] / 2.0))
        img = self.dot(top) > self.dot(bot)
        
        ret = top.copy()
        ret[ img] = top[ img]
        ret[~img] = bot[~img]
        return ret
    
    def constrained_x(self): return self.constrained(np.array([1,0,0]))
    def constrained_y(self): return self.constrained(np.array([0,1,0]))
    def constrained_z(self): return self.constrained(np.array([0,0,1]))
    
    def dot(self, q): return np.sum(self.qs * q.qs, axis=-1)
    
    def copy(self): return Quaternions(np.copy(self.qs))
    
    def reshape(self, s):
        self.qs.reshape(s)
        return self
    
    def interpolate(self, ws):
        return Quaternions.exp(np.average(abs(self).log, axis=0, weights=ws))
    
    def euler(self, order='xyz'):
        
        q = self.normalized().qs
        q0 = q[...,0]
        q1 = q[...,1]
        q2 = q[...,2]
        q3 = q[...,3]
        es = np.zeros(self.shape + (3,))
        
        if   order == 'xyz':
            es[...,0] = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            es[...,1] = np.arcsin((2 * (q0 * q2 - q3 * q1)).clip(-1,1))
            es[...,2] = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        elif order == 'yzx':
            es[...,0] = np.arctan2(2 * (q1 * q0 - q2 * q3), -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0)
            es[...,1] = np.arctan2(2 * (q2 * q0 - q1 * q3),  q1 * q1 - q2 * q2 - q3 * q3 + q0 * q0)
            es[...,2] = np.arcsin((2 * (q1 * q2 + q3 * q0)).clip(-1,1))
        else:
            raise NotImplementedError('Cannot convert from ordering %s' % order)
        
        """
        
        # These conversion don't appear to work correctly for Maya.
        # http://bediyap.com/programming/convert-quaternion-to-euler-rotations/
        
        if   order == 'xyz':
            es[fa + (0,)] = np.arctan2(2 * (q0 * q3 - q1 * q2), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
            es[fa + (1,)] = np.arcsin((2 * (q1 * q3 + q0 * q2)).clip(-1,1))
            es[fa + (2,)] = np.arctan2(2 * (q0 * q1 - q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
        elif order == 'yzx':
            es[fa + (0,)] = np.arctan2(2 * (q0 * q1 - q2 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3)
            es[fa + (1,)] = np.arcsin((2 * (q1 * q2 + q0 * q3)).clip(-1,1))
            es[fa + (2,)] = np.arctan2(2 * (q0 * q2 - q1 * q3), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
        elif order == 'zxy':
            es[fa + (0,)] = np.arctan2(2 * (q0 * q2 - q1 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
            es[fa + (1,)] = np.arcsin((2 * (q0 * q1 + q2 * q3)).clip(-1,1))
            es[fa + (2,)] = np.arctan2(2 * (q0 * q3 - q1 * q2), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3) 
        elif order == 'xzy':
            es[fa + (0,)] = np.arctan2(2 * (q0 * q2 + q1 * q3), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
            es[fa + (1,)] = np.arcsin((2 * (q0 * q3 - q1 * q2)).clip(-1,1))
            es[fa + (2,)] = np.arctan2(2 * (q0 * q1 + q2 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3)
        elif order == 'yxz':
            es[fa + (0,)] = np.arctan2(2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3)
            es[fa + (1,)] = np.arcsin((2 * (q0 * q1 - q2 * q3)).clip(-1,1))
            es[fa + (2,)] = np.arctan2(2 * (q1 * q3 + q0 * q2), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
        elif order == 'zyx':
            es[fa + (0,)] = np.arctan2(2 * (q0 * q1 + q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
            es[fa + (1,)] = np.arcsin((2 * (q0 * q2 - q1 * q3)).clip(-1,1))
            es[fa + (2,)] = np.arctan2(2 * (q0 * q3 + q1 * q2), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
        else:
            raise KeyError('Unknown ordering %s' % order)
        
        """
        
        # https://github.com/ehsan/ogre/blob/master/OgreMain/src/OgreMatrix3.cpp
        # Use this class and convert from matrix
        
        return es
        
    
    def average(self):
        
        if len(self.shape) == 1:
            
            import numpy.core.umath_tests as ut
            system = ut.matrix_multiply(self.qs[:,:,np.newaxis], self.qs[:,np.newaxis,:]).sum(axis=0)
            w, v = np.linalg.eigh(system)
            qiT_dot_qref = (self.qs[:,:,np.newaxis] * v[np.newaxis,:,:]).sum(axis=1)
            return Quaternions(v[:,np.argmin((1.-qiT_dot_qref**2).sum(axis=0))])            
        
        else:
            
            raise NotImplementedError('Cannot average multi-dimensionsal Quaternions')

    def angle_axis(self):
        
        norm = self.normalized()        
        s = np.sqrt(1 - (norm.reals**2.0))
        s[s == 0] = 0.001
        
        angles = 2.0 * np.arccos(norm.reals)
        axis = norm.imaginaries / s[...,np.newaxis]
        
        return angles, axis
        
    
    def transforms(self):
        
        qw = self.qs[...,0]
        qx = self.qs[...,1]
        qy = self.qs[...,2]
        qz = self.qs[...,3]
        
        x2 = qx + qx; y2 = qy + qy; z2 = qz + qz;
        xx = qx * x2; yy = qy * y2; wx = qw * x2;
        xy = qx * y2; yz = qy * z2; wy = qw * y2;
        xz = qx * z2; zz = qz * z2; wz = qw * z2;
        
        m = np.empty(self.shape + (3,3))
        m[...,0,0] = 1.0 - (yy + zz)
        m[...,0,1] = xy - wz
        m[...,0,2] = xz + wy        
        m[...,1,0] = xy + wz
        m[...,1,1] = 1.0 - (xx + zz)
        m[...,1,2] = yz - wx        
        m[...,2,0] = xz - wy
        m[...,2,1] = yz + wx
        m[...,2,2] = 1.0 - (xx + yy)
        
        return m
    
    def ravel(self):
        return self.qs.ravel()
    
    @classmethod
    def id(cls, n):
        
        if isinstance(n, tuple):
            qs = np.zeros(n + (4,))
            qs[...,0] = 1.0
            return Quaternions(qs)
        
        if isinstance(n, int) or isinstance(n, long):
            qs = np.zeros((n,4))
            qs[:,0] = 1.0
            return Quaternions(qs)
        
        raise TypeError('Cannot Construct Quaternion from %s type' % str(type(n)))

    @classmethod
    def id_like(cls, a):
        qs = np.zeros(a.shape + (4,))
        qs[...,0] = 1.0
        return Quaternions(qs)
        
    @classmethod
    def exp(cls, ws):
    
        ts = np.sum(ws**2.0, axis=-1)**0.5
        ts[ts == 0] = 0.001
        ls = np.sin(ts) / ts
        
        qs = np.empty(ws.shape[:-1] + (4,))
        qs[...,0] = np.cos(ts)
        qs[...,1] = ws[...,0] * ls
        qs[...,2] = ws[...,1] * ls
        qs[...,3] = ws[...,2] * ls
        
        return Quaternions(qs).normalized()
        
    @classmethod
    def slerp(cls, q0s, q1s, a):
        
        fst, snd = cls._broadcast(q0s.qs, q1s.qs)
        fst, a = cls._broadcast(fst, a, scalar=True)
        snd, a = cls._broadcast(snd, a, scalar=True)
        
        len = np.sum(fst * snd, axis=-1)
        
        neg = len < 0.0
        len[neg] = -len[neg]
        snd[neg] = -snd[neg]
        
        amount0 = np.zeros(a.shape)
        amount1 = np.zeros(a.shape)

        linear = (1.0 - len) < 0.01
        omegas = np.arccos(len[~linear])
        sinoms = np.sin(omegas)
        
        amount0[ linear] = 1.0 - a[linear]
        amount1[ linear] =       a[linear]
        amount0[~linear] = np.sin((1.0 - a[~linear]) * omegas) / sinoms
        amount1[~linear] = np.sin(       a[~linear]  * omegas) / sinoms
        
        return Quaternions(
            amount0[...,np.newaxis] * fst + 
            amount1[...,np.newaxis] * snd)
    
    @classmethod
    def between(cls, v0s, v1s):
        a = np.cross(v0s, v1s)
        w = np.sqrt((v0s**2).sum(axis=-1) * (v1s**2).sum(axis=-1)) + (v0s * v1s).sum(axis=-1)
        return Quaternions(np.concatenate([w[...,np.newaxis], a], axis=-1)).normalized()
    
    @classmethod
    def from_angle_axis(cls, angles, axis):
        axis    = axis / (np.sqrt(np.sum(axis**2, axis=-1)) + 1e-10)[...,np.newaxis]
        sines   = np.sin(angles / 2.0)[...,np.newaxis]
        cosines = np.cos(angles / 2.0)[...,np.newaxis]
        return Quaternions(np.concatenate([cosines, axis * sines], axis=-1))
    
    @classmethod
    def from_euler(cls, es, order='xyz', world=False):
    
        axis = {
            'x' : np.array([1,0,0]),
            'y' : np.array([0,1,0]),
            'z' : np.array([0,0,1]),
        }
        
        q0s = Quaternions.from_angle_axis(es[...,0], axis[order[0]])
        q1s = Quaternions.from_angle_axis(es[...,1], axis[order[1]])
        q2s = Quaternions.from_angle_axis(es[...,2], axis[order[2]])
        
        return (q2s * (q1s * q0s)) if world else (q0s * (q1s * q2s))
    
    @classmethod
    def from_transforms(cls, ts):
        
        d0, d1, d2 = ts[...,0,0], ts[...,1,1], ts[...,2,2]
        
        q0 = ( d0 + d1 + d2 + 1.0) / 4.0
        q1 = ( d0 - d1 - d2 + 1.0) / 4.0
        q2 = (-d0 + d1 - d2 + 1.0) / 4.0
        q3 = (-d0 - d1 + d2 + 1.0) / 4.0
        
        q0 = np.sqrt(q0.clip(0,None))
        q1 = np.sqrt(q1.clip(0,None))
        q2 = np.sqrt(q2.clip(0,None))
        q3 = np.sqrt(q3.clip(0,None))
        
        c0 = (q0 >= q1) & (q0 >= q2) & (q0 >= q3)
        c1 = (q1 >= q0) & (q1 >= q2) & (q1 >= q3)
        c2 = (q2 >= q0) & (q2 >= q1) & (q2 >= q3)
        c3 = (q3 >= q0) & (q3 >= q1) & (q3 >= q2)
        
        q1[c0] *= np.sign(ts[c0,2,1] - ts[c0,1,2])
        q2[c0] *= np.sign(ts[c0,0,2] - ts[c0,2,0])
        q3[c0] *= np.sign(ts[c0,1,0] - ts[c0,0,1])
        
        q0[c1] *= np.sign(ts[c1,2,1] - ts[c1,1,2])
        q2[c1] *= np.sign(ts[c1,1,0] + ts[c1,0,1])
        q3[c1] *= np.sign(ts[c1,0,2] + ts[c1,2,0])  
        
        q0[c2] *= np.sign(ts[c2,0,2] - ts[c2,2,0])
        q1[c2] *= np.sign(ts[c2,1,0] + ts[c2,0,1])
        q3[c2] *= np.sign(ts[c2,2,1] + ts[c2,1,2])  
        
        q0[c3] *= np.sign(ts[c3,1,0] - ts[c3,0,1])
        q1[c3] *= np.sign(ts[c3,2,0] + ts[c3,0,2])
        q2[c3] *= np.sign(ts[c3,2,1] + ts[c3,1,2])  
        
        qs = np.empty(ts.shape[:-2] + (4,))
        qs[...,0] = q0
        qs[...,1] = q1
        qs[...,2] = q2
        qs[...,3] = q3
        
        return cls(qs)

    def mat2quat(M):
        ''' Calculate quaternion corresponding to given rotation matrix

        Parameters
        ----------
        M : array-like
          3x3 rotation matrix

        Returns
        -------
        q : (4,) array
          closest quaternion to input matrix, having positive q[0]

        Notes
        -----
        Method claimed to be robust to numerical errors in M

        Constructs quaternion by calculating maximum eigenvector for matrix
        K (constructed from input `M`).  Although this is not tested, a
        maximum eigenvalue of 1 corresponds to a valid rotation.

        A quaternion q*-1 corresponds to the same rotation as q; thus the
        sign of the reconstructed quaternion is arbitrary, and we return
        quaternions with positive w (q[0]).

        References
        ----------
        * http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        * Bar-Itzhack, Itzhack Y. (2000), "New method for extracting the
          quaternion from a rotation matrix", AIAA Journal of Guidance,
          Control and Dynamics 23(6):1085-1087 (Engineering Note), ISSN
          0731-5090

        Examples
        --------
        >>> import numpy as np
        >>> q = mat2quat(np.eye(3)) # Identity rotation
        >>> np.allclose(q, [1, 0, 0, 0])
        True
        >>> q = mat2quat(np.diag([1, -1, -1]))
        >>> np.allclose(q, [0, 1, 0, 0]) # 180 degree rotn around axis 0
        True

        '''
        # Qyx refers to the contribution of the y input vector component to
        # the x output vector component.  Qyx is therefore the same as
        # M[0,1].  The notation is from the Wikipedia article.
        Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
        # Fill only lower half of symmetric matrix
        K = np.array([
            [Qxx - Qyy - Qzz, 0,               0,               0              ],
            [Qyx + Qxy,       Qyy - Qxx - Qzz, 0,               0              ],
            [Qzx + Qxz,       Qzy + Qyz,       Qzz - Qxx - Qyy, 0              ],
            [Qyz - Qzy,       Qzx - Qxz,       Qxy - Qyx,       Qxx + Qyy + Qzz]]
            ) / 3.0
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K)
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[0] < 0:
            q *= -1
        return q
        
    
    