from typing import overload
import logging

import numpy as np
import fcl
from scipy.spatial.transform import Rotation as R
import trimesh

from linkmotion.typing.numpy import Vector3, Matrix4x4

logger = logging.getLogger(__name__)


def is_vector_transformable(vector: np.ndarray) -> bool:
    """Checks if a numpy array is a transformable 3D vector or a list of 3D vectors.

    Args:
        vector: The numpy array to check.

    Returns:
        True if the array shape is (3,) or (N, 3), False otherwise.
    """
    if not isinstance(vector, np.ndarray):
        return False

    # Must be 1D or 2D array
    if vector.ndim not in (1, 2):
        return False

    # Last dimension must be 3 (for 3D vectors)
    if vector.shape[-1] != 3:
        return False

    # Must be numeric dtype
    if not np.issubdtype(vector.dtype, np.number):
        return False

    return True


class Transform:
    """Represents a 3D rigid transformation (rotation and translation).

    This class provides a convenient way to handle 3D transformations,
    offering conversions to and from other common libraries like FCL and Trimesh,
    as well as applying transformations to various geometric
    objects.

    Attributes:
        rotation (R): A `scipy.spatial.transform.Rotation` object representing the orientation.
        position (Vector3): A numpy array of shape (3,) representing the translation.
    """

    def __init__(self, rotate: R | None = None, translate: Vector3 | None = None):
        """Initializes the Transform.

        Args:
            rotate: The rotation component. Defaults to identity rotation.
            translate: The translation component (position vector). Defaults to the zero vector.
        """
        self.rotation: R = R.identity() if rotate is None else rotate

        if translate is None:
            self.position: Vector3 = np.array((0, 0, 0), dtype=float)
        else:
            translate_array = np.asarray(translate, dtype=float)
            self.position: Vector3 = translate_array

    def __repr__(self) -> str:
        """Returns a concise string representation for debugging."""
        return (
            f"Transform(rotation={self.rotation.as_rotvec()}, position={self.position})"
        )

    def copy(self: "Transform") -> "Transform":
        """Creates a deep copy of the Transform object.

        Returns:
            A new Transform instance with the same rotation and position.
        """
        # More efficient: copy quaternion directly instead of matrix conversion
        quat = self.rotation.as_quat().copy()
        return self.__class__(R.from_quat(quat), self.position.copy())

    def is_identity(self) -> bool:
        """Checks if the transform is an identity transform.

        Returns:
            True if the transform is identity, False otherwise.
        """
        # Use allclose for floating point comparison and more efficient check
        is_position_zero = np.allclose(self.position, 0.0, atol=1e-9)
        # Check rotation matrix diagonal elements for identity (more efficient than quaternion)
        rot_matrix = self.rotation.as_matrix()
        is_rotation_identity = np.allclose(
            np.diag(rot_matrix), 1.0, atol=1e-9
        ) and np.allclose(rot_matrix - np.diag(np.diag(rot_matrix)), 0.0, atol=1e-9)
        return is_position_zero and is_rotation_identity

    def __str__(self) -> str:
        """Returns a string representation of the transform."""
        return f"Transform(\n  rotation:\n{self.rotation.as_matrix()},\n  position:\n{self.position}\n)"

    def __eq__(self, other) -> bool:
        """Checks for equality between two Transform objects.

        Comparison is done using `numpy.allclose` to account for floating-point inaccuracies.

        Args:
            other: The object to compare with.

        Returns:
            True if the rotation and position are approximately equal, False otherwise.
        """
        if not isinstance(other, Transform):
            return NotImplemented
        pos_equal = np.allclose(self.position, other.position)
        rot_equal = np.allclose(self.rotation.as_quat(), other.rotation.as_quat())
        return pos_equal and rot_equal

    @classmethod
    def from_fcl(cls: type["Transform"], transform: fcl.Transform) -> "Transform":
        """Creates a Transform instance from an fcl.Transform object.

        Args:
            transform: The fcl.Transform object.

        Returns:
            A new Transform instance.
        """
        rotate = R.from_matrix(transform.getRotation().copy())
        translate = transform.getTranslation().copy()
        return cls(rotate, translate)

    @classmethod
    def from_4x4(cls: type["Transform"], matrix: Matrix4x4) -> "Transform":
        """Creates a Transform instance from a 4x4 homogeneous transformation matrix.

        Args:
            matrix: The 4x4 numpy array.

        Returns:
            A new Transform instance.
        """
        rotate = R.from_matrix(matrix.copy()[:3, :3])
        translate = matrix[:3, 3]
        return cls(rotate, translate)

    def to_fcl(self) -> fcl.Transform:
        """Converts this Transform to an fcl.Transform object.

        Returns:
            An fcl.Transform object.
        """
        t = self.copy()
        return fcl.Transform(t.rotation.as_matrix(), t.position)

    def to_4x4(self) -> Matrix4x4:
        """Converts this Transform to a 4x4 homogeneous transformation matrix.

        Returns:
            A 4x4 numpy array representing the transformation.
        """
        matrix: Matrix4x4 = np.eye(4)
        matrix[:3, :3] = self.rotation.as_matrix()
        matrix[:3, 3] = self.position
        return matrix

    @overload
    def apply(self, obj: np.ndarray) -> np.ndarray: ...

    @overload
    def apply(self, obj: trimesh.Trimesh) -> trimesh.Trimesh: ...

    @overload
    def apply(self, obj: "Transform") -> "Transform": ...

    @overload
    def apply(self, obj: fcl.Transform) -> fcl.Transform: ...

    # @overload
    # def apply(self, obj: fcl.CollisionGeometry) -> fcl.CollisionObject: ...

    # @overload
    # def apply(self, obj: fcl.CollisionObject) -> None: ...

    def apply(self, obj):
        """Applies this transformation to a given object.

        The behavior depends on the type of the input object `obj`. For most types,
        it returns a new, transformed object.

        Note:
            When applying to an `fcl.CollisionObject`, the object is modified
            in-place and the method returns `None`. This is an exception to the
            general behavior.

        Args:
            obj: The object to be transformed. Supported types include:
                - Transform
                - np.ndarray (as Vector3 or Vector3s)
                - fcl.Transform
                - trimesh.Trimesh
                - fcl.CollisionGeometry
                - fcl.CollisionObject

        Returns:
            The transformed object. The return type depends on the input type.
            Returns `None` if `obj` is an `fcl.CollisionObject`.

        Raises:
            TypeError: If the object type is not supported.
        """
        match obj:
            case Transform():
                if self.is_identity():  # Optimization for identity transform
                    return obj.copy()
                new_translate = self.rotation.apply(obj.position) + self.position
                new_rotate = self.rotation * obj.rotation
                return Transform(new_rotate, new_translate)
            case np.ndarray() as v if is_vector_transformable(v):
                if self.is_identity():  # Optimization for identity transform
                    return v.copy()
                return self.rotation.apply(v) + self.position
            case fcl.Transform():
                fcl_transform = Transform.from_fcl(obj)
                return self.apply(fcl_transform).to_fcl()
            case trimesh.Trimesh():
                if self.is_identity():  # Optimization for identity transform
                    return obj.copy()
                new_mesh = obj.copy()
                new_mesh.vertices = self.apply(new_mesh.vertices)
                return new_mesh
            case fcl.CollisionGeometry():
                return fcl.CollisionObject(obj, self.to_fcl())
            case fcl.CollisionObject():
                new_t = self.apply(obj.getTransform())
                obj.setTransform(new_t)
                logger.info(
                    "side-effect: transform is applied and fcl.CollisionObject is modified in-place."
                )
                return None
            case _:
                raise TypeError(f"Unsupported type for apply: {type(obj)}")

    def transformed(self, transform: "Transform") -> "Transform":
        """Returns a new Transform by applying another transform to this one.

        This is equivalent to `transform.apply(self)`.

        Args:
            transform: The transform to apply.

        Returns:
            The new, resulting Transform.
        """
        return transform.apply(self)

    def translate(self, translate: Vector3):
        """Applies a translation to this transform in-place.

        Args:
            translate: The translation vector to add to the current position.
        """
        translate_array = np.asarray(translate, dtype=float)
        self.position = self.position + translate_array

    def rotate(self, rotate: R):
        """Applies a rotation to this transform in-place.

        The rotation is applied with respect to the global origin.

        Args:
            rotate: The rotation to apply.
        """
        self.position = rotate.apply(self.position)
        self.rotation = rotate * self.rotation

    def rotate_around(self, center: Vector3, rotate: R):
        """Rotates this transform around a specific point in space, in-place.

        Args:
            center: The 3D point (vector) to rotate around.
            rotate: The rotation to apply.
        """
        # Move position relative to the center
        relative_pos = self.position - center
        # Rotate the relative position
        new_relative_pos = rotate.apply(relative_pos)
        # Move back and update the position
        self.position = new_relative_pos + center
        # Apply rotation to the object's orientation
        self.rotation = rotate * self.rotation

    def transform(self, transform: "Transform"):
        """Applies another transform to this one in-place.

        Args:
            transform: The transform to apply.
        """
        new_t = transform.apply(self)
        self.position = new_t.position
        self.rotation = new_t.rotation
