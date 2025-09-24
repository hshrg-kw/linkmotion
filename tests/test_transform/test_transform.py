import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
import fcl

from linkmotion.transform import Transform
from linkmotion.transform.transform import is_vector_transformable


# --- Fixtures (preparation for test) ---
@pytest.fixture
def identity_transform():
    """Returns an identity Transform object."""
    return Transform()


@pytest.fixture
def sample_transform():
    """Returns a sample Transform object (90 deg rotation around Z, translation)."""
    rotation = R.from_euler("z", 90, degrees=True)
    translation = np.array([1.0, 2.0, 3.0])
    return Transform(rotation, translation)


@pytest.fixture
def another_transform():
    """Returns another sample Transform for composition tests."""
    rotation = R.from_euler("x", -90, degrees=True)
    translation = np.array([4.0, 5.0, 6.0])
    return Transform(rotation, translation)


# --- Test Cases ---


def test_init_default(identity_transform: Transform):
    """Tests default constructor to be an identity transform."""
    assert np.allclose(identity_transform.position, np.zeros(3))
    assert np.allclose(identity_transform.rotation.as_matrix(), np.eye(3))
    assert identity_transform == Transform(R.identity(), np.array([0.0, 0.0, 0.0]))


def test_init_with_values(sample_transform: Transform):
    """Tests constructor with specified rotation and translation."""
    rotation = R.from_euler("z", 90, degrees=True)
    translation = np.array([1.0, 2.0, 3.0])
    assert np.allclose(sample_transform.position, translation)
    assert np.allclose(sample_transform.rotation.as_matrix(), rotation.as_matrix())


def test_copy(sample_transform: Transform):
    """Tests the copy method."""
    original = sample_transform
    copied = original.copy()

    assert copied == original
    assert copied is not original
    assert copied.position is not original.position

    # Modify original and check that copy is unaffected
    original.translate(np.array([10, 10, 10]))
    assert copied != original


def test_eq(sample_transform: Transform):
    """Tests the equality operator."""
    t1 = sample_transform
    t2 = sample_transform.copy()
    t3 = Transform(R.from_euler("z", 91, degrees=True), np.array([1, 2, 3]))
    t4 = Transform(R.from_euler("z", 90, degrees=True), np.array([1, 2, 3.1]))

    assert (t1 == t2) is True
    assert (t1 == t3) is False
    assert (t1 == t4) is False
    assert (t1 == "not a transform") is False


def test_to_and_from_4x4(sample_transform: Transform):
    """Tests conversion to and from a 4x4 homogeneous matrix."""
    matrix_4x4 = sample_transform.to_4x4()
    reconstructed_transform = Transform.from_4x4(matrix_4x4)

    assert matrix_4x4.shape == (4, 4)
    assert np.allclose(matrix_4x4[:3, :3], sample_transform.rotation.as_matrix())
    assert np.allclose(matrix_4x4[:3, 3], sample_transform.position)
    assert np.allclose(matrix_4x4[3, :], [0, 0, 0, 1])
    assert sample_transform == reconstructed_transform


def test_to_and_from_fcl(sample_transform: Transform):
    """Tests conversion to and from fcl.Transform."""
    fcl_tf = sample_transform.to_fcl()
    reconstructed_transform = Transform.from_fcl(fcl_tf)

    assert isinstance(fcl_tf, fcl.Transform)
    assert sample_transform == reconstructed_transform


def test_apply_to_vector(sample_transform: Transform):
    """Tests applying the transform to a single 3D vector."""
    vector = np.array([5.0, 0.0, 0.0])
    # Expected: Rotate (5,0,0) by +90deg Z -> (0,5,0). Then translate by (1,2,3) -> (1,7,3)
    expected_result = np.array([1.0, 7.0, 3.0])

    result = sample_transform.apply(vector)
    assert np.allclose(result, expected_result)


def test_apply_to_vectors(sample_transform: Transform):
    """Tests applying the transform to a list of 3D vectors."""
    vectors = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    # Expected:
    # v1 -> (1, 7, 3)
    # v2: Rotate (0,5,0) by +90deg Z -> (-5,0,0). Then translate by (1,2,3) -> (-4,2,3)
    expected_results = np.array([[1.0, 7.0, 3.0], [-4.0, 2.0, 3.0]])

    results = sample_transform.apply(vectors)
    assert np.allclose(results, expected_results)


def test_apply_identity_to_vector(identity_transform: Transform):
    """Tests applying an identity transform to a vector."""
    vector = np.array([1.2, 3.4, 5.6])
    result = identity_transform.apply(vector)
    assert np.allclose(result, vector)
    assert result is not vector  # Should be a copy


def test_apply_to_transform(sample_transform: Transform, another_transform: Transform):
    """Tests transform composition by applying a transform to another."""
    # Combine transforms using apply
    composed_transform = sample_transform.apply(another_transform)

    # Combine transforms using 4x4 matrix multiplication
    mat_a = sample_transform.to_4x4()
    mat_b = another_transform.to_4x4()
    expected_mat = mat_a @ mat_b

    assert np.allclose(composed_transform.to_4x4(), expected_mat)


def test_apply_to_trimesh(sample_transform: Transform):
    """Tests applying the transform to a trimesh object."""
    # A simple triangle mesh
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    transformed_mesh = sample_transform.apply(mesh)

    expected_vertices = sample_transform.apply(vertices)

    assert isinstance(transformed_mesh, trimesh.Trimesh)
    assert transformed_mesh is not mesh
    assert np.allclose(transformed_mesh.vertices, expected_vertices)


def test_apply_unsupported_type(sample_transform: Transform):
    """Tests that applying to an unsupported type raises TypeError."""
    with pytest.raises(TypeError):
        sample_transform.apply("a string")
    with pytest.raises(TypeError):
        sample_transform.apply(123)


def test_translate_inplace(sample_transform: Transform):
    """Tests the in-place translate method."""
    original_pos = sample_transform.position.copy()
    translation = np.array([10, -5, 2])

    sample_transform.translate(translation)

    assert np.allclose(sample_transform.position, original_pos + translation)


def test_rotate_inplace(sample_transform: Transform):
    """Tests the in-place rotate method."""
    original_pos = sample_transform.position.copy()
    original_rot = sample_transform.rotation

    # Rotate by 90 degrees around X axis
    rotate_by = R.from_euler("x", 90, degrees=True)
    sample_transform.rotate(rotate_by)

    expected_pos = rotate_by.apply(original_pos)
    expected_rot = rotate_by * original_rot

    assert np.allclose(sample_transform.position, expected_pos)
    assert np.allclose(sample_transform.rotation.as_quat(), expected_rot.as_quat())


def test_rotate_around_inplace(identity_transform: Transform):
    """Tests the in-place rotate_around method."""
    # Start with an object at (5, 0, 0)
    identity_transform.position = np.array([5.0, 0.0, 0.0])

    # Rotate it 90 degrees around Z axis, centered at (3, 0, 0)
    center = np.array([3.0, 0.0, 0.0])
    rotate_by = R.from_euler("z", 90, degrees=True)

    identity_transform.rotate_around(center, rotate_by)

    # Expected result:
    # Relative vector from center: (5,0,0) - (3,0,0) = (2,0,0)
    # Rotated relative vector: (0,2,0)
    # New position: (3,0,0) + (0,2,0) = (3,2,0)
    expected_position = np.array([3.0, 2.0, 0.0])
    expected_rotation = rotate_by  # Since original was identity

    assert np.allclose(identity_transform.position, expected_position)
    assert np.allclose(
        identity_transform.rotation.as_matrix(), expected_rotation.as_matrix()
    )


def test_transform_inplace(sample_transform: Transform, another_transform: Transform):
    """Tests the in-place transform method."""
    # Create a copy to compare against
    t_to_modify = sample_transform.copy()

    # Calculate expected result using the functional `apply` method
    expected_result = another_transform.apply(sample_transform)

    # Perform the in-place modification
    t_to_modify.transform(another_transform)

    assert t_to_modify == expected_result


# FCL integration tests for transform.py
def test_apply_to_fcl_transform():
    t = Transform(translate=np.array([1, 2, 3]))
    fcl_t = fcl.Transform()
    result = t.apply(fcl_t)
    assert isinstance(result, fcl.Transform)


# Edge case tests
def test_is_vector_transformable_invalid():
    assert not is_vector_transformable(np.array([1, 2]))  # Wrong shape
    assert not is_vector_transformable(np.array([[1], [2], [3], [4]]))  # 4D
    assert not is_vector_transformable("not an array")  # Not ndarray
    assert not is_vector_transformable(np.array([[[1, 2, 3]]]))  # 3D array
    assert not is_vector_transformable(np.array(["a", "b", "c"]))  # Non-numeric


def test_is_vector_transformable_valid():
    """Tests valid inputs for is_vector_transformable."""
    assert is_vector_transformable(np.array([1, 2, 3]))  # 1D vector
    assert is_vector_transformable(np.array([[1, 2, 3], [4, 5, 6]]))  # 2D array
    assert is_vector_transformable(np.array([1.0, 2.0, 3.0]))  # Float array
    assert is_vector_transformable(np.array([1, 2, 3], dtype=np.int32))  # Int array


def test_transform_init_with_various_inputs():
    """Tests Transform initialization with various valid inputs."""
    # Test with different types that get converted to arrays
    t1 = Transform(translate=[1, 2, 3])  # List
    assert np.allclose(t1.position, [1, 2, 3])

    t2 = Transform(translate=(4, 5, 6))  # Tuple
    assert np.allclose(t2.position, [4, 5, 6])

    t3 = Transform(translate=np.array([7, 8, 9]))  # Array
    assert np.allclose(t3.position, [7, 8, 9])


def test_translate_with_various_inputs():
    """Tests translate method with various valid inputs."""
    t = Transform()

    # Test with list
    t.translate([1, 2, 3])
    assert np.allclose(t.position, [1, 2, 3])

    # Test with tuple
    t.translate((1, 1, 1))
    assert np.allclose(t.position, [2, 3, 4])

    # Test with array
    t.translate(np.array([-1, -1, -1]))
    assert np.allclose(t.position, [1, 2, 3])


def test_is_identity_edge_cases():
    """Tests is_identity method with edge cases."""
    # Test near-zero values (should be identity)
    t1 = Transform(translate=np.array([1e-10, 1e-10, 1e-10]))
    assert t1.is_identity()

    # Test values just above tolerance (should not be identity)
    t2 = Transform(translate=np.array([1e-8, 0, 0]))
    assert not t2.is_identity()

    # Test small rotation (should not be identity)
    # Note: Very small rotations might still be considered identity due to floating point precision
    small_rotation = R.from_euler("z", 0.001, degrees=True)  # Use larger angle
    t3 = Transform(rotate=small_rotation)
    assert not t3.is_identity()

    # Test explicit identity
    t4 = Transform()
    assert t4.is_identity()


def test_transformed_method():
    """Tests the transformed method."""
    t1 = Transform(translate=np.array([1, 0, 0]))
    t2 = Transform(translate=np.array([0, 1, 0]))

    result = t1.transformed(t2)
    expected = t2.apply(t1)

    assert result == expected
    assert result is not t1  # Should be a new object
    assert result is not t2


def test_str_representation():
    """Tests the string representation of Transform."""
    t = Transform(translate=np.array([1, 2, 3]))
    str_repr = str(t)
    assert "Transform(" in str_repr
    assert "rotation:" in str_repr
    assert "position:" in str_repr
    assert "[1 2 3]" in str_repr or "[1. 2. 3.]" in str_repr


def test_apply_fcl_collision_object():
    """Tests applying transform to FCL CollisionObject (in-place modification)."""
    # Create a simple box geometry
    box = fcl.Box(1, 1, 1)
    initial_transform = fcl.Transform()
    collision_obj = fcl.CollisionObject(box, initial_transform)

    # Apply a transform
    t = Transform(translate=np.array([5, 0, 0]))
    result = t.apply(collision_obj)

    # Should return None (in-place modification)
    assert result is None

    # Check that the collision object was modified
    modified_transform = collision_obj.getTransform()
    expected_translation = np.array([5, 0, 0])
    assert np.allclose(modified_transform.getTranslation(), expected_translation)


def test_apply_fcl_collision_geometry():
    """Tests applying transform to FCL CollisionGeometry."""
    box = fcl.Box(1, 1, 1)
    t = Transform(translate=np.array([2, 3, 4]))

    result = t.apply(box)

    assert isinstance(result, fcl.CollisionObject)
    transform = result.getTransform()
    assert np.allclose(transform.getTranslation(), [2, 3, 4])


def test_from_4x4_edge_cases():
    """Tests Transform.from_4x4 with edge cases."""
    # Identity matrix
    identity_matrix = np.eye(4)
    t = Transform.from_4x4(identity_matrix)
    assert t.is_identity()

    # Matrix with only translation
    trans_matrix = np.eye(4)
    trans_matrix[:3, 3] = [10, 20, 30]
    t = Transform.from_4x4(trans_matrix)
    assert np.allclose(t.position, [10, 20, 30])
    assert np.allclose(t.rotation.as_matrix(), np.eye(3))


def test_equality_with_floating_point_precision():
    """Tests equality handling with floating point precision issues."""
    # Create two transforms that should be equal but might have precision differences
    t1 = Transform(translate=np.array([1.0, 2.0, 3.0]))
    t2 = Transform(translate=np.array([1.0000000001, 2.0000000001, 3.0000000001]))

    # Should be considered equal due to allclose tolerance
    assert t1 == t2

    # But not if difference is too large
    t3 = Transform(translate=np.array([1.001, 2.0, 3.0]))
    assert t1 != t3


def test_copy_independence():
    """Tests that copied transforms are truly independent."""
    original = Transform(translate=np.array([1, 2, 3]))
    copied = original.copy()

    # Modify original position directly
    original.position[0] = 999

    # Copy should be unaffected
    assert copied.position[0] == 1
    assert original.position[0] == 999
