import pytest
import trimesh
import fcl
import numpy as np

from linkmotion.robot.shape.mesh import MeshShape
from linkmotion.transform import Transform


@pytest.fixture
def box_mesh() -> trimesh.Trimesh:
    """Provides a simple Trimesh box as a test fixture."""
    return trimesh.creation.box(extents=[1, 2, 3])


@pytest.fixture
def sphere_mesh() -> trimesh.Trimesh:
    """Provides a simple Trimesh sphere as a test fixture."""
    return trimesh.creation.icosphere(radius=1.0)


def test_mesh_shape_initialization_basic(box_mesh):
    """Tests basic initialization with only a collision mesh."""
    shape = MeshShape(collision_mesh=box_mesh)

    # Assert that the visual mesh is a copy of the collision mesh
    assert isinstance(shape.visual_mesh, trimesh.Trimesh)
    np.testing.assert_array_equal(shape.visual_mesh.vertices, box_mesh.vertices)
    assert shape.visual_mesh is not box_mesh  # Should be a copy

    # Assert that the collision primitive is correctly created
    assert isinstance(shape.collision_primitive, fcl.BVHModel)
    np.testing.assert_array_equal(shape.collision_mesh.vertices, box_mesh.vertices)

    # Assert default transform is identity
    assert shape.default_transform == Transform()


def test_mesh_shape_initialization_with_visual_mesh(box_mesh, sphere_mesh):
    """Tests initialization with a separate visual mesh."""
    transform = Transform(translate=np.array([1, 1, 1]))
    shape = MeshShape(
        collision_mesh=box_mesh, visual_mesh=sphere_mesh, default_transform=transform
    )

    # Assert visual mesh is the provided sphere
    np.testing.assert_array_equal(shape.visual_mesh.vertices, sphere_mesh.vertices)

    # Assert collision primitive is based on the box
    np.testing.assert_array_equal(shape.collision_mesh.vertices, box_mesh.vertices)

    # Assert transform is set correctly
    print(shape.default_transform.position)
    print(shape.default_transform.rotation)
    assert shape.default_transform != Transform()
    np.testing.assert_array_equal(shape.default_transform.position, [1, 1, 1])


def test_from_other(box_mesh):
    """Tests the from_other class method for deep copying."""
    original_transform = Transform(translate=np.array([0, 0, 1]))
    original_shape = MeshShape(
        collision_mesh=box_mesh, default_transform=original_transform
    )

    new_shape = MeshShape.from_other(original_shape)

    # Assert they are different objects
    assert new_shape is not original_shape

    # Assert the data is equal but not the same instance
    np.testing.assert_array_equal(
        new_shape.collision_mesh.vertices,
        original_shape.collision_mesh.vertices,
    )
    assert new_shape.collision_mesh is not original_shape.collision_mesh

    np.testing.assert_array_equal(
        new_shape.default_transform.to_4x4(), original_shape.default_transform.to_4x4()
    )
    assert new_shape.default_transform is not original_shape.default_transform


def test_transformed_collision_mesh_no_transform(box_mesh):
    """Tests transformed_collision_mesh with no transform (uses default)."""
    default_transform = Transform(translate=np.array([1, 2, 3]))
    shape = MeshShape(collision_mesh=box_mesh, default_transform=default_transform)

    transformed_mesh = shape.transformed_collision_mesh()

    # Should apply the default transform
    assert isinstance(transformed_mesh, trimesh.Trimesh)
    # Verify that the mesh has been transformed by checking vertices
    expected_vertices = default_transform.apply(box_mesh).vertices
    np.testing.assert_array_almost_equal(transformed_mesh.vertices, expected_vertices)


def test_transformed_collision_mesh_with_transform(box_mesh):
    """Tests transformed_collision_mesh with an explicit transform."""
    default_transform = Transform(translate=np.array([1, 0, 0]))
    shape = MeshShape(collision_mesh=box_mesh, default_transform=default_transform)

    additional_transform = Transform(translate=np.array([0, 2, 0]))
    transformed_mesh = shape.transformed_collision_mesh(additional_transform)

    # Should apply both transforms (additional_transform.apply(default_transform))
    assert isinstance(transformed_mesh, trimesh.Trimesh)
    combined_transform = additional_transform.apply(default_transform)
    expected_vertices = combined_transform.apply(box_mesh).vertices
    np.testing.assert_array_almost_equal(transformed_mesh.vertices, expected_vertices)


def test_transformed_collision_mesh_identity(box_mesh):
    """Tests transformed_collision_mesh with identity transforms."""
    shape = MeshShape(collision_mesh=box_mesh)  # Default transform is identity

    transformed_mesh = shape.transformed_collision_mesh()

    # Should be the same as the original mesh (but a transformed copy)
    assert isinstance(transformed_mesh, trimesh.Trimesh)
    np.testing.assert_array_almost_equal(transformed_mesh.vertices, box_mesh.vertices)


def test_mesh_shape_copy(box_mesh):
    """Tests the copy method creates an independent copy."""
    original_transform = Transform(translate=np.array([1, 2, 3]))
    original_shape = MeshShape(
        collision_mesh=box_mesh,
        default_transform=original_transform,
        color=np.array([1.0, 0.0, 0.0, 1.0]),
    )

    copied_shape = original_shape.copy()

    # Assert they are different objects
    assert copied_shape is not original_shape

    # Assert the data is equal
    np.testing.assert_array_equal(
        copied_shape.collision_mesh.vertices, original_shape.collision_mesh.vertices
    )
    np.testing.assert_array_equal(
        copied_shape.default_transform.to_4x4(),
        original_shape.default_transform.to_4x4(),
    )
    np.testing.assert_array_equal(copied_shape.color, original_shape.color)


def test_mesh_shape_repr(box_mesh):
    """Tests the __repr__ method."""
    shape = MeshShape(collision_mesh=box_mesh)

    repr_str = repr(shape)

    assert "MeshShape" in repr_str
    assert "vertices" in repr_str
    assert "faces" in repr_str
    assert str(len(box_mesh.vertices)) in repr_str
    assert str(len(box_mesh.faces)) in repr_str


def test_create_collision_primitive(box_mesh):
    """Tests that the collision primitive is correctly created."""
    shape = MeshShape(collision_mesh=box_mesh)

    # Create a new collision primitive
    new_primitive = shape.create_collision_primitive()

    assert isinstance(new_primitive, fcl.BVHModel)
    # The collision primitive should be valid for collision detection
