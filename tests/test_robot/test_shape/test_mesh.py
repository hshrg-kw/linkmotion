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
