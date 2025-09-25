import pytest
import numpy as np
import trimesh
import fcl
from scipy.spatial.transform import Rotation


from linkmotion.transform import Transform
from linkmotion.robot.shape.base import ShapeBase
from linkmotion.robot.shape.box import Box
from linkmotion.robot.shape.capsule import Capsule
from linkmotion.robot.shape.cylinder import Cylinder
from linkmotion.robot.shape.cone import Cone
from linkmotion.robot.shape.sphere import Sphere

# --- Test Data and Fixtures ---

SHAPE_CLASSES = [Box, Capsule, Cylinder, Cone, Sphere]
TEST_COLOR = np.array([0.1, 0.2, 0.3, 0.4])


@pytest.fixture
def sample_transform():
    """Provides a sample non-identity transform."""
    translation = np.array([1.0, 2.0, 3.0])
    rotation = Rotation.from_euler("xyz", [np.pi / 2, 0, 0])
    return Transform(translate=translation, rotate=rotation)


@pytest.fixture(params=SHAPE_CLASSES)
def shape_instance(request):
    """Provides an instance of each shape class for parameterized tests."""
    shape_class = request.param
    if shape_class == Box:
        return Box(extents=np.array([1, 2, 3]), color=TEST_COLOR)
    elif shape_class == Capsule:
        return Capsule(radius=0.5, height=2.0, color=TEST_COLOR)
    elif shape_class == Cylinder:
        return Cylinder(radius=0.5, height=2.0, color=TEST_COLOR)
    elif shape_class == Cone:
        return Cone(radius=0.5, height=2.0, color=TEST_COLOR)
    elif shape_class == Sphere:
        return Sphere(radius=1.0, color=TEST_COLOR)
    return None


# --- Test Cases ---


def test_shape_creation(shape_instance: ShapeBase):
    """Tests basic instantiation and property assignment."""
    assert shape_instance is not None
    assert isinstance(shape_instance.visual_mesh, trimesh.Trimesh)
    assert isinstance(shape_instance.collision_primitive, fcl.CollisionGeometry)
    assert shape_instance.default_transform == Transform()


def test_shape_creation_with_transform(sample_transform: Transform):
    """Tests instantiation with a non-default transform."""
    box = Box(extents=np.array([1, 1, 1]), default_transform=sample_transform)
    assert box.default_transform == sample_transform

    sphere = Sphere(radius=1.0, center=sample_transform.position)
    np.testing.assert_allclose(
        sphere.default_transform.position, sample_transform.position
    )


def test_mesh_and_collision_origin_centered(shape_instance: ShapeBase):
    """Tests if the visual mesh is centered at the origin."""
    # The centroid of a symmetric mesh should be at or very close to the origin.
    mesh_centroid = shape_instance.visual_mesh.centroid
    if (
        isinstance(shape_instance, Box)
        or isinstance(shape_instance, Capsule)
        or isinstance(shape_instance, Cylinder)
        or isinstance(shape_instance, Sphere)
    ):
        np.testing.assert_allclose(mesh_centroid, [0, 0, 0], atol=1e-9)
    return None


def test_color_application(shape_instance: ShapeBase):
    """Tests if the color is correctly applied to the visual mesh."""
    if shape_instance.visual_mesh.visual is not None:
        assert shape_instance.visual_mesh.visual.kind == "vertex"
        mesh_color = shape_instance.visual_mesh.visual.vertex_colors[0]
        expected_color_255 = (TEST_COLOR * 255).astype(np.uint8)
        np.testing.assert_array_equal(mesh_color, expected_color_255)


def test_from_other_method(shape_instance: ShapeBase):
    """Tests the from_other class method for correct copying."""
    new_transform = Transform(translate=np.array([10, 20, 30]))
    shape_instance.default_transform = new_transform
    print(shape_instance.default_transform)

    copied_instance = shape_instance.__class__.from_other(shape_instance)
    print(copied_instance.default_transform)

    assert isinstance(copied_instance, shape_instance.__class__)
    assert copied_instance is not shape_instance
    # print(copied_instance.default_transform)
    # print(shape_instance.default_transform)
    assert copied_instance.default_transform == shape_instance.default_transform
    assert copied_instance.default_transform is not shape_instance.default_transform


def test_transformed_visual_mesh(
    shape_instance: ShapeBase, sample_transform: Transform
):
    """Tests the transformation of the visual mesh."""
    original_bounds = shape_instance.visual_mesh.bounds.copy()
    expected_bounds = sample_transform.apply(original_bounds)
    expected_bounds = np.sort(expected_bounds, axis=0)

    shape_instance.default_transform = sample_transform
    transformed_mesh = shape_instance.transformed_visual_mesh()
    target_bounds = np.sort(transformed_mesh.bounds, axis=0)

    np.testing.assert_allclose(target_bounds, expected_bounds, atol=1e-6)


def test_transformed_collision_object(
    shape_instance: ShapeBase, sample_transform: Transform
):
    """Tests the creation of a transformed FCL collision object."""
    shape_instance.default_transform = sample_transform

    collision_object = shape_instance.transformed_collision_object()
    fcl_tf = collision_object.getTransform()

    np.testing.assert_allclose(fcl_tf.getTranslation(), sample_transform.position)


def test_sphere_collision_primitive_bugfix():
    """Ensures Sphere creates an fcl.Sphere, not a Capsule."""
    sphere = Sphere(radius=1.0)
    primitive = sphere.create_collision_primitive()
    assert isinstance(primitive, fcl.Sphere)
    assert primitive.radius == 1.0
