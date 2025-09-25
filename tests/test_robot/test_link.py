import pytest
import numpy as np
import trimesh

from linkmotion.robot import Link
from linkmotion.robot.shape.box import Box
from linkmotion.robot.shape.sphere import Sphere
from linkmotion.robot.shape.cylinder import Cylinder
from linkmotion.robot.shape.cone import Cone
from linkmotion.robot.shape.capsule import Capsule
from linkmotion.robot.shape.mesh import MeshShape
from linkmotion.transform import Transform


@pytest.fixture
def link1():
    return Link.from_sphere("link1", 1)


class TestLink:
    def test_init(self, link1: Link):
        assert link1.name == "link1"
        assert link1.shape is not None

    def test_str_representation(self, link1: Link):
        str_repr = str(link1)
        assert "Link" in str_repr
        assert "link1" in str_repr
        assert "shape=" in str_repr

    def test_from_box_valid(self):
        extents = np.array([1.0, 2.0, 3.0])
        link = Link.from_box("box_link", extents)
        assert link.name == "box_link"
        assert isinstance(link.shape, Box)
        assert np.array_equal(link.shape.extents, extents)

    def test_from_box_with_transform_and_color(self):
        extents = np.array([1.0, 1.0, 1.0])
        transform = Transform(translate=np.array([1.0, 2.0, 3.0]))
        color = np.array([1.0, 0.0, 0.0, 1.0])
        link = Link.from_box("colored_box", extents, transform, color)
        assert link.name == "colored_box"
        if link.shape.color is not None:
            assert np.array_equal(link.shape.color, color)
        assert np.array_equal(link.shape.default_transform.position, [1, 2, 3])

    def test_from_cylinder_valid(self):
        link = Link.from_cylinder("cyl_link", radius=1.5, height=3.0)
        assert link.name == "cyl_link"
        assert isinstance(link.shape, Cylinder)
        assert link.shape.radius == 1.5
        assert link.shape.height == 3.0

    def test_from_cone_valid(self):
        link = Link.from_cone("cone_link", radius=2.0, height=4.0)
        assert link.name == "cone_link"
        assert isinstance(link.shape, Cone)
        assert link.shape.radius == 2.0
        assert link.shape.height == 4.0

    def test_from_capsule_valid(self):
        link = Link.from_capsule("capsule_link", radius=1.0, height=2.0)
        assert link.name == "capsule_link"
        assert isinstance(link.shape, Capsule)
        assert link.shape.radius == 1.0
        assert link.shape.height == 2.0

    def test_from_sphere_with_center(self):
        center = np.array([1.0, 2.0, 3.0])
        link = Link.from_sphere("sphere_with_center", radius=2.0, center=center)
        assert link.name == "sphere_with_center"
        assert isinstance(link.shape, Sphere)
        assert np.array_equal(link.shape.center, center)

    def test_from_mesh_valid(self):
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        link = Link.from_mesh("mesh_link", mesh)
        assert link.name == "mesh_link"
        assert isinstance(link.shape, MeshShape)

    def test_collision_object_default_transform(self, link1: Link):
        collision_obj = link1.collision_object()
        assert collision_obj is not None

    def test_collision_object_with_transform(self, link1: Link):
        transform = Transform(translate=np.array([1.0, 0.0, 0.0]))
        collision_obj = link1.collision_object(transform)
        assert collision_obj is not None

    def test_visual_mesh_default_transform(self, link1: Link):
        mesh = link1.visual_mesh()
        assert isinstance(mesh, trimesh.Trimesh)

    def test_visual_mesh_with_transform(self, link1: Link):
        transform = Transform(translate=np.array([1.0, 0.0, 0.0]))
        mesh = link1.visual_mesh(transform)
        assert isinstance(mesh, trimesh.Trimesh)

    def test_from_other_creates_independent_copy(self, link1: Link):
        new_link = Link.from_other(link1, "copied_link")
        assert new_link.name == "copied_link"
        assert new_link is not link1
        assert new_link.shape is not link1.shape

    def test_from_other_with_transform(self, link1: Link):
        transform = Transform(translate=np.array([5.0, 5.0, 5.0]))
        new_link = Link.from_other(link1, "transformed_link", transform)
        assert new_link.name == "transformed_link"
        expected_pos = transform.apply(link1.shape.default_transform).position
        assert np.allclose(new_link.shape.default_transform.position, expected_pos)
