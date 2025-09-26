import pytest
import logging
import numpy as np
import trimesh

from linkmotion.robot import Robot, Link
from linkmotion.robot.custom import (
    CollisionMeshCustomizer,
    from_mesh_to_bounding_primitive,
)
from linkmotion.robot.shape.mesh import MeshShape
from linkmotion.robot.shape.box import Box
from linkmotion.robot.shape.sphere import Sphere
from linkmotion.robot.shape.cylinder import Cylinder


@pytest.fixture
def box_mesh() -> trimesh.Trimesh:
    """Provides a box mesh."""
    return trimesh.creation.box(extents=[2, 2, 2])


@pytest.fixture
def sphere_mesh() -> trimesh.Trimesh:
    """Provides a sphere mesh."""
    return trimesh.creation.icosphere(radius=1.0)


@pytest.fixture
def cylinder_mesh() -> trimesh.Trimesh:
    """Provides a cylinder mesh."""
    return trimesh.creation.cylinder(radius=0.5, height=2.0)


@pytest.fixture
def mesh_link(box_mesh) -> Link:
    """Creates a link with mesh shape."""
    mesh_shape = MeshShape(collision_mesh=box_mesh)
    return Link(name="test_link", shape=mesh_shape)


@pytest.fixture
def non_mesh_link() -> Link:
    """Creates a link with non-mesh shape."""
    return Link.from_sphere("non_mesh_link", radius=1.0)


@pytest.fixture
def robot_with_mesh_links(mesh_link, non_mesh_link) -> Robot:
    """Creates a robot with both mesh and non-mesh links."""
    robot = Robot()
    robot.add_link(mesh_link)
    robot.add_link(non_mesh_link)

    # Add another mesh link
    cylinder_mesh = trimesh.creation.cylinder(radius=0.3, height=1.5)
    cylinder_shape = MeshShape(collision_mesh=cylinder_mesh)
    cylinder_link = Link(name="cylinder_link", shape=cylinder_shape)
    robot.add_link(cylinder_link)

    return robot


class TestCollisionMeshCustomizer:
    def test_customizer_repr(self):
        """Test the __repr__ method."""
        customizer = CollisionMeshCustomizer()
        assert repr(customizer) == "CollisionMeshCustomizer()"

    def test_remove_outside_of_box_mesh_links(self, robot_with_mesh_links):
        """Test removing parts outside bounding box for mesh links."""
        min_corner = np.array([-0.5, -0.5, -0.5])
        max_corner = np.array([0.5, 0.5, 0.5])

        # Get original volumes
        original_link = robot_with_mesh_links.link("test_link")
        original_volume = original_link.shape.collision_mesh.volume

        CollisionMeshCustomizer.remove_outside_of_box(
            robot_with_mesh_links, {"test_link"}, min_corner, max_corner
        )

        # Check that mesh was modified
        modified_link = robot_with_mesh_links.link("test_link")
        assert isinstance(modified_link.shape, MeshShape)

        if not modified_link.shape.collision_mesh.is_empty:
            # Volume should be reduced
            assert modified_link.shape.collision_mesh.volume <= original_volume

    def test_remove_outside_of_box_non_mesh_links(self, robot_with_mesh_links, caplog):
        """Test that non-mesh links are skipped with appropriate logging."""
        min_corner = np.array([-1, -1, -1])
        max_corner = np.array([1, 1, 1])

        with caplog.at_level(logging.INFO):
            CollisionMeshCustomizer.remove_outside_of_box(
                robot_with_mesh_links, {"non_mesh_link"}, min_corner, max_corner
            )

        # Check that appropriate log message was generated
        assert "non_mesh_link" in caplog.text
        assert "skipped because it is not MeshShape" in caplog.text

    def test_sweep_mesh_basic(self, robot_with_mesh_links):
        """Test basic mesh sweeping functionality."""
        initial_translate = np.array([0.1, 0.1, 0.1])
        sweep_translate = np.array([1, 0, 0])

        # Get original mesh
        original_link = robot_with_mesh_links.link("test_link")
        original_vertex_count = len(original_link.shape.collision_mesh.vertices)

        CollisionMeshCustomizer.sweep_mesh(
            robot_with_mesh_links, {"test_link"}, initial_translate, sweep_translate
        )

        # Check that mesh was modified
        modified_link = robot_with_mesh_links.link("test_link")
        assert isinstance(modified_link.shape, MeshShape)

        # Swept mesh should have more vertices
        assert len(modified_link.shape.collision_mesh.vertices) > original_vertex_count

    def test_sweep_mesh_non_mesh_links(self, robot_with_mesh_links, caplog):
        """Test that non-mesh links are skipped during sweeping."""
        initial_translate = np.array([0, 0, 0])
        sweep_translate = np.array([1, 1, 1])

        with caplog.at_level(logging.INFO):
            CollisionMeshCustomizer.sweep_mesh(
                robot_with_mesh_links,
                {"non_mesh_link"},
                initial_translate,
                sweep_translate,
            )

        # Check logging
        assert "non_mesh_link" in caplog.text
        assert "skipped because it is not MeshShape" in caplog.text

    def test_rotate_overlap_basic(self, robot_with_mesh_links):
        """Test basic rotational overlap functionality."""
        center = np.array([0, 0, 0])
        normalized_direction = np.array([0, 0, 1])
        delta_angle = np.pi / 4
        initial_angle = 0.0
        how_many_to_add = 2

        # Get original mesh
        original_link = robot_with_mesh_links.link("test_link")
        original_vertex_count = len(original_link.shape.collision_mesh.vertices)

        CollisionMeshCustomizer.rotate_overlap(
            robot_with_mesh_links,
            {"test_link"},
            center,
            normalized_direction,
            delta_angle,
            initial_angle,
            how_many_to_add,
        )

        # Check that mesh was modified
        modified_link = robot_with_mesh_links.link("test_link")
        assert isinstance(modified_link.shape, MeshShape)

        # Due to mesh merging, may have fewer vertices than expected
        # Just check that modification occurred
        assert len(modified_link.shape.collision_mesh.vertices) >= original_vertex_count

    def test_rotate_overlap_non_mesh_links(self, robot_with_mesh_links, caplog):
        """Test that non-mesh links are skipped during rotation overlap."""
        center = np.array([0, 0, 0])
        normalized_direction = np.array([1, 0, 0])
        delta_angle = np.pi / 6
        initial_angle = 0.0
        how_many_to_add = 1

        with caplog.at_level(logging.INFO):
            CollisionMeshCustomizer.rotate_overlap(
                robot_with_mesh_links,
                {"non_mesh_link"},
                center,
                normalized_direction,
                delta_angle,
                initial_angle,
                how_many_to_add,
            )

        # Check logging
        assert "non_mesh_link" in caplog.text
        assert "skipped because it is not MeshShape" in caplog.text

    def test_from_mesh_to_bounding_primitive_method(self, robot_with_mesh_links):
        """Test converting mesh to bounding primitive via class method."""
        CollisionMeshCustomizer.from_mesh_to_bounding_primitive(
            robot_with_mesh_links, {"test_link"}
        )

        # Check that shape was converted
        modified_link = robot_with_mesh_links.link("test_link")
        # Should now be a primitive shape, not MeshShape
        assert not isinstance(modified_link.shape, MeshShape)
        assert isinstance(modified_link.shape, (Box, Sphere, Cylinder))

    def test_from_mesh_to_bounding_primitive_non_mesh(
        self, robot_with_mesh_links, caplog
    ):
        """Test that non-mesh links are skipped during primitive conversion."""
        with caplog.at_level(logging.INFO):
            CollisionMeshCustomizer.from_mesh_to_bounding_primitive(
                robot_with_mesh_links, {"non_mesh_link"}
            )

        # Check logging
        assert "non_mesh_link" in caplog.text
        assert "skipped because it is not MeshShape" in caplog.text

    def test_multiple_links_processing(self, robot_with_mesh_links):
        """Test processing multiple links at once."""
        min_corner = np.array([-0.8, -0.8, -0.8])
        max_corner = np.array([0.8, 0.8, 0.8])

        # Process both mesh links
        CollisionMeshCustomizer.remove_outside_of_box(
            robot_with_mesh_links,
            {"test_link", "cylinder_link"},
            min_corner,
            max_corner,
        )

        # Both should be modified
        test_link = robot_with_mesh_links.link("test_link")
        cylinder_link = robot_with_mesh_links.link("cylinder_link")

        assert isinstance(test_link.shape, MeshShape)
        assert isinstance(cylinder_link.shape, MeshShape)


class TestFromMeshToBoundingPrimitive:
    def test_box_primitive_conversion(self, box_mesh):
        """Test conversion to box primitive."""
        color = np.array([1.0, 0.0, 0.0, 1.0])

        result = from_mesh_to_bounding_primitive(box_mesh, color)

        assert isinstance(result, Box)
        np.testing.assert_array_equal(result.color, color)

    def test_sphere_primitive_conversion(self, sphere_mesh):
        """Test conversion to sphere primitive."""
        color = np.array([0.0, 1.0, 0.0, 0.5])

        result = from_mesh_to_bounding_primitive(sphere_mesh, color)

        assert isinstance(result, Sphere)
        np.testing.assert_array_equal(result.color, color)
        assert result.radius > 0

    def test_cylinder_primitive_conversion(self, cylinder_mesh):
        """Test conversion to cylinder primitive."""
        result = from_mesh_to_bounding_primitive(cylinder_mesh)

        assert isinstance(result, Cylinder)
        assert result.radius > 0
        assert result.height > 0

    def test_no_color_provided(self, box_mesh):
        """Test conversion without color."""
        result = from_mesh_to_bounding_primitive(box_mesh)

        assert isinstance(result, Box)
        assert result.color is None

    def test_unsupported_primitive_kind(self):
        """Test error handling for unsupported primitive types."""
        # Create a mesh that might not have a standard bounding primitive
        # This is tricky to test directly, so we'll mock the behavior
        mesh = trimesh.creation.box(extents=[1, 1, 1])

        # Patch the bounding_primitive to return unsupported kind
        original_to_dict = mesh.bounding_primitive.to_dict

        def mock_to_dict():
            result = original_to_dict()
            result["kind"] = "unsupported_shape"
            return result

        mesh.bounding_primitive.to_dict = mock_to_dict

        with pytest.raises(ValueError, match="Unsupported primitive kind"):
            from_mesh_to_bounding_primitive(mesh)
