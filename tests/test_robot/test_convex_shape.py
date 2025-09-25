import pytest
import numpy as np
import trimesh

from linkmotion.robot import Link
from linkmotion.robot.shape.convex import ConvexShape
from linkmotion.transform import Transform


class TestConvexShape:
    """Tests for ConvexShape class."""

    def test_convex_shape_from_box_mesh(self):
        """Test creating ConvexShape from a box mesh."""
        collision_mesh = trimesh.creation.box(extents=[2, 3, 4])
        convex_shape = ConvexShape(collision_mesh)

        assert isinstance(convex_shape.collision_mesh, trimesh.Trimesh)
        assert len(convex_shape.collision_mesh.vertices) == 8  # Box has 8 vertices
        assert len(convex_shape.collision_mesh.faces) == 12  # Box has 12 faces
        assert convex_shape.collision_primitive is not None

    def test_convex_shape_with_separate_visual_mesh(self):
        """Test ConvexShape with different collision and visual meshes."""
        collision_mesh = trimesh.creation.box(extents=[1, 1, 1])
        visual_mesh = trimesh.creation.icosphere(radius=0.5)

        convex_shape = ConvexShape(collision_mesh, visual_mesh)

        # Collision mesh should be the box
        assert len(convex_shape.collision_mesh.vertices) == 8
        assert len(convex_shape.collision_mesh.faces) == 12

        # Visual mesh should be the icosphere
        assert (
            len(convex_shape.visual_mesh.vertices) > 8
        )  # Icosphere has more vertices than box

    def test_convex_shape_with_transform_and_color(self):
        """Test ConvexShape with custom transform and color."""
        collision_mesh = trimesh.creation.cylinder(radius=1, height=2)
        transform = Transform(translate=np.array([1.0, 2.0, 3.0]))
        color = np.array([1.0, 0.5, 0.0, 1.0])

        convex_shape = ConvexShape(
            collision_mesh, default_transform=transform, color=color
        )

        assert np.array_equal(convex_shape.default_transform.position, [1, 2, 3])
        if convex_shape.color is not None:
            assert np.array_equal(convex_shape.color, color)

    def test_convex_shape_invalid_collision_mesh_type(self):
        """Test ConvexShape with invalid collision mesh type."""
        with pytest.raises(ValueError, match="must be a trimesh.Trimesh"):
            ConvexShape("not_a_mesh")

    def test_convex_shape_empty_vertices(self):
        """Test ConvexShape with mesh that has no vertices."""
        empty_mesh = trimesh.Trimesh(vertices=[], faces=[])
        with pytest.raises(ValueError, match="must have vertices and faces"):
            ConvexShape(empty_mesh)

    def test_convex_shape_no_faces(self):
        """Test ConvexShape with mesh that has vertices but no faces."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        mesh_no_faces = trimesh.Trimesh(vertices=vertices, faces=[])
        with pytest.raises(ValueError, match="must have vertices and faces"):
            ConvexShape(mesh_no_faces)

    def test_convex_shape_repr(self):
        """Test ConvexShape string representation."""
        collision_mesh = trimesh.creation.box(extents=[1, 1, 1])
        convex_shape = ConvexShape(collision_mesh)

        repr_str = repr(convex_shape)
        assert "ConvexShape" in repr_str
        assert "vertices=8" in repr_str
        assert "faces=12" in repr_str

    def test_convex_shape_copy(self):
        """Test ConvexShape copy method."""
        collision_mesh = trimesh.creation.box(extents=[2, 2, 2])
        visual_mesh = trimesh.creation.icosphere(radius=1)
        transform = Transform(translate=np.array([5.0, 0.0, 0.0]))
        color = np.array([0.0, 1.0, 0.0, 1.0])

        original = ConvexShape(collision_mesh, visual_mesh, transform, color)
        copied = original.copy()

        # Should be different objects
        assert copied is not original
        assert copied.collision_mesh is not original.collision_mesh
        assert copied.visual_mesh is not original.visual_mesh

        # But should have same properties
        assert len(copied.collision_mesh.vertices) == len(
            original.collision_mesh.vertices
        )
        assert len(copied.collision_mesh.faces) == len(original.collision_mesh.faces)
        assert np.array_equal(
            copied.default_transform.position, original.default_transform.position
        )
        if original.color is not None and copied.color is not None:
            assert np.array_equal(copied.color, original.color)

    def test_convex_shape_from_other(self):
        """Test creating ConvexShape using from_other method."""
        collision_mesh = trimesh.creation.cylinder(radius=1, height=3)
        original = ConvexShape(collision_mesh)

        transform = Transform(translate=np.array([10.0, 20.0, 30.0]))
        transformed = ConvexShape.from_other(original, transform)

        # Should have same collision mesh properties
        assert len(transformed.collision_mesh.vertices) == len(
            original.collision_mesh.vertices
        )
        assert len(transformed.collision_mesh.faces) == len(
            original.collision_mesh.faces
        )

        # But different transform
        expected_pos = transform.apply(original.default_transform).position
        assert np.allclose(transformed.default_transform.position, expected_pos)

    def test_convex_shape_collision_primitive_creation(self):
        """Test that FCL collision primitive is created correctly."""
        collision_mesh = trimesh.creation.icosphere(radius=2, subdivisions=1)
        convex_shape = ConvexShape(collision_mesh)

        # Should create FCL Convex object
        import fcl

        assert isinstance(convex_shape.collision_primitive, fcl.Convex)

    def test_convex_shape_transformed_collision_object(self):
        """Test creating transformed collision object."""
        collision_mesh = trimesh.creation.box(extents=[1, 1, 1])
        convex_shape = ConvexShape(collision_mesh)

        # Test with no transform
        collision_obj = convex_shape.transformed_collision_object()
        assert collision_obj is not None

        # Test with transform
        transform = Transform(translate=np.array([1.0, 2.0, 3.0]))
        transformed_collision_obj = convex_shape.transformed_collision_object(transform)
        assert transformed_collision_obj is not None

    def test_convex_shape_transformed_visual_mesh(self):
        """Test creating transformed visual mesh."""
        collision_mesh = trimesh.creation.box(extents=[2, 2, 2])
        convex_shape = ConvexShape(collision_mesh)

        # Test with no transform
        visual_mesh = convex_shape.transformed_visual_mesh()
        assert isinstance(visual_mesh, trimesh.Trimesh)

        # Test with transform
        transform = Transform(translate=np.array([5.0, 0.0, 0.0]))
        transformed_visual_mesh = convex_shape.transformed_visual_mesh(transform)
        assert isinstance(transformed_visual_mesh, trimesh.Trimesh)

        # Visual mesh should be different after transformation
        assert not np.allclose(visual_mesh.vertices, transformed_visual_mesh.vertices)

    def test_convex_shape_complex_mesh(self):
        """Test ConvexShape with a more complex mesh."""
        # Create a more complex mesh (a torus)
        collision_mesh = trimesh.creation.torus(major_radius=2, minor_radius=0.5)
        # Use different mesh type for visual (icosphere vs torus)
        visual_mesh = trimesh.creation.icosphere(radius=2.5, subdivisions=3)

        convex_shape = ConvexShape(collision_mesh, visual_mesh)

        # Should handle complex meshes
        assert len(convex_shape.collision_mesh.vertices) > 0
        assert len(convex_shape.collision_mesh.faces) > 0
        assert convex_shape.collision_primitive is not None

        # Visual mesh should be different from collision mesh
        assert len(convex_shape.visual_mesh.vertices) != len(
            convex_shape.collision_mesh.vertices
        )

    def test_convex_shape_with_default_visual_mesh(self):
        """Test ConvexShape when visual_mesh is None (uses collision_mesh)."""
        collision_mesh = trimesh.creation.icosphere(radius=1)
        convex_shape = ConvexShape(collision_mesh)  # visual_mesh=None by default

        # Visual mesh should be a copy of collision mesh
        assert len(convex_shape.visual_mesh.vertices) == len(
            convex_shape.collision_mesh.vertices
        )
        assert len(convex_shape.visual_mesh.faces) == len(
            convex_shape.collision_mesh.faces
        )

        # But should be different objects
        assert convex_shape.visual_mesh is not convex_shape.collision_mesh


class TestLinkWithConvexShape:
    """Test Link integration with ConvexShape."""

    def test_link_with_convex_shape(self):
        """Test creating Link with ConvexShape."""
        collision_mesh = trimesh.creation.box(extents=[1, 2, 3])
        convex_shape = ConvexShape(collision_mesh)
        link = Link("convex_link", convex_shape)

        assert link.name == "convex_link"
        assert isinstance(link.shape, ConvexShape)
        assert link.shape is convex_shape

    def test_link_convex_collision_object(self):
        """Test Link collision object with ConvexShape."""
        collision_mesh = trimesh.creation.cylinder(radius=1, height=2)
        convex_shape = ConvexShape(collision_mesh)
        link = Link("convex_cylinder", convex_shape)

        collision_obj = link.collision_object()
        assert collision_obj is not None

        # Test with transform
        transform = Transform(translate=np.array([1.0, 0.0, 0.0]))
        transformed_collision_obj = link.collision_object(transform)
        assert transformed_collision_obj is not None

    def test_link_convex_visual_mesh(self):
        """Test Link visual mesh with ConvexShape."""
        collision_mesh = trimesh.creation.icosphere(radius=1.5)
        convex_shape = ConvexShape(collision_mesh)
        link = Link("convex_sphere", convex_shape)

        visual_mesh = link.visual_mesh()
        assert isinstance(visual_mesh, trimesh.Trimesh)

        # Test with transform
        transform = Transform(translate=np.array([0.0, 2.0, 0.0]))
        transformed_visual_mesh = link.visual_mesh(transform)
        assert isinstance(transformed_visual_mesh, trimesh.Trimesh)
        assert not np.allclose(visual_mesh.vertices, transformed_visual_mesh.vertices)
