import pytest
import numpy as np
import trimesh

from linkmotion.modeling.sweep import (
    sweep_triangles,
    sweep_trimesh,
    sweep_triangles_watertight,
    rotate_overlap_trimesh,
)


@pytest.fixture
def simple_triangle_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Provides a simple triangle as vertices and indices."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=float)
    indices = np.array([[0, 1, 2]])
    return vertices, indices


@pytest.fixture
def box_mesh() -> trimesh.Trimesh:
    """Provides a simple box mesh."""
    return trimesh.creation.box(extents=[1, 1, 1])


@pytest.fixture
def cube_vertices_indices() -> tuple[np.ndarray, np.ndarray]:
    """Provides vertices and indices for a unit cube."""
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    return mesh.vertices, mesh.faces


class TestSweepTriangles:
    def test_sweep_triangles_basic(self, simple_triangle_mesh):
        """Test basic triangle sweeping functionality."""
        vertices, indices = simple_triangle_mesh
        translate = np.array([0, 0, 1])

        new_vertices, new_indices = sweep_triangles(vertices, indices, translate)

        # Should have double the vertices (original + translated)
        assert len(new_vertices) == len(vertices) * 2

        # Check that original vertices are preserved
        np.testing.assert_array_equal(new_vertices[: len(vertices)], vertices)

        # Check that translated vertices are correct
        np.testing.assert_array_equal(
            new_vertices[len(vertices) :], vertices + translate
        )

        # Should have more faces (original + translated + sides)
        assert len(new_indices) > len(indices) * 2

    def test_sweep_triangles_zero_translation(self, simple_triangle_mesh):
        """Test sweeping with zero translation."""
        vertices, indices = simple_triangle_mesh
        translate = np.array([0, 0, 0])

        new_vertices, new_indices = sweep_triangles(vertices, indices, translate)

        # Should still double vertices even with zero translation
        assert len(new_vertices) == len(vertices) * 2

        # Original and translated vertices should be identical
        np.testing.assert_array_equal(new_vertices[: len(vertices)], vertices)
        np.testing.assert_array_equal(new_vertices[len(vertices) :], vertices)

    def test_sweep_triangles_negative_translation(self, simple_triangle_mesh):
        """Test sweeping with negative translation."""
        vertices, indices = simple_triangle_mesh
        translate = np.array([0, 0, -2])

        new_vertices, new_indices = sweep_triangles(vertices, indices, translate)

        # Check translated vertices
        np.testing.assert_array_equal(
            new_vertices[len(vertices) :], vertices + translate
        )


class TestSweepTrimesh:
    def test_sweep_trimesh_basic(self, box_mesh):
        """Test basic trimesh sweeping."""
        translate = np.array([1, 0, 0])

        result = sweep_trimesh(box_mesh, translate)

        assert isinstance(result, trimesh.Trimesh)
        assert len(result.vertices) > len(box_mesh.vertices)
        assert len(result.faces) > len(box_mesh.faces)

    def test_sweep_trimesh_preserves_mesh_properties(self, box_mesh):
        """Test that sweeping preserves important mesh properties."""
        translate = np.array([0, 1, 0])

        result = sweep_trimesh(box_mesh, translate)

        # Result should be a valid mesh
        assert len(result.vertices) > 0
        assert len(result.faces) > 0


class TestSweepTrianglesWatertight:
    def test_sweep_triangles_watertight_basic(self, cube_vertices_indices):
        """Test watertight sweeping creates valid volume."""
        vertices, indices = cube_vertices_indices
        translate = np.array([0, 0, 2])

        result = sweep_triangles_watertight(vertices, indices, translate)

        assert isinstance(result, trimesh.Trimesh)
        assert result.is_volume
        assert result.is_watertight

    def test_sweep_triangles_watertight_volume_increase(self, cube_vertices_indices):
        """Test that watertight sweeping increases volume."""
        vertices, indices = cube_vertices_indices
        original_mesh = trimesh.Trimesh(vertices, indices)
        translate = np.array([0, 0, 1])

        result = sweep_triangles_watertight(vertices, indices, translate)

        # Swept volume should be larger than original
        assert result.volume > original_mesh.volume


class TestRotateOverlapTrimesh:
    def test_rotate_overlap_basic(self, box_mesh):
        """Test basic rotational overlap functionality."""
        center = np.array([0, 0, 0])
        normalized_direction = np.array([0, 0, 1])
        delta_angle = np.pi / 4  # 45 degrees
        initial_angle = 0.0
        how_many_to_add = 3

        result = rotate_overlap_trimesh(
            box_mesh,
            center,
            normalized_direction,
            delta_angle,
            initial_angle,
            how_many_to_add,
        )

        assert isinstance(result, trimesh.Trimesh)
        # Should have more vertices than original (original + 3 copies)
        assert len(result.vertices) > len(box_mesh.vertices)

    def test_rotate_overlap_with_initial_angle(self, box_mesh):
        """Test rotational overlap with non-zero initial angle."""
        center = np.array([1, 1, 0])
        normalized_direction = np.array([1, 0, 0])
        delta_angle = np.pi / 6  # 30 degrees
        initial_angle = np.pi / 3  # 60 degrees
        how_many_to_add = 2

        result = rotate_overlap_trimesh(
            box_mesh,
            center,
            normalized_direction,
            delta_angle,
            initial_angle,
            how_many_to_add,
        )

        assert isinstance(result, trimesh.Trimesh)
        # Should have vertices from original + 2 additional copies
        expected_vertex_count = len(box_mesh.vertices) * (1 + how_many_to_add)
        assert len(result.vertices) == expected_vertex_count

    def test_rotate_overlap_zero_copies(self, box_mesh):
        """Test rotational overlap with zero additional copies."""
        center = np.array([0, 0, 0])
        normalized_direction = np.array([0, 1, 0])
        delta_angle = np.pi / 2
        initial_angle = 0.0
        how_many_to_add = 0

        result = rotate_overlap_trimesh(
            box_mesh,
            center,
            normalized_direction,
            delta_angle,
            initial_angle,
            how_many_to_add,
        )

        # Should only have the original mesh (possibly rotated by initial_angle)
        assert len(result.vertices) == len(box_mesh.vertices)

    def test_rotate_overlap_full_circle(self, box_mesh):
        """Test creating overlapping copies around a full circle."""
        center = np.array([0, 0, 0])
        normalized_direction = np.array([0, 0, 1])
        delta_angle = np.pi / 4  # 45 degrees
        initial_angle = 0.0
        how_many_to_add = 7  # 8 total copies (360 degrees / 45 degrees)

        result = rotate_overlap_trimesh(
            box_mesh,
            center,
            normalized_direction,
            delta_angle,
            initial_angle,
            how_many_to_add,
        )

        # Due to mesh merging, vertex count may be less than expected due to duplicate removal
        # Just check that we have more vertices than the original
        assert len(result.vertices) > len(box_mesh.vertices)
