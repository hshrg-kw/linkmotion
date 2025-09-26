import pytest
import numpy as np
import trimesh

from linkmotion.modeling.remove import remove_outside_of_box


@pytest.fixture
def box_mesh() -> trimesh.Trimesh:
    """Provides a unit box mesh centered at origin."""
    return trimesh.creation.box(extents=[2, 2, 2])


@pytest.fixture
def sphere_mesh() -> trimesh.Trimesh:
    """Provides a unit sphere mesh centered at origin."""
    return trimesh.creation.icosphere(radius=1.0)


@pytest.fixture
def cylinder_mesh() -> trimesh.Trimesh:
    """Provides a cylinder mesh."""
    return trimesh.creation.cylinder(radius=0.5, height=2.0)


class TestRemoveOutsideOfBox:
    def test_remove_outside_box_no_clipping(self, box_mesh):
        """Test when bounding box contains entire mesh."""
        min_corner = np.array([-2, -2, -2])
        max_corner = np.array([2, 2, 2])

        result = remove_outside_of_box(box_mesh, min_corner, max_corner)

        # Should return similar mesh since nothing is clipped
        assert isinstance(result, trimesh.Trimesh)
        assert not result.is_empty
        # Volume should be similar (allowing for small numerical differences)
        assert abs(result.volume - box_mesh.volume) < 0.1

    def test_remove_outside_box_complete_clipping(self, box_mesh):
        """Test when bounding box excludes entire mesh."""
        min_corner = np.array([5, 5, 5])
        max_corner = np.array([6, 6, 6])

        result = remove_outside_of_box(box_mesh, min_corner, max_corner)

        # Should return empty mesh
        assert isinstance(result, trimesh.Trimesh)
        assert result.is_empty or len(result.vertices) == 0

    def test_remove_outside_box_partial_clipping(self, box_mesh):
        """Test partial clipping of mesh."""
        # Clip half the box
        min_corner = np.array([-1, -1, -1])
        max_corner = np.array([0, 1, 1])

        result = remove_outside_of_box(box_mesh, min_corner, max_corner)

        assert isinstance(result, trimesh.Trimesh)
        if not result.is_empty:
            # Clipped volume should be less than original
            assert result.volume < box_mesh.volume
            # All vertices should be within the clipping box
            assert np.all(result.vertices >= min_corner - 1e-6)  # Small tolerance
            assert np.all(result.vertices <= max_corner + 1e-6)

    def test_remove_outside_box_sphere(self, sphere_mesh):
        """Test clipping a sphere mesh."""
        # Clip sphere to positive octant
        min_corner = np.array([0, 0, 0])
        max_corner = np.array([1, 1, 1])

        result = remove_outside_of_box(sphere_mesh, min_corner, max_corner)

        assert isinstance(result, trimesh.Trimesh)
        if not result.is_empty:
            # Volume should be less than original sphere
            assert result.volume < sphere_mesh.volume
            # All vertices should be in positive octant
            assert np.all(
                result.vertices >= -1e-6
            )  # Small tolerance for numerical errors

    def test_remove_outside_box_cylinder(self, cylinder_mesh):
        """Test clipping a cylinder mesh."""
        # Clip cylinder with a smaller box
        min_corner = np.array([-0.3, -0.3, -0.5])
        max_corner = np.array([0.3, 0.3, 0.5])

        result = remove_outside_of_box(cylinder_mesh, min_corner, max_corner)

        assert isinstance(result, trimesh.Trimesh)
        if not result.is_empty:
            # Volume should be less than original
            assert result.volume < cylinder_mesh.volume

    def test_remove_outside_box_edge_cases(self, box_mesh):
        """Test edge cases with very small or zero-size bounding boxes."""
        # Very small bounding box
        min_corner = np.array([-0.01, -0.01, -0.01])
        max_corner = np.array([0.01, 0.01, 0.01])

        result = remove_outside_of_box(box_mesh, min_corner, max_corner)

        assert isinstance(result, trimesh.Trimesh)
        # Should be very small or empty
        if not result.is_empty:
            assert result.volume < 0.1

    def test_remove_outside_box_inverted_bounds(self, box_mesh):
        """Test with inverted min/max bounds."""
        # Swapped min and max - should result in empty or very small mesh
        min_corner = np.array([1, 1, 1])
        max_corner = np.array([-1, -1, -1])

        result = remove_outside_of_box(box_mesh, min_corner, max_corner)

        assert isinstance(result, trimesh.Trimesh)
        # Should be empty since bounds are inverted
        assert result.is_empty or result.volume < 1e-6

    def test_remove_outside_box_single_plane_clip(self, box_mesh):
        """Test clipping with bounding box that clips only one side."""
        # Only clip the positive X side
        min_corner = np.array([-2, -2, -2])
        max_corner = np.array([0, 2, 2])

        result = remove_outside_of_box(box_mesh, min_corner, max_corner)

        assert isinstance(result, trimesh.Trimesh)
        if not result.is_empty:
            # Should have roughly half the volume
            assert 0.4 < result.volume / box_mesh.volume < 0.6
            # All X coordinates should be <= 0
            assert np.all(result.vertices[:, 0] <= 0.01)  # Small tolerance

    def test_remove_outside_box_preserves_mesh_validity(self, sphere_mesh):
        """Test that clipped mesh remains valid."""
        min_corner = np.array([-0.5, -0.5, -0.5])
        max_corner = np.array([0.5, 0.5, 0.5])

        result = remove_outside_of_box(sphere_mesh, min_corner, max_corner)

        if not result.is_empty:
            # Basic mesh validity checks
            assert len(result.vertices) > 0
            assert len(result.faces) > 0
            assert result.vertices.shape[1] == 3  # 3D vertices
            assert result.faces.shape[1] == 3  # Triangle faces
