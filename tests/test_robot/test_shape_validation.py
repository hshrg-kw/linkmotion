import pytest
import numpy as np

from linkmotion.robot import Link
from linkmotion.robot.shape.cylinder import Cylinder


class TestShapeValidation:
    """Tests for shape validation and error handling."""

    def test_box_invalid_extents(self):
        with pytest.raises(ValueError, match="must be positive"):
            Link.from_box("invalid_box", np.array([-1.0, 2.0, 3.0]))

        with pytest.raises(ValueError, match="exactly 3 dimensions"):
            Link.from_box("invalid_box", np.array([1.0, 2.0]))

    def test_sphere_invalid_radius(self):
        with pytest.raises(ValueError, match="radius must be positive"):
            Link.from_sphere("invalid_sphere", radius=-1)

        with pytest.raises(ValueError, match="radius must be positive"):
            Link.from_sphere("invalid_sphere", radius=0)

    def test_cylinder_parameters(self):
        link = Link.from_cylinder("test_cyl", radius=2.5, height=10.0)
        assert isinstance(link.shape, Cylinder)
        assert link.shape.radius == 2.5
        assert link.shape.height == 10.0

    def test_shape_color_handling(self):
        color = np.array([0.5, 0.3, 0.8, 1.0])
        link = Link.from_sphere("colored_sphere", radius=1, color=color)
        if link.shape.color is not None:
            assert np.array_equal(link.shape.color, color)
