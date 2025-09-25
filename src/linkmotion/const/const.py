"""Physical and numerical constants for robotics applications.

This module contains commonly used constants in robotics applications,
including numerical tolerance values, mesh processing parameters, and
physical constants.
"""

import numpy as np
from typing import Final

# Numerical tolerance constants
SMALL_NUM: Final[float] = 1.0e-10
"""Small numerical tolerance for floating-point comparisons.

Used to handle floating-point precision issues when checking if values
are effectively zero or when comparing floating-point numbers for equality.
Typical use: `abs(value) < SMALL_NUM` instead of `value == 0.0`.
"""

LARGE_NUM: Final[float] = 1.0e10
"""Large numerical value representing "infinity" in joint limits.

Used as default upper/lower bounds for joint limits when no specific
limits are defined. This represents an effectively unlimited range
for joint motion without using actual infinity values.
"""

# Mesh processing constants
MESH_ACCURACY: Final[float] = 1.0
"""Spatial accuracy threshold for mesh processing operations (in meters).

Defines the resolution for mesh simplification, collision detection,
and geometric operations. Smaller values provide higher accuracy but
increase computational cost. Value of 1.0 meter is suitable for
coarse-grained robot-scale operations.
"""

MESH_ANGULAR_ACCURACY: Final[float] = float(np.radians(60.0))
"""Angular accuracy threshold for mesh processing operations (in radians).

Defines the angular resolution for mesh operations involving rotations
or surface normal calculations. Set to 60 degrees (π/3 radians) as a
reasonable balance between accuracy and performance for robotics applications.
"""
