"""
Basic Transform Usage Example

This example demonstrates the fundamental operations of the Transform class,
including creation, manipulation, and application to geometric objects.
"""

import logging
import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh

from linkmotion.transform import Transform

# Set up logging to see transform operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate basic Transform operations."""
    logger.info("=== Basic Transform Usage Example ===")

    # 1. Creating transforms
    logger.info("\n1. Creating transforms:")

    # Identity transform (no rotation, no translation)
    identity = Transform()
    logger.info(f"Identity transform: {identity.is_identity()}")

    # Translation-only transform
    translation = Transform(translate=np.array([2.0, 3.0, 1.0]))
    logger.info(f"Translation: {translation.position}")

    # Rotation-only transform (90 degrees around Z-axis)
    rotation_z = R.from_euler("z", 90, degrees=True)
    rotation_transform = Transform(rotate=rotation_z)
    logger.info(f"Rotation matrix:\n{rotation_transform.rotation.as_matrix()}")

    # Combined rotation and translation
    combined = Transform(
        rotate=R.from_euler("xyz", [30, 45, 60], degrees=True),
        translate=np.array([1.0, 2.0, 3.0]),
    )
    logger.info(f"Combined transform position: {combined.position}")

    # 2. Transform operations
    logger.info("\n2. Transform operations:")

    # Copy a transform
    copied_transform = combined.copy()
    logger.info(f"Copied transform equals original: {copied_transform == combined}")

    # In-place modifications
    combined.translate(np.array([1.0, 0.0, 0.0]))  # Add translation
    logger.info(f"After additional translation: {combined.position}")

    combined.rotate(R.from_euler("z", 45, degrees=True))  # Add rotation
    logger.info("Applied additional 45° Z rotation")

    # 3. Applying transforms to vectors
    logger.info("\n3. Applying transforms to vectors:")

    # Single vector
    point = np.array([1.0, 0.0, 0.0])
    transformed_point = translation.apply(point)
    logger.info(f"Original point: {point}")
    logger.info(f"Transformed point: {transformed_point}")

    # Multiple vectors
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    transformed_points = rotation_transform.apply(points)
    logger.info(f"Original points:\n{points}")
    logger.info(f"Rotated points:\n{transformed_points}")

    # 4. Transform composition
    logger.info("\n4. Transform composition:")

    # Apply one transform to another
    t1 = Transform(translate=np.array([1.0, 0.0, 0.0]))
    t2 = Transform(translate=np.array([0.0, 1.0, 0.0]))

    # Two ways to compose transforms
    composed1 = t1.apply(t2)  # Apply t1 to t2
    composed2 = t2.transformed(t1)  # Apply t1 to t2 (same result)

    logger.info(f"Composed transform 1: {composed1.position}")
    logger.info(f"Composed transform 2: {composed2.position}")
    logger.info(f"Results are equal: {composed1 == composed2}")

    # 5. Working with 4x4 matrices
    logger.info("\n5. Matrix conversion:")

    matrix_4x4 = combined.to_4x4()
    logger.info(f"4x4 transformation matrix:\n{matrix_4x4}")

    # Convert back from matrix
    from_matrix = Transform.from_4x4(matrix_4x4)
    logger.info(f"Reconstructed from matrix equals original: {from_matrix == combined}")

    # 6. Working with meshes
    logger.info("\n6. Applying transforms to 3D meshes:")

    # Create a simple cube mesh
    cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    logger.info(f"Original cube center: {cube.centroid}")

    # Transform the mesh
    transform = Transform(
        rotate=R.from_euler("z", 45, degrees=True), translate=np.array([2.0, 2.0, 0.0])
    )
    transformed_cube = transform.apply(cube)
    logger.info(f"Transformed cube center: {transformed_cube.centroid}")

    # 7. Practical robotics example
    logger.info("\n7. Robotics example - Robot arm pose:")

    # Base frame
    base_transform = Transform()

    # Shoulder joint (rotate around Z, translate up)
    shoulder_transform = Transform(
        rotate=R.from_euler("z", 30, degrees=True),
        translate=np.array([0.0, 0.0, 0.3]),  # 30cm high
    )

    # Elbow joint (rotate around Y, translate forward)
    elbow_transform = Transform(
        rotate=R.from_euler("y", -45, degrees=True),
        translate=np.array([0.4, 0.0, 0.0]),  # 40cm forward
    )

    # Calculate end-effector position
    shoulder_world = base_transform.apply(shoulder_transform)
    end_effector_world = shoulder_world.apply(elbow_transform)

    logger.info(f"End-effector position: {end_effector_world.position}")
    logger.info(
        f"End-effector orientation (Euler XYZ): {end_effector_world.rotation.as_euler('xyz', degrees=True)}"
    )

    logger.info("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
