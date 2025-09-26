"""Simple example demonstrating linkmotion.modeling features.

This example shows how to use mesh operations like clipping,
sweeping, and rotation overlap with trimesh objects.
"""

import numpy as np
import trimesh

from linkmotion.modeling.remove import remove_outside_of_box
from linkmotion.modeling.sweep import sweep_trimesh, rotate_overlap_trimesh


def main() -> None:
    """Demonstrate modeling operations on a simple cube mesh."""
    # Create a simple cube mesh
    cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    print(f"Original cube vertices: {len(cube.vertices)}, faces: {len(cube.faces)}")

    # Example 1: Remove parts outside a bounding box
    min_corner = np.array([-0.5, -0.5, -0.5])
    max_corner = np.array([0.5, 0.5, 0.5])
    clipped_cube = remove_outside_of_box(cube, min_corner, max_corner)
    print(
        f"Clipped cube vertices: {len(clipped_cube.vertices)}, faces: {len(clipped_cube.faces)}"
    )

    # Example 2: Sweep mesh by translation
    translate_vector = np.array([0.0, 0.0, 1.0])
    swept_cube = sweep_trimesh(cube, translate_vector)
    print(
        f"Swept cube vertices: {len(swept_cube.vertices)}, faces: {len(swept_cube.faces)}"
    )

    # Example 3: Create overlapping rotated copies
    center = np.array([0.0, 0.0, 0.0])
    rotation_axis = np.array([0.0, 0.0, 1.0])  # Z-axis
    delta_angle = np.pi / 6  # 30 degrees
    initial_angle = 0.0
    num_copies = 3

    rotated_mesh = rotate_overlap_trimesh(
        cube, center, rotation_axis, delta_angle, initial_angle, num_copies
    )
    print(
        f"Rotated overlapping mesh vertices: {len(rotated_mesh.vertices)}, faces: {len(rotated_mesh.faces)}"
    )

    print("Modeling operations completed successfully!")


if __name__ == "__main__":
    main()
