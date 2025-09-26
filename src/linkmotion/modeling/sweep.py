import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from linkmotion.transform import Transform
from linkmotion.typing.numpy import Vertices, Indices, Vector3


def sweep_triangles(
    vertices: Vertices,
    indices: Indices,
    translate: Vector3,
) -> tuple[Vertices, Indices]:
    """Sweep triangular mesh by translation, creating side faces.

    Args:
        vertices: Array of vertex coordinates.
        indices: Array of triangle face indices.
        translate: Translation vector for sweeping.

    Returns:
        Tuple of (new_vertices, new_indices) for the swept mesh.
    """
    # remove duplicated vertices
    mesh = trimesh.Trimesh(vertices, indices)
    vertices = mesh.vertices
    indices = mesh.faces

    new_vertices = vertices + translate
    vertex_num = len(vertices)
    new_indices = indices + vertex_num

    unique_edges = mesh.edges_unique

    # inside outside is not defined
    side_indices1 = np.hstack((unique_edges, unique_edges[:, 0:1]))
    side_indices1 += np.array([0, vertex_num, vertex_num])
    side_indices2 = np.hstack((unique_edges, unique_edges[:, 1:2]))
    side_indices2 += np.array([0, 0, vertex_num])

    # above numpy implementation is identical with below for-loop
    # side_indices = []
    # for i, e in enumerate(unique_edges):
    #     side_indices += [
    #         [e[0], e[1] + vertex_num, e[0] + vertex_num],
    #         [e[0], e[1], e[1] + vertex_num],
    #     ]
    # side_indices = np.array(side_indices)
    # return (
    #     np.vstack((vertices, new_vertices)),
    #     np.vstack((indices, new_indices, side_indices)),
    # )

    return (
        np.vstack((vertices, new_vertices)),
        np.vstack((indices, new_indices, side_indices1, side_indices2)),
    )


def sweep_trimesh(mesh: trimesh.Trimesh, translate: Vector3) -> trimesh.Trimesh:
    """Sweep a trimesh by translation.

    Args:
        mesh: Input triangular mesh.
        translate: Translation vector for sweeping.

    Returns:
        New trimesh representing the swept geometry.
    """
    new_vertices, new_indices = sweep_triangles(mesh.vertices, mesh.faces, translate)
    return trimesh.Trimesh(new_vertices, new_indices)


def sweep_triangles_watertight(
    vertices: Vertices, indices: Indices, translate: Vector3
) -> trimesh.Trimesh:
    """Create a watertight swept mesh by extruding triangles.

    Args:
        vertices: Array of vertex coordinates.
        indices: Array of triangle face indices.
        translate: Translation vector for sweeping.

    Returns:
        Watertight trimesh representing the swept volume.
    """
    mesh0 = trimesh.Trimesh(vertices, indices)
    mesh1 = trimesh.Trimesh(vertices + translate, indices)

    meshes = []
    for face in indices:
        v = np.vstack([vertices[face], vertices[face] + translate])
        f = np.array(
            [
                [2, 1, 0],
                [3, 4, 5],
                [0, 1, 3],
                [3, 1, 4],
                [1, 2, 4],
                [4, 2, 5],
                [2, 0, 5],
                [5, 0, 3],
            ]
        )
        mesh = trimesh.Trimesh(vertices=v, faces=f, validate=True)
        if mesh.is_volume:
            meshes.append(mesh)

    return trimesh.boolean.union([mesh0, mesh1] + meshes)


def rotate_overlap_trimesh(
    mesh: trimesh.Trimesh,
    center: Vector3,
    normalized_direction: Vector3,
    delta_angle: float,
    initial_angle: float,
    how_many_to_add: int,
) -> trimesh.Trimesh:
    """Create overlapping rotated copies of a mesh around an axis.

    Args:
        mesh: Input triangular mesh.
        center: Center point of rotation.
        normalized_direction: Normalized rotation axis vector.
        delta_angle: Angle increment between copies (radians).
        initial_angle: Starting rotation angle (radians).
        how_many_to_add: Number of additional rotated copies to create.

    Returns:
        Combined trimesh with all rotated copies.
    """
    to_center = Transform(translate=-center)
    rotate = R.from_rotvec(initial_angle * normalized_direction)
    rot_to_original = Transform(rotate, center)
    centered_mesh = to_center.apply(mesh)
    mesh = rot_to_original.apply(centered_mesh)

    vertices = mesh.vertices
    faces = mesh.faces

    for i in range(how_many_to_add):
        current_angle = initial_angle + delta_angle * (i + 1)
        rotate = R.from_rotvec(current_angle * normalized_direction)
        rot_to_original = Transform(rotate, center)
        new_mesh = rot_to_original.apply(centered_mesh)
        old_vertices_num = len(vertices)
        vertices = np.vstack((vertices, new_mesh.vertices))
        faces = np.vstack((faces, new_mesh.faces + old_vertices_num))
    return trimesh.Trimesh(vertices=vertices, faces=faces)
