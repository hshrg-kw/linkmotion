import trimesh

from linkmotion.typing.numpy import Vector3


def remove_outside_of_box(
    mesh: trimesh.Trimesh, min_: Vector3, max_: Vector3
) -> trimesh.Trimesh:
    """Remove parts of mesh that are outside the specified bounding box.

    Args:
        mesh: Input triangular mesh to clip.
        min_: Minimum corner of the bounding box.
        max_: Maximum corner of the bounding box.

    Returns:
        Clipped mesh with parts outside the box removed.
    """
    clipped_mesh = mesh.slice_plane(plane_origin=min_, plane_normal=[1, 0, 0])
    if clipped_mesh is None:
        return trimesh.Trimesh()

    clipped_mesh = clipped_mesh.slice_plane(plane_origin=min_, plane_normal=[0, 1, 0])
    if clipped_mesh is None:
        return trimesh.Trimesh()

    clipped_mesh = clipped_mesh.slice_plane(plane_origin=min_, plane_normal=[0, 0, 1])
    if clipped_mesh is None:
        return trimesh.Trimesh()

    clipped_mesh = clipped_mesh.slice_plane(plane_origin=max_, plane_normal=[-1, 0, 0])
    if clipped_mesh is None:
        return trimesh.Trimesh()

    clipped_mesh = clipped_mesh.slice_plane(plane_origin=max_, plane_normal=[0, -1, 0])
    if clipped_mesh is None:
        return trimesh.Trimesh()

    clipped_mesh = clipped_mesh.slice_plane(plane_origin=max_, plane_normal=[0, 0, -1])
    if clipped_mesh is None:
        return trimesh.Trimesh()

    return trimesh.Trimesh(vertices=clipped_mesh.vertices, faces=clipped_mesh.faces)
