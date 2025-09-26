# type: ignore
from typing import Literal

try:
    import k3d
except ImportError:
    raise ImportError("k3d is not installed. Please install it via 'pip install k3d'")
import trimesh
import numpy as np

from linkmotion.visual.base import _get_or_create_plot, rgba_to_hex
from linkmotion.typing.numpy import Vector3s, RGBA0to1s


class MeshVisualizer:
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @staticmethod
    def vertices(
        vertices: Vector3s,
        indices: np.ndarray[tuple[int, Literal[3]], np.dtype[np.int32]],
        plot: k3d.Plot | None = None,
        normals: Vector3s | None = None,
        colors: RGBA0to1s | None = None,
        opacity: float | None = None,
        name: str | None = None,
        side: Literal["front", "back", "double"] = "front",
    ) -> k3d.Plot:
        plot = _get_or_create_plot(plot)

        if normals is None:
            normals = []
            flat_shading = True
        else:
            flat_shading = False

        if opacity is None:
            opacity = (
                float(np.max(colors[:, 3]))
                if colors is not None and len(colors) > 0
                else 1.0
            )

        if colors is None:
            colors = []
        else:
            colors = rgba_to_hex(colors)

        plot += k3d.mesh(
            vertices=np.ascontiguousarray(vertices, np.float32),
            indices=np.ascontiguousarray(indices, np.uint32),
            normals=np.ascontiguousarray(normals, np.float32)
            if normals is not None
            else None,
            colors=np.ascontiguousarray(colors, np.uint32)
            if colors is not None
            else None,
            flat_shading=flat_shading,
            opacity=opacity,
            name=name,
            side=side,
        )

        return plot

    @staticmethod
    def trimesh(
        mesh: trimesh.Trimesh,
        plot: k3d.Plot | None = None,
        opacity: float | None = None,
        name: str | None = None,
        side: Literal["front", "back", "double"] = "front",
    ) -> k3d.Plot:
        vertices = mesh.vertices
        indices = mesh.faces
        normals = mesh.vertex_normals
        colors = mesh.visual.vertex_colors

        if colors is not None:
            colors = colors / 255.0

        return MeshVisualizer.vertices(
            vertices, indices, plot, normals, colors, opacity, name, side
        )
