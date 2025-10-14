# type: ignore
"""Mesh visualization utilities for rendering 3D geometric meshes.

This module provides capabilities for visualizing 3D meshes using k3d.
It supports rendering meshes from raw vertex/index data as well as
trimesh objects with optional colors, normals, and transparency.
"""

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
    """Static methods for visualizing 3D meshes in various formats.

    This class provides visualization capabilities for rendering 3D meshes
    from raw geometric data (vertices and indices) or from trimesh objects.
    Supports vertex colors, normals, transparency, and various rendering options.
    """

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
        """Visualizes a mesh from raw vertex and index data.

        This method renders a 3D mesh defined by vertices (3D points) and
        indices (triangular faces). Supports optional vertex normals for smooth
        shading, vertex colors, and various rendering options.

        Args:
            vertices: Array of 3D vertex positions with shape (N, 3).
            indices: Array of triangle face indices with shape (M, 3), where
                each row contains three indices into the vertices array.
            plot: Optional existing k3d.Plot to add the mesh to.
                If None, creates a new plot.
            normals: Optional array of vertex normals with shape (N, 3).
                If None, uses flat shading instead of smooth shading.
            colors: Optional array of vertex colors with shape (N, 4) in RGBA
                format with values in range [0, 1].
            opacity: Optional opacity value for the entire mesh (0.0 to 1.0).
                If None, uses the maximum alpha value from vertex colors,
                or 1.0 if no colors are provided.
            name: Optional name for the mesh object in the plot.
            side: Which side(s) of faces to render - "front", "back", or "double".

        Returns:
            The k3d.Plot object containing the visualized mesh.
        """
        plot = _get_or_create_plot(plot)

        # Determine shading mode based on normal availability
        if normals is None:
            normals = []
            flat_shading = True  # Use flat shading without normals
        else:
            flat_shading = False  # Use smooth shading with normals

        # Calculate opacity from vertex colors if not explicitly provided
        if opacity is None:
            opacity = (
                float(np.max(colors[:, 3]))
                if colors is not None and len(colors) > 0
                else 1.0
            )

        # Convert RGBA colors to hexadecimal format for k3d
        if colors is None:
            colors = []
        else:
            colors = rgba_to_hex(colors)

        # Add the mesh to the plot with all specified properties
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
        """Visualizes a trimesh.Trimesh object.

        This convenience method extracts vertex data, face indices, normals,
        and colors from a trimesh object and renders it using the vertices()
        method. Automatically handles color conversion from 8-bit (0-255)
        to normalized (0-1) format.

        Args:
            mesh: The trimesh.Trimesh object to visualize.
            plot: Optional existing k3d.Plot to add the mesh to.
                If None, creates a new plot.
            opacity: Optional opacity value for the mesh (0.0 to 1.0).
                If None, uses the alpha values from the mesh's vertex colors.
            name: Optional name for the mesh object in the plot.
            side: Which side(s) of faces to render - "front", "back", or "double".

        Returns:
            The k3d.Plot object containing the visualized mesh.
        """
        # Extract geometric data from the trimesh object
        vertices = mesh.vertices
        indices = mesh.faces
        normals = mesh.vertex_normals
        colors = mesh.visual.vertex_colors

        # Convert colors from 8-bit (0-255) to normalized (0-1) format
        if colors is not None:
            colors = colors / 255.0

        # Delegate to the vertices method for actual rendering
        return MeshVisualizer.vertices(
            vertices, indices, plot, normals, colors, opacity, name, side
        )
