import numpy as np
from typing import Literal, TypeAlias

# Requires Python 3.9+ and NumPy 1.21+

# --- Vector (3D) ---
Vector3: TypeAlias = np.ndarray[tuple[Literal[3]], np.dtype[np.float64]]
Vector3s: TypeAlias = np.ndarray[tuple[int, Literal[3]], np.dtype[np.float64]]
Vertices: TypeAlias = Vector3s
Normals: TypeAlias = Vector3s

Indices: TypeAlias = np.ndarray[tuple[int, Literal[3]], np.dtype[np.int64]]

# --- Vector (4D) ---
Vector4: TypeAlias = np.ndarray[tuple[Literal[4]], np.dtype[np.float64]]
RGBA0to1: TypeAlias = Vector4
RGBA255: TypeAlias = np.ndarray[tuple[Literal[4]], np.dtype[np.uint8]]

Vector4s: TypeAlias = np.ndarray[tuple[int, Literal[4]], np.dtype[np.float64]]
RGBA0to1s: TypeAlias = Vector4s
RGBA255s: TypeAlias = np.ndarray[tuple[int, Literal[4]], np.dtype[np.uint8]]

# --- Matrix ---
Matrix3x3: TypeAlias = np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float64]]
Matrix4x4: TypeAlias = np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.float64]]
