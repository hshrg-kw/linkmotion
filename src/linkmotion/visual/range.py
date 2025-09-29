import plotly.graph_objects as go
import numpy as np


def plot_2d(
    mesh_grid: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    x_points: np.ndarray[tuple[int], np.dtype[np.float64]],
    y_points: np.ndarray[tuple[int], np.dtype[np.float64]],
    title: str = "2D Contour Plot",
    x_label: str = "X Axis",
    y_label: str = "Y Axis",
    z_label: str = "Z Axis",
    z_min: float | None = None,
    z_max: float | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    x, y = np.meshgrid(x_points, y_points)

    trace = go.Surface(
        x=x,
        y=y,
        z=mesh_grid.transpose(),
        colorscale="Blues",
        colorbar=dict(title=z_label),
        cmin=z_min,
        cmax=z_max,
    )

    fig = go.Figure(data=[trace])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
            xaxis_range=[x_min, x_max],
            yaxis_range=[y_min, y_max],
            zaxis_range=[z_min, z_max],
        ),
    )

    fig.show()


def plot_3d(
    mesh_grid: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    x_points: np.ndarray[tuple[int], np.dtype[np.float64]],
    y_points: np.ndarray[tuple[int], np.dtype[np.float64]],
    time_points: np.ndarray[tuple[int], np.dtype[np.float64]],
    title: str = "3D Contour Plot",
    x_label: str = "X Axis",
    y_label: str = "Y Axis",
    z_label: str = "Z Axis",
    time_label: str = "Time Axis",
    z_min: float | None = None,
    z_max: float | None = None,
):
    x, y = np.meshgrid(x_points, y_points)

    time_steps = len(time_points)
    z_data = np.transpose(mesh_grid, (2, 1, 0))
    fig = go.Figure()
    for t, z in enumerate(z_data):
        fig.add_trace(
            go.Surface(
                z=z,
                x=x,
                y=y,
                visible=(t == 0),
                showscale=False,
                cmin=z_min,
                cmax=z_max,
            )
        )

    steps = []
    for t in range(time_steps):
        step = dict(
            method="update",
            args=[{"visible": [False] * time_steps}, {"title": f"{time_label}: {t}"}],
            label=f"{time_points[t]:.1f}",
        )
        step["args"][0]["visible"][t] = True  # pyright: ignore[reportArgumentType, reportIndexIssue]
        steps.append(step)

    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": f"{time_label}: "},
            pad={"t": 50},
            steps=steps,
        )
    ]

    fig.update_layout(
        title=title,
        sliders=sliders,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
            zaxis=dict(range=(z_min, z_max)),
        ),
        width=1000,
        scene_aspectmode="cube",
    )

    fig.show()
