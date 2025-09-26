import logging

import numpy as np
import trimesh

from linkmotion.robot import Robot
from linkmotion.robot.shape.box import Box
from linkmotion.robot.shape.cylinder import Cylinder
from linkmotion.robot.shape.sphere import Sphere
from linkmotion.transform import Transform
from linkmotion.modeling.remove import remove_outside_of_box
from linkmotion.modeling.sweep import sweep_trimesh, rotate_overlap_trimesh
from linkmotion.robot.shape.mesh import MeshShape
from linkmotion.robot.shape.base import ShapeBase
from linkmotion.typing.numpy import Vector3, RGBA0to1

logger = logging.getLogger(__name__)


class CollisionMeshCustomizer:
    """Utility class for customizing robot collision meshes.

    Provides static methods for various mesh operations including clipping,
    sweeping, rotation, and primitive conversion.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @staticmethod
    def remove_outside_of_box(
        robot: Robot, link_names: set[str], min_: Vector3, max_: Vector3
    ) -> None:
        """Remove parts of robot meshes that are outside the specified bounding box.

        Args:
            robot: Robot instance to modify.
            link_names: Set of link names to process.
            min_: Minimum corner of the bounding box.
            max_: Maximum corner of the bounding box.
        """
        for link_name in link_names:
            link = robot.link(link_name)
            shape = link.shape
            if not isinstance(shape, MeshShape):
                logger.info(
                    f"the remove_outside_of_box operation of Link {link_name} shape is skipped because it is not MeshShape"
                )
                continue

            old_collision_mesh = shape.collision_mesh
            old_visual_mesh = shape.visual_mesh

            new_collision_mesh = remove_outside_of_box(old_collision_mesh, min_, max_)
            new_visual_mesh = remove_outside_of_box(old_visual_mesh, min_, max_)

            link.shape = MeshShape(
                collision_mesh=new_collision_mesh,
                visual_mesh=new_visual_mesh,
                default_transform=shape.default_transform,
                color=shape.color,
            )

    @staticmethod
    def sweep_mesh(
        robot: Robot,
        link_names: set[str],
        initial_translate: Vector3,
        sweep_translate: Vector3,
    ) -> None:
        """Sweep robot meshes by first translating then sweeping.

        Args:
            robot: Robot instance to modify.
            link_names: Set of link names to process.
            initial_translate: Initial translation before sweeping.
            sweep_translate: Translation vector for the sweep operation.
        """
        for link_name in link_names:
            link = robot.link(link_name)
            shape = link.shape
            if not isinstance(shape, MeshShape):
                logger.info(
                    f"the sweep_mesh operation of Link {link_name} shape is skipped because it is not MeshShape"
                )
                continue

            translate = Transform(translate=initial_translate)
            old_collision_mesh = shape.collision_mesh
            old_visual_mesh = shape.visual_mesh

            new_collision_mesh = translate.apply(old_collision_mesh)
            new_visual_mesh = translate.apply(old_visual_mesh)

            new_collision_mesh = sweep_trimesh(new_collision_mesh, sweep_translate)
            new_visual_mesh = sweep_trimesh(new_visual_mesh, sweep_translate)

            link.shape = MeshShape(
                collision_mesh=new_collision_mesh,
                visual_mesh=new_visual_mesh,
                default_transform=shape.default_transform,
                color=shape.color,
            )

    @staticmethod
    def rotate_overlap(
        robot: Robot,
        link_names: set[str],
        center: Vector3,
        normalized_direction: Vector3,
        delta_angle: float,
        initial_angle: float,
        how_many_to_add: int,
    ) -> None:
        """Create overlapping rotated copies of robot meshes.

        Args:
            robot: Robot instance to modify.
            link_names: Set of link names to process.
            center: Center point of rotation.
            normalized_direction: Normalized rotation axis vector.
            delta_angle: Angle increment between copies (radians).
            initial_angle: Starting rotation angle (radians).
            how_many_to_add: Number of additional rotated copies to create.
        """
        for link_name in link_names:
            link = robot.link(link_name)

            shape = link.shape
            if not isinstance(shape, MeshShape):
                logger.info(
                    f"the rotate_overlap operation of Link {link_name} shape is skipped because it is not MeshShape"
                )
                continue

            old_collision_mesh = shape.collision_mesh
            old_visual_mesh = shape.visual_mesh

            logger.info("rotating collision mesh")
            new_collision_mesh = rotate_overlap_trimesh(
                old_collision_mesh,
                center,
                normalized_direction,
                delta_angle,
                initial_angle,
                how_many_to_add,
            )
            logger.info("completed rotating collision mesh")

            logger.info("rotating visual mesh")
            new_visual_mesh = rotate_overlap_trimesh(
                old_visual_mesh,
                center,
                normalized_direction,
                delta_angle,
                initial_angle,
                how_many_to_add,
            )
            logger.info("completed rotating visual mesh")

            link.shape = MeshShape(
                collision_mesh=new_collision_mesh,
                visual_mesh=new_visual_mesh,
                default_transform=shape.default_transform,
                color=shape.color,
            )

    @staticmethod
    def from_mesh_to_bounding_primitive(robot: Robot, link_names: set[str]) -> None:
        """Convert robot mesh shapes to their bounding primitive shapes.

        Args:
            robot: Robot instance to modify.
            link_names: Set of link names to process.
        """
        for link_name in link_names:
            link = robot.link(link_name)
            shape = link.shape
            if not isinstance(shape, MeshShape):
                logger.info(
                    f"the from_mesh_to_bounding_primitive operation of Link {link_name} shape is skipped because it is not MeshShape"
                )
                continue
            link.shape = from_mesh_to_bounding_primitive(
                shape.collision_mesh, shape.color
            )


def from_mesh_to_bounding_primitive(
    mesh: trimesh.Trimesh,
    color: RGBA0to1 | None = None,
) -> ShapeBase:
    """Convert a trimesh to its bounding primitive shape.

    Args:
        mesh: Input triangular mesh.
        color: Optional color for the resulting shape.

    Returns:
        Bounding primitive shape (Box, Sphere, or Cylinder).

    Raises:
        ValueError: If the primitive kind is not supported.
    """
    primitive_mesh = mesh.bounding_primitive
    d = primitive_mesh.to_dict()
    transform = Transform.from_4x4(np.array(d["transform"]))
    if d["kind"] == "box":
        return Box(np.array(d["extents"]), transform, color)
    elif d["kind"] == "sphere":
        return Sphere(d["radius"], transform.position, color)
    elif d["kind"] == "cylinder":
        return Cylinder(d["radius"], d["height"], transform, color)
    else:
        raise ValueError(f"Unsupported primitive kind: {d['kind']}")
