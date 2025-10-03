import logging

import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R

from linkmotion.robot.robot import Robot
from linkmotion.robot.joint import JointType, Joint
from linkmotion.transform import Transform
from linkmotion.typing.numpy import Vector3
from linkmotion.transform.manager import TransformManager

logger = logging.getLogger(__name__)


class JointLimitError(Exception):
    def __init__(self, value: float, joint: Joint):
        self.value = value
        self.joint = joint
        message = f"'{value}' is out of limits for joint '{joint.name}': [{joint.min}, {joint.max}]"
        super().__init__(message)


class MoveManager:
    """Manages the movement and state of a robot's links and joints.

    This class provides an interface to move individual robot joints,
    query the spatial transformations of links, and retrieve geometric
    information like visual or collision meshes in their current state.

    Attributes:
        robot (Robot): The robot model instance.
        transform_manager (TransformManager): Manages the hierarchy and
            computation of transformations between links.
        link_name_to_id (dict[str, int]): A mapping from link names to
            their unique integer IDs.
    """

    def __init__(self, robot: Robot):
        """Initializes the MoveManager.

        Args:
            robot (Robot): An instance of the Robot class containing its
                kinematic structure (links and joints).
        """
        self.robot = robot
        self.transform_manager = TransformManager()
        self.joint_values_map = dict[str, float]()

        # Create a mapping from link names to integer IDs for quick lookups.
        self.link_name_to_id = {
            link.name: idx for idx, link in enumerate(self.robot.traverse_links())
        }

        # Build the transform tree based on the robot's link hierarchy.
        for link_name, link_id in self.link_name_to_id.items():
            parent_joint = self.robot.parent_joint(link_name)
            parent_id = None
            if parent_joint:
                try:
                    parent_link_name = parent_joint.parent_link_name
                    parent_id = self.link_name_to_id[parent_link_name]
                except KeyError:
                    # This indicates an inconsistent robot model, which is a critical error.
                    logger.error(
                        f"Parent link '{parent_joint.parent_link_name}' not found for link '{link_name}'."
                    )
                    raise ValueError(
                        f"Inconsistent robot model: Parent link '{parent_joint.parent_link_name}' not found."
                    )

            # Each node is initialized with a default identity transform.
            self.transform_manager.add_node(link_id, parent_id, Transform())

        logger.info(
            f"MoveManager initialized for robot with {len(self.link_name_to_id)} links."
        )

    def move(self, joint_name: str, value: float | Transform):
        """Sets the position or orientation of a specified joint.

        The expected type of 'value' depends on the joint type.
        - PRISMATIC, REVOLUTE, CONTINUOUS: Requires a float.
        - PLANAR, FLOATING: Requires a Transform object.
        - FIXED: Cannot be moved and will raise an error.

        Args:
            joint_name (str): The name of the joint to move.
            value (float | Transform): The target value for the joint.

        Raises:
            ValueError: If the joint name is not found, if the joint is
                of a fixed type, if the value type is incorrect for the
                joint type, or if a required attribute (like 'center' for
                revolute joints) is missing.
        """
        try:
            joint = self.robot.joint(joint_name)
            child_link = self.robot.link(joint.child_link_name)
            child_link_id = self.link_name_to_id[child_link.name]
        except KeyError:
            logger.error(
                f"Move failed: Joint '{joint_name}' or its child link not found."
            )
            raise ValueError(
                f"Joint '{joint_name}' or its associated links not found in the robot model."
            )

        logger.debug(
            f"Moving joint '{joint_name}' of type '{joint.type}' to value: {value}"
        )

        local_transform = None
        match joint.type:
            case JointType.PRISMATIC:
                if isinstance(value, (int, float)):
                    if value > joint.max or value < joint.min:
                        raise JointLimitError(value, joint)
                    translate = value * np.array(joint.direction)
                    local_transform = Transform(translate=translate)
                else:
                    raise ValueError(
                        f"Value for prismatic joint must be a float, got {type(value)}"
                    )

            case JointType.REVOLUTE | JointType.CONTINUOUS:
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Value for revolute joint must be a float, got {type(value)}"
                    )
                if joint.center is None:
                    raise ValueError(
                        f"Joint '{joint_name}' does not have a defined center for rotation."
                    )
                if joint.type == JointType.REVOLUTE and (
                    value > joint.max or value < joint.min
                ):
                    raise JointLimitError(value, joint)

                # To rotate around a specific point (joint.center), we apply a sequence of transforms:
                # 1. Translate to origin: Move the object so the rotation center is at (0,0,0).
                # 2. Rotate: Perform the rotation around the axis.
                # 3. Translate back: Move the object back to its original position.
                # This is equivalent to T * R * T^-1.
                rotation = R.from_rotvec(value * np.array(joint.direction))
                center_point = np.array(joint.center)

                # The Transform class is assumed to handle this sequence correctly.
                # Transform(rot, center) likely creates a transform T(center) * Rot(rot) * T(-center).
                local_transform = Transform(
                    rotate=rotation, translate=center_point
                ).apply(Transform(translate=-center_point))

            case JointType.PLANAR | JointType.FLOATING:
                if not isinstance(value, Transform):
                    raise ValueError(
                        f"Value for planar/floating joint must be a Transform, got {type(value)}"
                    )
                local_transform = value

            case JointType.FIXED:
                logger.warning(f"Attempted to move a fixed joint: '{joint_name}'")
                raise ValueError(f"Cannot move fixed joint: {joint_name}")

            case _:
                raise ValueError(f"Unsupported joint type: {joint.type}")

        if local_transform is not None:
            self.transform_manager.set_local_transform(child_link_id, local_transform)
            if isinstance(value, (int, float)):
                self.joint_values_map[joint_name] = value

    def get_transform(self, link_name: str) -> Transform:
        """Retrieves the world-space transform of a given link.

        Args:
            link_name (str): The name of the link.

        Returns:
            Transform: The transform object representing the link's
                position and orientation in the world frame.

        Raises:
            ValueError: If the link name is not found.
        """
        try:
            link_id = self.link_name_to_id[link_name]
            return self.transform_manager.get_world_transform(link_id)
        except KeyError:
            logger.error(f"Link '{link_name}' not found.")
            raise ValueError(f"Link '{link_name}' not found in the robot model.")

    def get_center(self, joint_name: str) -> Vector3 | None:
        """Calculates the world-space position of a joint's center.

        Args:
            joint_name (str): The name of the joint.

        Returns:
            Vector3 | None: The 3D coordinate of the joint's center in the
                world frame, or None if the joint has no defined center.

        Raises:
            ValueError: If the joint name is not found.
        """
        try:
            joint = self.robot.joint(joint_name)
        except KeyError:
            raise ValueError(f"Joint '{joint_name}' not found in the robot model.")

        if joint.center is None:
            return None

        transform = self.get_transform(joint.parent_link_name)
        return transform.apply(joint.center)

    def get_direction(self, joint_name: str) -> Vector3:
        """Calculates the world-space direction vector of a joint's axis.

        Args:
            joint_name (str): The name of the joint.

        Returns:
            Vector3: The 3D direction vector of the joint's axis in the
                world frame.

        Raises:
            ValueError: If the joint name is not found.
        """
        try:
            joint = self.robot.joint(joint_name)
        except KeyError:
            raise ValueError(f"Joint '{joint_name}' not found in the robot model.")

        transform = self.get_transform(joint.parent_link_name)
        rotate_transform = Transform(rotate=transform.rotation)
        return rotate_transform.apply(joint.direction)

    def get_link_visual_mesh(self, link_name: str) -> trimesh.Trimesh:
        """Gets the visual mesh of a link, transformed to its world position.

        Args:
            link_name (str): The name of the link.

        Returns:
            trimesh.Trimesh: The transformed visual mesh.

        Raises:
            ValueError: If the link name is not found.
        """
        try:
            link = self.robot.link(link_name)
            transform = self.get_transform(link_name)
            return link.visual_mesh(transform)
        except KeyError:
            raise ValueError(f"Link '{link_name}' not found in the robot model.")

    def get_link_collision_obj(self, link_name: str) -> trimesh.Trimesh:
        """Gets the collision object of a link, transformed to its world position.

        Args:
            link_name (str): The name of the link.

        Returns:
            trimesh.Trimesh: The transformed collision object.

        Raises:
            ValueError: If the link name is not found.
        """
        try:
            link = self.robot.link(link_name)
            transform = self.get_transform(link_name)
            return link.collision_object(transform)
        except KeyError:
            raise ValueError(f"Link '{link_name}' not found in the robot model.")

    def reset_move(self):
        """Resets all joint movements to their initial state (identity transform)."""
        logger.info("Resetting all joint transforms.")
        self.transform_manager.reset_all_transforms()
        self.joint_values_map.clear()

    def copy(self) -> "MoveManager":
        """Creates a copy of the MoveManager instance.

        TransformManager is a deep copy to ensure independent state.
        robot is shallow copied as it is assumed to be immutable.

        Returns:
            MoveManager: A new instance of MoveManager with the same
                robot model and transform state.
        """
        new_manager = MoveManager(self.robot)
        new_manager.transform_manager = self.transform_manager.copy()
        new_manager.joint_values_map = self.joint_values_map.copy()
        return new_manager

    def joint_value(self, joint_name: str) -> float:
        """Gets the current value of a joint.

        Args:
            joint_name (str): The name of the joint.

        Returns:
            float: The current value of the joint
        """
        joint_type = self.robot.joint(joint_name).type
        if joint_type in [JointType.FIXED, JointType.PLANAR, JointType.FLOATING]:
            raise ValueError(
                f"Joint '{joint_name}' of type '{joint_type}' has no single value."
            )
        elif joint_type in [
            JointType.REVOLUTE,
            JointType.PRISMATIC,
            JointType.CONTINUOUS,
        ]:
            ret = self.joint_values_map.get(joint_name, 0)
            return ret
        else:
            raise ValueError(
                f"Unsupported joint type: {joint_type} of joint '{joint_name}'."
            )
