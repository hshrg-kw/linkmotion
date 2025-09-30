"""URDF parser for creating Robot objects from URDF files."""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import trimesh

from linkmotion.robot.robot import Robot
from linkmotion.robot.link import Link
from linkmotion.robot.joint import Joint, JointType
from linkmotion.robot.shape.box import Box
from linkmotion.robot.shape.sphere import Sphere
from linkmotion.robot.shape.cylinder import Cylinder
from linkmotion.robot.shape.mesh import MeshShape
from linkmotion.transform.transform import Transform
from linkmotion.typing.numpy import RGBA0to1

logger = logging.getLogger(__name__)


class UrdfParser:
    """Parser for URDF (Unified Robot Description Format) files."""

    def __init__(self):
        """Initialize the URDF parser."""
        self._joint_type_map = {
            "revolute": JointType.REVOLUTE,
            "continuous": JointType.CONTINUOUS,
            "prismatic": JointType.PRISMATIC,
            "fixed": JointType.FIXED,
            "floating": JointType.FLOATING,
            "planar": JointType.PLANAR,
        }

    def parse_file(self, urdf_path: str | Path) -> Robot:
        """Parse a URDF file and create a Robot object.

        Args:
            urdf_path: Path to the URDF file.

        Returns:
            A Robot object created from the URDF.

        Raises:
            FileNotFoundError: If the URDF file doesn't exist.
            ValueError: If the URDF format is invalid.
        """
        urdf_path = Path(urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            return self._parse_robot_element(root, urdf_path.parent)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML in URDF file: {e}") from e

    def parse_string(self, urdf_string: str) -> Robot:
        """Parse a URDF string and create a Robot object.

        Args:
            urdf_string: URDF content as a string.

        Returns:
            A Robot object created from the URDF.

        Raises:
            ValueError: If the URDF format is invalid.
        """
        try:
            root = ET.fromstring(urdf_string)
            return self._parse_robot_element(root, None)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML in URDF string: {e}") from e

    def _parse_robot_element(
        self, robot_elem: ET.Element, urdf_dir: Optional[Path] = None
    ) -> Robot:
        """Parse the root robot element.

        Args:
            robot_elem: The root XML element of the robot.
            urdf_dir: Directory containing the URDF file (for relative mesh paths).

        Returns:
            A Robot object.
        """
        if robot_elem.tag != "robot":
            raise ValueError("Root element must be 'robot'")

        robot = Robot()

        # Parse materials first to build a color dictionary
        materials = self._parse_materials(robot_elem)

        # First pass: Parse links and joints with local transforms only
        joint_elements = {}
        for link_elem in robot_elem.findall("link"):
            link = self._parse_link_local(link_elem, materials, urdf_dir)
            robot.add_link(link)

        for joint_elem in robot_elem.findall("joint"):
            joint = self._parse_joint_local(joint_elem)
            joint_elements[joint.name] = joint_elem
            robot.add_joint(joint)

        # Second pass: Apply accumulated transforms from kinematic chain
        self._apply_kinematic_transforms(robot, joint_elements, materials)

        robot_name = robot_elem.get("name", "unnamed_robot")
        logger.info(
            f"Parsed URDF robot '{robot_name}' with {len(robot.links())} links and {len(robot._joint_dict)} joints"
        )

        return robot

    def _parse_materials(self, robot_elem: ET.Element) -> Dict[str, RGBA0to1]:
        """Parse material definitions and return a color dictionary.

        Args:
            robot_elem: The root robot element.

        Returns:
            Dictionary mapping material names to RGBA colors.
        """
        materials = {}
        for material_elem in robot_elem.findall("material"):
            name = material_elem.get("name")
            if name:
                color = self._parse_color(material_elem)
                if color is not None:
                    materials[name] = color
        return materials

    def _parse_link_local(
        self,
        link_elem: ET.Element,
        materials: Dict[str, RGBA0to1],
        urdf_dir: Optional[Path] = None,
    ) -> Link:
        """Parse a link element.

        Args:
            link_elem: The link XML element.
            materials: Dictionary of material name to color mappings.
            urdf_dir: Directory containing the URDF file (for relative mesh paths).

        Returns:
            A Link object.
        """
        name = link_elem.get("name")
        if not name:
            raise ValueError("Link must have a name attribute")

        # Parse visual element for geometry
        visual_elem = link_elem.find("visual")
        if visual_elem is not None:
            geometry_elem = visual_elem.find("geometry")
            if geometry_elem is not None:
                # Parse transform from origin
                origin_transform = self._parse_origin(visual_elem.find("origin"))

                # Parse material color
                color = self._parse_material_reference(visual_elem, materials)

                # Create shape based on geometry type
                shape = self._parse_geometry(
                    geometry_elem, origin_transform, color, urdf_dir
                )
                if shape is not None:
                    return Link(name, shape)

        # If no visual geometry found, create a default box shape
        logger.warning(f"Link '{name}' has no visual geometry, creating default box")
        default_shape = Box(np.array([0.1, 0.1, 0.1]))
        return Link(name, default_shape)

    def _parse_geometry(
        self,
        geometry_elem: ET.Element,
        transform: Transform,
        color: Optional[RGBA0to1],
        urdf_dir: Optional[Path] = None,
    ) -> Optional[Box | Sphere | Cylinder | MeshShape]:
        """Parse a geometry element and return the appropriate shape.

        Args:
            geometry_elem: The geometry XML element.
            transform: Transform to apply to the shape.
            color: Color for the shape.
            urdf_dir: Directory containing the URDF file (for relative mesh paths).

        Returns:
            A shape object or None if geometry type is not supported.
        """
        # Box geometry
        box_elem = geometry_elem.find("box")
        if box_elem is not None:
            size_str = box_elem.get("size", "1 1 1")
            size = np.array([float(x) for x in size_str.split()])
            return Box(size, transform, color)

        # Sphere geometry
        sphere_elem = geometry_elem.find("sphere")
        if sphere_elem is not None:
            radius = float(sphere_elem.get("radius", "1.0"))
            center = transform.position if transform else None
            return Sphere(radius, center, color)

        # Cylinder geometry
        cylinder_elem = geometry_elem.find("cylinder")
        if cylinder_elem is not None:
            radius = float(cylinder_elem.get("radius", "1.0"))
            length = float(cylinder_elem.get("length", "1.0"))
            return Cylinder(radius, length, transform, color)

        # Mesh geometry
        mesh_elem = geometry_elem.find("mesh")
        if mesh_elem is not None:
            return self._parse_mesh_geometry(mesh_elem, transform, color, urdf_dir)

        # Other geometry types not yet supported
        logger.warning(f"Unsupported geometry type in element: {geometry_elem}")
        return None

    def _parse_mesh_geometry(
        self,
        mesh_elem: ET.Element,
        transform: Transform,
        color: Optional[RGBA0to1],
        urdf_dir: Optional[Path] = None,
    ) -> Optional[MeshShape]:
        """Parse a mesh geometry element and return a MeshShape.

        Args:
            mesh_elem: The mesh XML element.
            transform: Transform to apply to the shape.
            color: Color for the shape.
            urdf_dir: Directory containing the URDF file (for relative mesh paths).

        Returns:
            A MeshShape object or None if the mesh cannot be loaded.
        """
        filename = mesh_elem.get("filename")
        if not filename:
            logger.error("Mesh element is missing 'filename' attribute")
            return None

        # Handle scale attribute (default to 1.0 for all axes)
        scale_str = mesh_elem.get("scale", "1.0 1.0 1.0")
        try:
            scale_parts = scale_str.split()
            if len(scale_parts) == 1:
                # Single value - apply to all axes
                scale = np.array([float(scale_parts[0])] * 3)
            elif len(scale_parts) == 3:
                # Three values - x, y, z scales
                scale = np.array([float(x) for x in scale_parts])
            else:
                logger.warning(
                    f"Invalid scale format '{scale_str}', using default scale [1, 1, 1]"
                )
                scale = np.array([1.0, 1.0, 1.0])
        except ValueError:
            logger.warning(
                f"Could not parse scale '{scale_str}', using default scale [1, 1, 1]"
            )
            scale = np.array([1.0, 1.0, 1.0])

        try:
            # Load the mesh file using trimesh
            # Handle common URDF mesh file prefixes and resolve paths
            mesh_path = filename

            if filename.startswith("package://"):
                # Handle ROS package:// URIs - for now, just remove the prefix
                # In a full ROS implementation, this would resolve to the actual package path
                mesh_path = filename.replace("package://", "")
                logger.warning(
                    f"Package URI '{filename}' found. Using relative path '{mesh_path}'. "
                    "Consider providing absolute paths for mesh files."
                )
            elif filename.startswith("file://"):
                # Handle file:// URIs
                mesh_path = filename.replace("file://", "")

            # Convert to Path object for easier manipulation
            mesh_path = Path(mesh_path)

            # If path is not absolute and we have a URDF directory, resolve relative to URDF file
            if not mesh_path.is_absolute() and urdf_dir is not None:
                mesh_path = urdf_dir / mesh_path
                logger.debug(
                    f"Resolved relative mesh path '{filename}' to '{mesh_path}'"
                )

            # Load the mesh
            mesh = trimesh.load(str(mesh_path), force="mesh")

            # Ensure we have a Trimesh object (not a Scene or other type)
            if isinstance(mesh, trimesh.Scene):
                # If it's a scene, try to get the first mesh
                if len(mesh.geometry) > 0:
                    mesh = list(mesh.geometry.values())[0]
                else:
                    logger.error(f"Scene from '{mesh_path}' contains no geometry")
                    return None

            if not isinstance(mesh, trimesh.Trimesh):
                logger.error(f"Loaded geometry from '{mesh_path}' is not a valid mesh")
                return None

            # Apply scaling if specified
            if not np.allclose(scale, [1.0, 1.0, 1.0]):
                mesh.apply_scale(scale)

            # Create and return the MeshShape (filename and scale parameters removed)
            return MeshShape(
                collision_mesh=mesh,
                visual_mesh=None,  # Use the same mesh for both collision and visual
                default_transform=transform,
                color=color,
            )

        except Exception as e:
            logger.error(
                f"Failed to load mesh from '{filename}' (resolved to '{mesh_path}'): {e}"
            )
            return None

    def _apply_kinematic_transforms(
        self,
        robot: Robot,
        joint_elements: Dict[str, ET.Element],
        materials: Dict[str, RGBA0to1],
    ) -> None:
        """Apply accumulated transforms from the kinematic chain to links and joints.

        This method computes the forward kinematics by walking through the kinematic tree
        and accumulating transforms from each joint's origin. Every link and joint will
        have its position and orientation updated based on all ancestor joint transforms.

        Args:
            robot: The Robot object with links and joints already added.
            joint_elements: Dictionary mapping joint names to their XML elements.
            materials: Dictionary of material definitions.
        """
        # Find root links (links with no parent joints)
        root_links = self._find_root_links(robot)

        # Process each kinematic tree starting from root links
        for root_link_name in root_links:
            self._process_kinematic_subtree(
                robot, root_link_name, Transform(), joint_elements, materials
            )

    def _find_root_links(self, robot: Robot) -> List[str]:
        """Find all root links (links that have no parent joints).

        Args:
            robot: The Robot object.

        Returns:
            List of root link names.
        """
        root_links = []
        for link_name in robot._link_dict.keys():
            if link_name not in robot._child_link_to_parent_joint:
                root_links.append(link_name)
        return root_links

    def _process_kinematic_subtree(
        self,
        robot: Robot,
        link_name: str,
        accumulated_transform: Transform,
        joint_elements: Dict[str, ET.Element],
        materials: Dict[str, RGBA0to1],
    ) -> None:
        """Recursively process a kinematic subtree, applying accumulated transforms.

        Args:
            robot: The Robot object.
            link_name: Name of the current link being processed.
            accumulated_transform: Transform accumulated from all ancestor joints.
            joint_elements: Dictionary mapping joint names to their XML elements.
            materials: Dictionary of material definitions.
        """
        # Apply accumulated transform to the current link
        link = robot._link_dict[link_name]
        if (
            hasattr(link.shape, "default_transform")
            and link.shape.default_transform is not None
        ):
            # Combine the accumulated transform with the link's local transform
            combined_transform = accumulated_transform.apply(
                link.shape.default_transform
            )
            link.shape.default_transform = combined_transform
        else:
            # Set the accumulated transform as the link's transform
            link.shape.default_transform = accumulated_transform

        # Find all child joints of this link
        if link_name in robot._parent_link_to_child_joints:
            for child_joint_name in robot._parent_link_to_child_joints[link_name]:
                joint = robot._joint_dict[child_joint_name]
                joint_elem = joint_elements[child_joint_name]

                # Parse the joint's origin transform
                origin_elem = joint_elem.find("origin")
                joint_origin_transform = self._parse_origin(origin_elem)

                # Accumulate this joint's transform
                new_accumulated_transform = accumulated_transform.apply(
                    joint_origin_transform
                )

                # Update joint center and axis with accumulated transform
                if joint.center is not None:
                    # Joint center should be transformed by the accumulated transform up to this joint
                    joint.center = accumulated_transform.apply(joint.center)

                if joint.direction is not None:
                    # For direction vectors, we only apply rotation (not translation)
                    joint.direction = accumulated_transform.rotation.apply(
                        joint.direction
                    )

                # Recursively process the child link
                child_link_name = joint.child_link_name
                self._process_kinematic_subtree(
                    robot,
                    child_link_name,
                    new_accumulated_transform,
                    joint_elements,
                    materials,
                )

    def _parse_joint_local(self, joint_elem: ET.Element) -> Joint:
        """Parse a joint element.

        Args:
            joint_elem: The joint XML element.

        Returns:
            A Joint object.
        """
        name = joint_elem.get("name")
        if not name:
            raise ValueError("Joint must have a name attribute")

        joint_type_str = joint_elem.get("type", "fixed")
        if joint_type_str not in self._joint_type_map:
            raise ValueError(f"Unknown joint type: {joint_type_str}")
        joint_type = self._joint_type_map[joint_type_str]

        # Parse parent and child links
        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")

        if parent_elem is None or child_elem is None:
            raise ValueError(f"Joint '{name}' must have both parent and child links")

        parent_link_name = parent_elem.get("link")
        child_link_name = child_elem.get("link")

        if not parent_link_name or not child_link_name:
            raise ValueError(f"Joint '{name}' parent and child must specify link names")

        # Parse origin for joint center
        origin_elem = joint_elem.find("origin")
        center = None
        if origin_elem is not None:
            xyz_str = origin_elem.get("xyz", "0 0 0")
            center = np.array([float(x) for x in xyz_str.split()])

        # Parse axis for joint direction
        axis_elem = joint_elem.find("axis")
        direction = None
        if axis_elem is not None:
            xyz_str = axis_elem.get("xyz", "0 0 1")
            direction = np.array([float(x) for x in xyz_str.split()])

        # Parse limits
        min_limit = -float("inf")
        max_limit = float("inf")
        limit_elem = joint_elem.find("limit")
        if limit_elem is not None:
            lower_str = limit_elem.get("lower")
            if lower_str is not None:
                min_limit = float(lower_str)
            upper_str = limit_elem.get("upper")
            if upper_str is not None:
                max_limit = float(upper_str)

        return Joint(
            name=name,
            type=joint_type,
            child_link_name=child_link_name,
            parent_link_name=parent_link_name,
            direction=direction,
            center=center,
            min_=min_limit,
            max_=max_limit,
        )

    def _parse_origin(self, origin_elem: Optional[ET.Element]) -> Transform:
        """Parse an origin element and return a Transform.

        Args:
            origin_elem: The origin XML element, or None.

        Returns:
            A Transform object.
        """
        if origin_elem is None:
            return Transform()

        # Parse translation
        xyz_str = origin_elem.get("xyz", "0 0 0")
        translation = np.array([float(x) for x in xyz_str.split()])

        # Parse rotation (roll, pitch, yaw)
        rpy_str = origin_elem.get("rpy", "0 0 0")
        rpy = np.array([float(x) for x in rpy_str.split()])

        # Create rotation from Euler angles (roll, pitch, yaw)
        from scipy.spatial.transform import Rotation as R

        rotation = R.from_euler("xyz", rpy, degrees=False)

        return Transform(rotation, translation)

    def _parse_color(self, material_elem: ET.Element) -> Optional[RGBA0to1]:
        """Parse color from a material element.

        Args:
            material_elem: The material XML element.

        Returns:
            RGBA color array or None if no color found.
        """
        color_elem = material_elem.find("color")
        if color_elem is not None:
            rgba_str = color_elem.get("rgba", "1 1 1 1")
            return np.array([float(x) for x in rgba_str.split()])
        return None

    def _parse_material_reference(
        self, visual_elem: ET.Element, materials: Dict[str, RGBA0to1]
    ) -> Optional[RGBA0to1]:
        """Parse material reference and return color.

        Args:
            visual_elem: The visual XML element.
            materials: Dictionary of material definitions.

        Returns:
            RGBA color array or None.
        """
        material_elem = visual_elem.find("material")
        if material_elem is not None:
            # Check for inline color definition
            color = self._parse_color(material_elem)
            if color is not None:
                return color

            # Check for material reference
            name = material_elem.get("name")
            if name and name in materials:
                return materials[name]

        return None
