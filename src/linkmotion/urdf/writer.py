"""URDF writer for exporting Robot objects to URDF files."""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Dict
import hashlib

import numpy as np
import trimesh

from linkmotion.robot.robot import Robot
from linkmotion.robot.link import Link
from linkmotion.robot.joint import Joint, JointType
from linkmotion.robot.shape.box import Box
from linkmotion.robot.shape.sphere import Sphere
from linkmotion.robot.shape.cylinder import Cylinder
from linkmotion.robot.shape.mesh import MeshShape
from linkmotion.robot.shape.capsule import Capsule
from linkmotion.robot.shape.cone import Cone
from linkmotion.transform.transform import Transform
from linkmotion.typing.numpy import RGBA0to1

logger = logging.getLogger(__name__)


class UrdfWriter:
    """Writer for URDF (Unified Robot Description Format) files."""

    def __init__(self):
        """Initialize the URDF writer."""
        self._joint_type_map = {
            JointType.REVOLUTE: "revolute",
            JointType.CONTINUOUS: "continuous",
            JointType.PRISMATIC: "prismatic",
            JointType.FIXED: "fixed",
            JointType.FLOATING: "floating",
            JointType.PLANAR: "planar",
        }

    def write_file(
        self,
        robot: Robot,
        urdf_path: str | Path,
        robot_name: str = "robot",
        export_meshes: bool = True,
        meshes_dir: str = "meshes",
    ) -> None:
        """Write a Robot object to a URDF file with optional mesh export.

        Args:
            robot: The Robot object to write.
            urdf_path: Path where the URDF file will be saved.
            robot_name: Name to assign to the robot in the URDF.
            export_meshes: Whether to export mesh files alongside the URDF.
            meshes_dir: Directory name for mesh files (relative to URDF).
        """
        urdf_path = Path(urdf_path)
        urdf_path.parent.mkdir(parents=True, exist_ok=True)

        # Export meshes if requested
        mesh_filename_map = {}
        if export_meshes:
            mesh_filename_map = self._export_mesh_files(robot, urdf_path, meshes_dir)

        urdf_string = self.to_string(robot, robot_name, mesh_filename_map)

        with open(urdf_path, "w", encoding="utf-8") as f:
            f.write(urdf_string)

        logger.info(f"Wrote URDF file: {urdf_path}")
        if export_meshes and mesh_filename_map:
            logger.info(
                f"Exported {len(mesh_filename_map)} mesh files to {meshes_dir}/"
            )

    def to_string(
        self,
        robot: Robot,
        robot_name: str = "robot",
        mesh_filename_map: Optional[Dict] = None,
    ) -> str:
        """Convert a Robot object to URDF string.

        Args:
            robot: The Robot object to convert.
            robot_name: Name to assign to the robot in the URDF.
            mesh_filename_map: Optional mapping of mesh shapes to exported filenames.

        Returns:
            URDF content as a string.
        """
        root = self._create_robot_element(robot, robot_name, mesh_filename_map)

        # Format the XML with proper indentation
        self._indent_xml(root)

        # Create XML declaration and convert to string
        xml_str = ET.tostring(root, encoding="unicode")
        return f'<?xml version="1.0"?>\n{xml_str}\n'

    def _create_robot_element(
        self, robot: Robot, robot_name: str, mesh_filename_map: Optional[Dict] = None
    ) -> ET.Element:
        """Create the root robot XML element.

        Args:
            robot: The Robot object.
            robot_name: Name for the robot.
            mesh_filename_map: Optional mapping of mesh shapes to exported filenames.

        Returns:
            The root robot XML element.
        """
        robot_elem = ET.Element("robot")
        robot_elem.set("name", robot_name)

        # Collect unique materials from all links
        materials = self._collect_materials(robot)

        # Add material definitions
        for material_name, color in materials.items():
            material_elem = self._create_material_element(material_name, color)
            robot_elem.append(material_elem)

        # Compute relative transforms for proper URDF output
        link_relative_transforms, joint_relative_transforms, joint_relative_axes = (
            self._compute_relative_transforms(robot)
        )

        # Add links with relative transforms
        for link in robot.links():
            relative_transform = link_relative_transforms.get(link.name, Transform())
            link_elem = self._create_link_element(
                link, materials, relative_transform, mesh_filename_map
            )
            robot_elem.append(link_elem)

        # Add joints with relative transforms
        for joint in robot._joint_dict.values():
            relative_transform = joint_relative_transforms.get(joint.name, Transform())
            relative_axis = joint_relative_axes.get(joint.name, joint.direction)
            joint_elem = self._create_joint_element(
                joint, relative_transform, relative_axis
            )
            robot_elem.append(joint_elem)

        return robot_elem

    def _compute_relative_transforms(
        self, robot: Robot
    ) -> tuple[Dict[str, Transform], Dict[str, Transform], Dict[str, np.ndarray]]:
        """Compute relative transforms for links and joints from accumulated transforms.

        This method reverses the transform accumulation process to compute the relative
        transforms that should be written to the URDF file.

        Args:
            robot: The Robot object with accumulated transforms.

        Returns:
            Tuple of (link_relative_transforms, joint_relative_transforms, joint_relative_axes)
        """
        from typing import Dict
        import numpy as np

        link_relative_transforms: Dict[str, Transform] = {}
        joint_relative_transforms: Dict[str, Transform] = {}
        joint_relative_axes: Dict[str, np.ndarray] = {}

        # Find root links and process each kinematic tree
        root_links = self._find_root_links(robot)

        for root_link_name in root_links:
            self._compute_relative_transforms_recursive(
                robot,
                root_link_name,
                Transform(),
                link_relative_transforms,
                joint_relative_transforms,
                joint_relative_axes,
            )

        return link_relative_transforms, joint_relative_transforms, joint_relative_axes

    def _find_root_links(self, robot: Robot) -> list[str]:
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

    def _compute_relative_transforms_recursive(
        self,
        robot: Robot,
        link_name: str,
        link_world_transform: Transform,
        link_relative_transforms: Dict[str, Transform],
        joint_relative_transforms: Dict[str, Transform],
        joint_relative_axes: Dict[str, np.ndarray],
    ) -> None:
        """Recursively compute relative transforms for a kinematic subtree.

        Args:
            robot: The Robot object.
            link_name: Current link being processed.
            link_world_transform: World transform of the current link's coordinate frame.
            link_relative_transforms: Output dictionary for link relative transforms.
            joint_relative_transforms: Output dictionary for joint relative transforms.
            joint_relative_axes: Output dictionary for joint relative axes.
        """
        link = robot._link_dict[link_name]

        # Compute the link's visual origin relative to its own coordinate frame
        if (
            hasattr(link.shape, "default_transform")
            and link.shape.default_transform is not None
        ):
            # The shape's default_transform is in world coordinates
            # We need to express it relative to the link's coordinate frame
            shape_world_transform = link.shape.default_transform

            # Compute: link_frame^-1 * shape_world_transform
            link_inv = Transform(
                link_world_transform.rotation.inv(),
                -link_world_transform.rotation.inv().apply(
                    link_world_transform.position
                ),
            )
            relative_visual_transform = link_inv.apply(shape_world_transform)
            link_relative_transforms[link_name] = relative_visual_transform
        else:
            link_relative_transforms[link_name] = Transform()

        # Process child joints
        if link_name in robot._parent_link_to_child_joints:
            for child_joint_name in robot._parent_link_to_child_joints[link_name]:
                joint = robot._joint_dict[child_joint_name]

                # Compute joint origin relative to parent link frame
                if joint.center is not None:
                    # Joint center is in world coordinates, transform to parent link frame
                    link_inv = Transform(
                        link_world_transform.rotation.inv(),
                        -link_world_transform.rotation.inv().apply(
                            link_world_transform.position
                        ),
                    )
                    relative_joint_center = link_inv.apply(joint.center)
                    joint_relative_transforms[child_joint_name] = Transform(
                        translate=relative_joint_center
                    )
                else:
                    joint_relative_transforms[child_joint_name] = Transform()

                # Compute joint axis relative to parent link frame
                if joint.direction is not None:
                    # Rotate axis from world frame to parent link frame
                    relative_axis = link_world_transform.rotation.inv().apply(
                        joint.direction
                    )
                    joint_relative_axes[child_joint_name] = relative_axis
                else:
                    joint_relative_axes[child_joint_name] = joint.direction

                # Compute world transform of child link
                # Child link frame = parent link frame * joint transform
                if joint.center is not None:
                    joint_transform = Transform(translate=relative_joint_center)
                    child_world_transform = link_world_transform.apply(joint_transform)
                else:
                    child_world_transform = link_world_transform

                # Recursively process child link
                child_link_name = joint.child_link_name
                self._compute_relative_transforms_recursive(
                    robot,
                    child_link_name,
                    child_world_transform,
                    link_relative_transforms,
                    joint_relative_transforms,
                    joint_relative_axes,
                )

    def _collect_materials(self, robot: Robot) -> Dict[str, RGBA0to1]:
        """Collect unique materials from all links.

        Args:
            robot: The Robot object.

        Returns:
            Dictionary mapping material names to colors.
        """
        materials = {}
        for link in robot.links():
            color = self._get_link_color(link)
            if color is not None:
                # Create a material name based on the color
                material_name = self._color_to_material_name(color)
                materials[material_name] = color
        return materials

    def _get_link_color(self, link: Link) -> Optional[RGBA0to1]:
        """Get the color of a link's shape.

        Args:
            link: The Link object.

        Returns:
            RGBA color array or None.
        """
        shape = link.shape
        if hasattr(shape, "color") and shape.color is not None:
            return shape.color
        return None

    def _color_to_material_name(self, color: RGBA0to1) -> str:
        """Convert an RGBA color to a material name.

        Args:
            color: RGBA color array.

        Returns:
            Material name string.
        """
        # Create a simple material name based on color values
        r, g, b, a = color
        return f"material_{r:.2f}_{g:.2f}_{b:.2f}_{a:.2f}".replace(".", "_")

    def _create_material_element(self, name: str, color: RGBA0to1) -> ET.Element:
        """Create a material XML element.

        Args:
            name: Material name.
            color: RGBA color.

        Returns:
            Material XML element.
        """
        material_elem = ET.Element("material")
        material_elem.set("name", name)

        color_elem = ET.SubElement(material_elem, "color")
        color_str = " ".join(f"{x:.6f}" for x in color)
        color_elem.set("rgba", color_str)

        return material_elem

    def _create_link_element(
        self,
        link: Link,
        materials: Dict[str, RGBA0to1],
        relative_transform: Optional[Transform] = None,
        mesh_filename_map: Optional[Dict] = None,
    ) -> ET.Element:
        """Create a link XML element.

        Args:
            link: The Link object.
            materials: Dictionary of materials.

        Returns:
            Link XML element.
        """
        link_elem = ET.Element("link")
        link_elem.set("name", link.name)

        # Add geometry - this is required for valid URDF
        geometry_elem = self._create_geometry_element(link.shape, mesh_filename_map)
        if geometry_elem is None:
            logger.warning(
                f"Link '{link.name}' has no supported geometry for URDF export"
            )
            return link_elem

        # Create visual element
        visual_elem = ET.SubElement(link_elem, "visual")
        visual_elem.append(geometry_elem)

        # Add origin using the computed relative transform
        if relative_transform is not None and not relative_transform.is_identity():
            origin_elem = self._create_origin_element_from_transform(relative_transform)
            if origin_elem is not None:
                visual_elem.append(origin_elem)

        # Add material reference
        color = self._get_link_color(link)
        if color is not None:
            material_name = self._color_to_material_name(color)
            material_elem = ET.SubElement(visual_elem, "material")
            material_elem.set("name", material_name)

        # Create collision element (same geometry as visual)
        collision_elem = ET.SubElement(link_elem, "collision")

        # Clone the geometry element for collision
        collision_geometry_elem = self._create_geometry_element(
            link.shape, mesh_filename_map
        )
        if collision_geometry_elem is not None:
            collision_elem.append(collision_geometry_elem)

            # Add the same origin to collision as visual
            if relative_transform is not None and not relative_transform.is_identity():
                collision_origin_elem = self._create_origin_element_from_transform(
                    relative_transform
                )
                if collision_origin_elem is not None:
                    collision_elem.append(collision_origin_elem)

        return link_elem

    def _create_geometry_element(
        self, shape, mesh_filename_map: Optional[Dict] = None
    ) -> Optional[ET.Element]:
        """Create a geometry XML element from a shape.

        Args:
            shape: The shape object.
            mesh_filename_map: Optional mapping of mesh shapes to exported filenames.

        Returns:
            Geometry XML element or None if shape type is not supported.
        """
        geometry_elem = ET.Element("geometry")

        if isinstance(shape, Box):
            box_elem = ET.SubElement(geometry_elem, "box")
            size_str = " ".join(f"{x:.6f}" for x in shape.extents)
            box_elem.set("size", size_str)
            return geometry_elem

        elif isinstance(shape, Sphere):
            sphere_elem = ET.SubElement(geometry_elem, "sphere")
            sphere_elem.set("radius", f"{shape.radius:.6f}")
            return geometry_elem

        elif isinstance(shape, Cylinder):
            cylinder_elem = ET.SubElement(geometry_elem, "cylinder")
            cylinder_elem.set("radius", f"{shape.radius:.6f}")
            cylinder_elem.set("length", f"{shape.height:.6f}")
            return geometry_elem

        elif isinstance(shape, MeshShape):
            mesh_elem = ET.SubElement(geometry_elem, "mesh")

            # Use exported filename if available, otherwise use fallback
            filename = None
            if mesh_filename_map and shape in mesh_filename_map:
                filename = mesh_filename_map[shape]
            else:
                filename = "mesh.stl"  # Fallback placeholder
                logger.warning(
                    "Mesh shape exported with placeholder filename. Original filename not available."
                )

            mesh_elem.set("filename", filename)

            # Note: Scale attribute removed from MeshShape - meshes should be pre-scaled

            return geometry_elem

        elif isinstance(shape, (Capsule, Cone)):
            # Convert unsupported shapes to mesh geometry
            converted_mesh_shape = self._convert_unsupported_to_mesh(shape)

            if converted_mesh_shape is not None:
                # Use the original shape's filename mapping if available, otherwise use converted shape's filename
                mesh_elem = ET.SubElement(geometry_elem, "mesh")

                filename = None
                if mesh_filename_map and shape in mesh_filename_map:
                    # Use exported mesh file path
                    filename = mesh_filename_map[shape]
                else:
                    # Fallback filename
                    filename = f"{type(shape).__name__.lower()}_converted.stl"

                mesh_elem.set("filename", filename)

                # No scale needed for converted meshes (they're already at correct size)
                return geometry_elem
            else:
                logger.warning(
                    f"Failed to convert unsupported shape type: {type(shape)}"
                )
                return None

        else:
            logger.warning(f"Unsupported shape type for URDF export: {type(shape)}")
            return None

    def _export_mesh_files(
        self, robot: Robot, urdf_path: Path, meshes_dir: str
    ) -> Dict:
        """Export mesh files for all MeshShape objects in the robot.

        Args:
            robot: The Robot object containing MeshShape links.
            urdf_path: Path to the URDF file being written.
            meshes_dir: Directory name for mesh files (relative to URDF).

        Returns:
            Dictionary mapping MeshShape objects to their exported filenames.
        """
        mesh_filename_map = {}
        exported_meshes = set()  # Track exported mesh content to avoid duplicates

        # Create meshes directory
        meshes_path = urdf_path.parent / meshes_dir
        meshes_path.mkdir(exist_ok=True)

        for link in robot.links():
            shape_to_export = None
            original_shape = link.shape

            if isinstance(link.shape, MeshShape):
                shape_to_export = link.shape
            elif isinstance(link.shape, (Capsule, Cone)):
                # Convert unsupported shapes to mesh for export
                converted_shape = self._convert_unsupported_to_mesh(link.shape)
                if converted_shape is not None:
                    shape_to_export = converted_shape
                    # Map the original shape to the converted mesh for URDF generation
                    mesh_filename_map[original_shape] = None  # Will be set below

            if shape_to_export is not None:
                # Generate a unique filename based on mesh content
                mesh_hash = self._get_mesh_hash(shape_to_export.collision_mesh)
                filename = f"mesh_{mesh_hash}.stl"
                relative_filename = f"{meshes_dir}/{filename}"

                # Only export if we haven't already exported this mesh content
                if mesh_hash not in exported_meshes:
                    mesh_file_path = meshes_path / filename

                    try:
                        # Export as binary STL
                        shape_to_export.collision_mesh.export(
                            str(mesh_file_path), file_type="stl"
                        )
                        logger.debug(f"Exported mesh file: {mesh_file_path}")
                        exported_meshes.add(mesh_hash)
                    except Exception as e:
                        logger.error(
                            f"Failed to export mesh file {mesh_file_path}: {e}"
                        )
                        continue

                # Map the shape to its relative filename (for URDF reference)
                if isinstance(original_shape, MeshShape):
                    mesh_filename_map[original_shape] = relative_filename
                else:
                    # For converted shapes, we need to map both original and converted
                    mesh_filename_map[original_shape] = relative_filename
                    mesh_filename_map[shape_to_export] = relative_filename

        return mesh_filename_map

    def _get_mesh_hash(self, mesh) -> str:
        """Generate a hash for a mesh based on its vertices and faces.

        Args:
            mesh: The trimesh.Trimesh object.

        Returns:
            Hex string hash of the mesh content.
        """
        # Create hash from vertices and faces for uniqueness
        vertices_bytes = mesh.vertices.tobytes()
        faces_bytes = mesh.faces.tobytes()
        combined = vertices_bytes + faces_bytes

        return hashlib.md5(combined).hexdigest()[
            :8
        ]  # Use first 8 chars for readability

    def _convert_unsupported_to_mesh(self, shape) -> Optional[MeshShape]:
        """Convert unsupported shape types to MeshShape for URDF export.

        Args:
            shape: The unsupported shape object.

        Returns:
            MeshShape object with equivalent geometry, or None if conversion fails.
        """
        try:
            mesh = None
            filename = None

            if isinstance(shape, Capsule):
                # Create capsule mesh using trimesh
                mesh = trimesh.creation.capsule(
                    radius=shape.radius, height=shape.height
                )
                filename = f"capsule_r{shape.radius:.3f}_h{shape.height:.3f}.stl"

            elif isinstance(shape, Cone):
                # Create cone mesh using trimesh
                mesh = trimesh.creation.cone(radius=shape.radius, height=shape.height)

                # Fix coordinate system mismatch:
                # trimesh cone: base at z=0, apex at z=height
                # linkmotion Cone: base at z=-height/2, apex at z=+height/2
                # Need to translate mesh by -height/2 in Z direction
                z_offset = -shape.height / 2.0
                mesh.apply_translation([0, 0, z_offset])

                filename = f"cone_r{shape.radius:.3f}_h{shape.height:.3f}.stl"

            else:
                logger.warning(f"Cannot convert unsupported shape type: {type(shape)}")
                return None

            if mesh is not None:
                # Create MeshShape with the same properties as the original shape
                mesh_shape = MeshShape(
                    collision_mesh=mesh,
                    visual_mesh=None,  # Use collision mesh for visual too
                    default_transform=shape.default_transform,
                    color=shape.color,
                )

                logger.info(
                    f"Converted {type(shape).__name__} to mesh geometry: {filename}"
                )
                return mesh_shape

        except Exception as e:
            logger.error(f"Failed to convert {type(shape)} to mesh: {e}")

        return None

    def _create_origin_element(self, shape) -> Optional[ET.Element]:
        """Create an origin XML element from a shape's transform.

        Args:
            shape: The shape object.

        Returns:
            Origin XML element or None if no transform.
        """
        transform = getattr(shape, "default_transform", None)
        if transform is None:
            return None

        origin_elem = ET.Element("origin")

        # Add translation
        translation = transform.position
        if not np.allclose(translation, [0, 0, 0]):
            xyz_str = " ".join(f"{x:.6f}" for x in translation)
            origin_elem.set("xyz", xyz_str)

        # Add rotation (convert to RPY)
        rpy = transform.rotation.as_euler("xyz", degrees=False)
        if not np.allclose(rpy, [0, 0, 0]):
            rpy_str = " ".join(f"{x:.6f}" for x in rpy)
            origin_elem.set("rpy", rpy_str)

        # Only return the element if it has attributes
        if origin_elem.attrib:
            return origin_elem
        return None

    def _create_origin_element_from_transform(self, transform: Transform) -> ET.Element:
        """Create an origin XML element from a Transform object.

        Args:
            transform: The Transform object to convert.

        Returns:
            Origin XML element or None if transform is identity.
        """
        origin_elem = ET.Element("origin")

        # If the transform is identity, return with zero attributes
        if transform.is_identity():
            origin_elem.set("rpy", "0 0 0")
            origin_elem.set("xyz", "0 0 0")
            return origin_elem

        # Add translation
        translation = transform.position
        if not np.allclose(translation, [0, 0, 0]):
            xyz_str = " ".join(f"{x:.6f}" for x in translation)
            origin_elem.set("xyz", xyz_str)

        # Add rotation (convert to RPY)
        rpy = transform.rotation.as_euler("xyz", degrees=False)
        if not np.allclose(rpy, [0, 0, 0]):
            rpy_str = " ".join(f"{x:.6f}" for x in rpy)
            origin_elem.set("rpy", rpy_str)

        # Only return the element if it has attributes
        if origin_elem.attrib:
            return origin_elem

        # If no attributes were added, return identity
        origin_elem.set("rpy", "0 0 0")
        origin_elem.set("xyz", "0 0 0")
        return origin_elem

    def _create_joint_element(
        self,
        joint: Joint,
        relative_transform: Optional[Transform] = None,
        relative_axis: Optional[np.ndarray] = None,
    ) -> ET.Element:
        """Create a joint XML element.

        Args:
            joint: The Joint object.

        Returns:
            Joint XML element.
        """
        joint_elem = ET.Element("joint")
        joint_elem.set("name", joint.name)
        joint_elem.set("type", self._joint_type_map[joint.type])

        # Add parent link
        parent_elem = ET.SubElement(joint_elem, "parent")
        parent_elem.set("link", joint.parent_link_name)

        # Add child link
        child_elem = ET.SubElement(joint_elem, "child")
        child_elem.set("link", joint.child_link_name)

        # Add origin using the computed relative transform
        if relative_transform is not None and not relative_transform.is_identity():
            origin_elem = self._create_origin_element_from_transform(relative_transform)
            joint_elem.append(origin_elem)
        else:
            # If no relative transform, add identity origin
            origin_elem = ET.Element("origin")
            origin_elem.set("rpy", "0 0 0")
            origin_elem.set("xyz", "0 0 0")
            joint_elem.append(origin_elem)
        # Add axis for non-fixed joints using the computed relative axis
        if joint.type != JointType.FIXED and relative_axis is not None:
            axis_elem = ET.SubElement(joint_elem, "axis")
            xyz_str = " ".join(f"{x:.6f}" for x in relative_axis)
            axis_elem.set("xyz", xyz_str)

        # Add limits for revolute and prismatic joints
        if joint.type in {JointType.REVOLUTE, JointType.PRISMATIC}:
            if not (joint.min == -float("inf") and joint.max == float("inf")):
                limit_elem = ET.SubElement(joint_elem, "limit")
                if joint.min != -float("inf"):
                    limit_elem.set("lower", f"{joint.min:.6f}")
                if joint.max != float("inf"):
                    limit_elem.set("upper", f"{joint.max:.6f}")
                # Add default effort and velocity if not specified
                limit_elem.set("effort", "10.0")
                limit_elem.set("velocity", "1.0")

        return joint_elem

    def _indent_xml(self, elem: ET.Element, level: int = 0) -> None:
        """Add indentation to XML elements for pretty printing with line breaks between major elements.

        Args:
            elem: The XML element to indent.
            level: Current indentation level.
        """
        # Major elements that should have extra line breaks before them
        major_elements = {"material", "link", "joint"}

        indent = "\n" + "  " * level

        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent

            # Process children
            for i, child in enumerate(elem):
                self._indent_xml(child, level + 1)

                # Add extra line break before major elements (except the first child)
                if i > 0 and child.tag in major_elements and level == 0:
                    # Add an extra newline before major elements at root level
                    child.tail = "\n" + child.tail if child.tail else "\n" + indent

            # Set the tail of the last child
            if len(elem) > 0:
                last_child = elem[-1]
                if not last_child.tail or not last_child.tail.strip():
                    last_child.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent
