import logging
from collections import deque, defaultdict
from typing import Dict, Iterator, List, Optional, Set

from linkmotion.robot.link import Link
from linkmotion.robot.joint import Joint, JointType
from linkmotion.transform.transform import Transform


logger = logging.getLogger(__name__)


class Robot:
    """Represents a robot as a collection of links and joints.

    This class models a robot's kinematic structure as a directed acyclic graph
    (a tree or a forest), where links are nodes and joints are edges. It provides
    methods for building the robot structure, querying its components, and
    traversing the kinematic tree.

    Attributes:
        _joint_dict: A dictionary mapping joint names to Joint objects.
        _link_dict: A dictionary mapping link names to Link objects.
        _child_link_to_parent_joint: A cache mapping a child link's name to its
            parent joint's name for O(1) parent lookups.
        _parent_link_to_child_joints: A cache mapping a parent link's name to
            a set of its child joints' names for O(1) children lookups.
    """

    def __init__(self):
        """Initializes an empty Robot model."""
        self._joint_dict: Dict[str, Joint] = {}
        self._link_dict: Dict[str, Link] = {}

        # Caches for efficient O(1) traversal of the robot's kinematic tree.
        # These are kept in sync by methods that modify the robot's structure.
        self._child_link_to_parent_joint: Dict[str, str] = {}
        self._parent_link_to_child_joints: Dict[str, Set[str]] = defaultdict(set)

    def __str__(self) -> str:
        """Returns a concise string summary of the robot."""
        return (
            f"Robot model with {len(self._link_dict)} links and "
            f"{len(self._joint_dict)} joints."
        )

    def add_link(self, link: Link):
        """Adds a link to the robot model.

        Args:
            link: The Link object to add.

        Raises:
            ValueError: If a link with the same name already exists.
        """
        if link.name in self._link_dict:
            err_msg = f"Link with name '{link.name}' already exists."
            logger.error(err_msg)
            raise ValueError(err_msg)
        self._link_dict[link.name] = link
        logger.info(f"Added link: '{link.name}'")

    def add_joint(self, joint: Joint):
        """Adds a joint to the robot model, connecting two existing links.

        Args:
            joint: The Joint object to add.

        Raises:
            ValueError: If a joint with the same name already exists, or if its
                        parent or child links are not found in the model.
        """
        if joint.name in self._joint_dict:
            err_msg = f"Joint with name '{joint.name}' already exists."
            logger.error(err_msg)
            raise ValueError(err_msg)
        if joint.parent_link_name not in self._link_dict:
            err_msg = f"Parent link '{joint.parent_link_name}' for joint '{joint.name}' not found."
            logger.error(err_msg)
            raise ValueError(err_msg)
        if joint.child_link_name not in self._link_dict:
            err_msg = f"Child link '{joint.child_link_name}' for joint '{joint.name}' not found."
            logger.error(err_msg)
            raise ValueError(err_msg)
        if joint.child_link_name in self._child_link_to_parent_joint:
            err_msg = f"Child link '{joint.child_link_name}' already has a parent joint. A link cannot have multiple parents."
            logger.error(err_msg)
            raise ValueError(err_msg)

        self._joint_dict[joint.name] = joint
        # Update caches to maintain data integrity for fast lookups.
        self._child_link_to_parent_joint[joint.child_link_name] = joint.name
        self._parent_link_to_child_joints[joint.parent_link_name].add(joint.name)
        logger.info(
            f"Added joint: '{joint.name}' connecting '{joint.parent_link_name}' -> '{joint.child_link_name}'"
        )

    def joint(self, joint_name: str) -> Joint:
        """Retrieves a joint by its name.

        Args:
            joint_name: The name of the joint to retrieve.

        Returns:
            The Joint object.

        Raises:
            ValueError: If the joint is not found.
        """
        try:
            return self._joint_dict[joint_name]
        except KeyError:
            err_msg = f"Joint '{joint_name}' not found in the robot model."
            logger.error(err_msg)
            raise ValueError(err_msg) from None

    def link(self, link_name: str) -> Link:
        """Retrieves a link by its name.

        Args:
            link_name: The name of the link to retrieve.

        Returns:
            The Link object.

        Raises:
            ValueError: If the link is not found.
        """
        try:
            return self._link_dict[link_name]
        except KeyError:
            err_msg = f"Link '{link_name}' not found in the robot model."
            logger.error(err_msg)
            raise ValueError(err_msg) from None

    def links(self) -> List[Link]:
        """Returns a list of all links in the robot.

        Returns:
            A list of all Link objects in the model.
        """
        return list(self._link_dict.values())

    def root_links(self) -> List[Link]:
        """Finds and returns all root links (links without a parent joint).

        Returns:
            A list of root Link objects.
        """
        # A link is a root if it's not a child in any joint connection.
        child_link_names = set(self._child_link_to_parent_joint.keys())
        return [
            link
            for name, link in self._link_dict.items()
            if name not in child_link_names
        ]

    def leaf_links(self) -> List[Link]:
        """Finds and returns all leaf links (links without child joints).

        Returns:
            A list of leaf Link objects.
        """
        # A link is a leaf if it's not a parent in any joint connection.
        parent_link_names = set(self._parent_link_to_child_joints.keys())
        return [
            link
            for name, link in self._link_dict.items()
            if name not in parent_link_names
        ]

    def parent_joint(self, link_name: str) -> Optional[Joint]:
        """Finds the parent joint of a given link using the cache.

        Args:
            link_name: The name of the link whose parent joint is sought.

        Returns:
            The parent Joint object, or None if the link is a root.
        """
        self.link(link_name)  # Ensures the link exists.
        parent_joint_name = self._child_link_to_parent_joint.get(link_name)
        if parent_joint_name:
            return self._joint_dict[parent_joint_name]
        return None

    def child_joints(self, link_name: str) -> List[Joint]:
        """Finds all child joints of a given link using the cache.

        Args:
            link_name: The name of the link whose child joints are sought.

        Returns:
            A list of child Joint objects. Returns an empty list if the link is a leaf.
        """
        self.link(link_name)  # Ensures the link exists.
        child_joint_names = self._parent_link_to_child_joints.get(link_name, set())
        return [self._joint_dict[name] for name in child_joint_names]

    def static_links(self) -> List[Link]:
        """Determines and returns all static links in the robot.

        A link is static if it's a root or connected to a static parent via a
        FIXED joint. This method traverses the kinematic tree(s) from the roots
        downwards, ensuring parents are processed before children.

        Returns:
            A list of Link objects that are determined to be static.
        """
        link_is_static: Dict[str, bool] = {}

        # The traverse_links generator ensures parent links are processed before their children.
        for link in self.traverse_links():
            parent_joint = self.parent_joint(link.name)

            if parent_joint is None:
                # Case 1: Root links are always considered static.
                link_is_static[link.name] = True
            else:
                # Case 2: The link has a parent.
                parent_link_name = parent_joint.parent_link_name
                # The parent's status must have been determined due to the traversal order.
                parent_is_static = link_is_static.get(parent_link_name, False)
                # This link is static if its parent is static AND the connecting joint is FIXED.
                is_fixed_joint = parent_joint.type == JointType.FIXED
                link_is_static[link.name] = parent_is_static and is_fixed_joint

        # Filter the dictionary to return the actual static Link objects.
        return [
            self.link(name) for name, is_static in link_is_static.items() if is_static
        ]

    def dynamic_links(self) -> List[Link]:
        """Returns all dynamic links in the robot.

        A dynamic link is any link that is not static. This is determined by
        taking the set difference between all links and static links.

        Returns:
            A list of all dynamic Link objects.
        """
        static_link_names = {link.name for link in self.static_links()}
        all_link_names = set(self._link_dict.keys())
        dynamic_link_names = all_link_names - static_link_names
        return [self.link(name) for name in dynamic_link_names]

    @staticmethod
    def from_other(other: "Robot", transform: Transform | None = None) -> "Robot":
        """Creates a new Robot instance as a deep copy of another.

        Args:
            other: The Robot instance to copy.

        Returns:
            A new, independent Robot instance.
        """
        if transform is None:
            transform = Transform()
        # Using copy.deepcopy is the most robust way to create an independent copy.
        new_robot = Robot()
        new_robot._joint_dict = {
            k: v.from_other(v, v.name, transform) for k, v in other._joint_dict.items()
        }
        new_robot._link_dict = {
            k: v.from_other(v, v.name, transform) for k, v in other._link_dict.items()
        }
        new_robot._child_link_to_parent_joint = other._child_link_to_parent_joint.copy()
        new_robot._parent_link_to_child_joints = defaultdict[str, set[str]](set)
        for k, v in other._parent_link_to_child_joints.items():
            new_robot._parent_link_to_child_joints[k] = v.copy()
        logger.info("Created a deep copy of a Robot instance.")
        return new_robot

    def traverse_child_links(
        self, start_link_name: str, include_self: bool = False
    ) -> Iterator[Link]:
        """Yields descendant links from a starting link using Breadth-First Search (BFS).

        Args:
            start_link_name: The name of the link to start traversal from.
            include_self: If True, the starting link is yielded first.

        Yields:
            Link objects in BFS order.

        Raises:
            ValueError: If the start_link_name is not found.
        """
        start_link = self.link(start_link_name)
        queue = deque([start_link])
        # Use a set for efficient O(1) 'in' checks.
        visited = {start_link_name}

        if include_self:
            yield start_link

        while queue:
            parent_link = queue.popleft()
            # Efficiently get child joints using the cache.
            for child_joint in self.child_joints(parent_link.name):
                child_link = self.link(child_joint.child_link_name)
                if child_link.name not in visited:
                    visited.add(child_link.name)
                    queue.append(child_link)
                    if include_self:
                        yield child_link
                    else:
                        # If not including self, only yield children
                        if parent_link.name == start_link_name:
                            yield child_link

    def traverse_parent_links(
        self, start_link_name: str, include_self: bool = False
    ) -> Iterator[Link]:
        """Yields ancestor links from a starting link up towards the root.

        Args:
            start_link_name: The name of the link to start traversal from.
            include_self: If True, the starting link is yielded first.

        Yields:
            Link objects, starting from the immediate parent to the root.

        Raises:
            ValueError: If the start_link_name is not found.
        """
        current_link = self.link(start_link_name)
        if include_self:
            yield current_link

        current_name = current_link.name
        # Loop while the current link has a parent, using the cache for efficiency.
        while current_name in self._child_link_to_parent_joint:
            parent_joint_name = self._child_link_to_parent_joint[current_name]
            parent_joint = self._joint_dict[parent_joint_name]
            parent_link = self.link(parent_joint.parent_link_name)
            yield parent_link
            current_name = parent_link.name

    def traverse_links(self) -> Iterator[Link]:
        """Yields all links in the model by traversing from the root(s).

        This generator iterates through each kinematic tree (forest), ensuring
        every link is visited exactly once. The traversal order is top-down (BFS).

        Yields:
            All Link objects within the robot model.
        """
        # A robot can be a "forest" of multiple kinematic trees.
        # This loop ensures we traverse every tree starting from its root.
        visited = set()
        for root_link in self.root_links():
            if root_link.name not in visited:
                # 'yield from' elegantly delegates the iteration to another generator.
                traversal_generator = self.traverse_child_links(
                    root_link.name, include_self=True
                )
                for link in traversal_generator:
                    if link.name not in visited:
                        visited.add(link.name)
                        yield link

    def rename_link(self, old_name: str, new_name: str):
        """Renames a link and updates all associated joints and caches.

        Args:
            old_name: The current name of the link.
            new_name: The new name for the link.

        Raises:
            ValueError: If `old_name` does not exist or `new_name` already exists.
        """
        if old_name == new_name:
            logger.debug(
                f"Skipping link rename: old and new names are identical ('{old_name}')."
            )
            return
        if new_name in self._link_dict:
            raise ValueError(f"Link with name '{new_name}' already exists.")

        link = self.link(old_name)  # Raises ValueError if old_name doesn't exist

        # 1. Update the link object itself and the main dictionary.
        link.name = new_name
        self._link_dict[new_name] = self._link_dict.pop(old_name)

        # 2. Update parent joint connection.
        if old_name in self._child_link_to_parent_joint:
            parent_joint_name = self._child_link_to_parent_joint.pop(old_name)
            self.joint(parent_joint_name).child_link_name = new_name
            self._child_link_to_parent_joint[new_name] = parent_joint_name

        # 3. Update child joints connection.
        if old_name in self._parent_link_to_child_joints:
            child_joint_names = self._parent_link_to_child_joints.pop(old_name)
            for joint_name in child_joint_names:
                self.joint(joint_name).parent_link_name = new_name
            self._parent_link_to_child_joints[new_name] = child_joint_names

        logger.info(
            f"Renamed link '{old_name}' to '{new_name}' and updated all connections."
        )

    def divide_link(
        self,
        link_name_to_be_divided: str,
        parent_link: Link,
        child_link: Link,
        new_joint: Joint,
    ):
        """Divides one link into two, connected by a new joint.

        This operation replaces a single link with a new parent link, a new
        joint, and a new child link, while preserving the original external
        connections.

        Original structure:
            parent_joint -> link_to_be_divided -> child_joints

        New structure:
            parent_joint -> parent_link -> new_joint -> child_link -> child_joints

        Args:
            link_name_to_be_divided: The name of the link to replace.
            parent_link: The new Link object that will become the parent part.
            child_link: The new Link object that will become the child part.
            new_joint: The new Joint connecting `parent_link` and `child_link`.

        Raises:
            ValueError: If new link/joint names conflict, or if the new joint
                        does not correctly connect the new links.
        """
        # 1. --- Pre-operation validation ---
        self.link(link_name_to_be_divided)  # Ensure the link to be divided exists.
        for name in (parent_link.name, child_link.name):
            if name in self._link_dict:
                raise ValueError(
                    f"New link name '{name}' conflicts with an existing link."
                )
        if new_joint.name in self._joint_dict:
            raise ValueError(
                f"New joint name '{new_joint.name}' conflicts with an existing joint."
            )
        if not (
            new_joint.parent_link_name == parent_link.name
            and new_joint.child_link_name == child_link.name
        ):
            raise ValueError(
                "The new joint must connect the new parent and child links."
            )

        # 2. --- Cache original connections before modifying the structure ---
        original_parent_joint = self.parent_joint(link_name_to_be_divided)
        original_child_joints = self.child_joints(link_name_to_be_divided)

        # 3. --- Atomically modify the robot structure ---
        # Remove the old link and its cache entries.
        self._link_dict.pop(link_name_to_be_divided)
        if link_name_to_be_divided in self._child_link_to_parent_joint:
            self._child_link_to_parent_joint.pop(link_name_to_be_divided)
        if link_name_to_be_divided in self._parent_link_to_child_joints:
            self._parent_link_to_child_joints.pop(link_name_to_be_divided)

        # Re-wire original parent joint to point to the new parent link.
        if original_parent_joint:
            original_parent_joint.child_link_name = parent_link.name
            self._child_link_to_parent_joint[parent_link.name] = (
                original_parent_joint.name
            )

        # Re-wire original child joints to point from the new child link.
        if original_child_joints:
            for cj in original_child_joints:
                cj.parent_link_name = child_link.name
            self._parent_link_to_child_joints[child_link.name] = {
                cj.name for cj in original_child_joints
            }

        # 4. --- Add the new components ---
        self.add_link(parent_link)
        self.add_link(child_link)
        self.add_joint(new_joint)

        logger.info(
            f"Divided link '{link_name_to_be_divided}' into '{parent_link.name}' and '{child_link.name}'."
        )

    def concatenate_robot(
        self,
        other_robot: "Robot",
        new_joint: Joint,
    ):
        """Merges another robot into this one by connecting them with a new joint.

        The other robot must have exactly one root link, which will become the
        child link of the new connecting joint.

        Args:
            other_robot: The robot to merge into this one.
            new_joint: The joint that connects a link from `self` (parent) to
                       the root link of `other_robot` (child).

        Raises:
            ValueError: If name conflicts exist between the robots, or if the
                        connection rules are violated (e.g., `other_robot` has
                        multiple roots).
        """
        # --- Validation ---
        if not (
            set(self._joint_dict.keys()).isdisjoint(other_robot._joint_dict.keys())
        ):
            raise ValueError("Joint name conflict detected between robots.")
        if not (set(self._link_dict.keys()).isdisjoint(other_robot._link_dict.keys())):
            raise ValueError("Link name conflict detected between robots.")

        other_roots = other_robot.root_links()
        if len(other_roots) != 1:
            raise ValueError(
                f"The other robot must have exactly one root link, but found {len(other_roots)}."
            )

        if new_joint.parent_link_name not in self._link_dict:
            raise ValueError(
                f"Parent link '{new_joint.parent_link_name}' of the new joint not found in the base robot."
            )
        if new_joint.child_link_name != other_roots[0].name:
            raise ValueError(
                f"Child link '{new_joint.child_link_name}' of the new joint must be the root of the other robot."
            )

        # --- Concatenation ---
        # Merge dictionaries. Python 3.9+ syntax `|` is clean.
        self._joint_dict.update(other_robot._joint_dict)
        self._link_dict.update(other_robot._link_dict)

        # Merge caches
        self._child_link_to_parent_joint.update(other_robot._child_link_to_parent_joint)
        for key, value in other_robot._parent_link_to_child_joints.items():
            self._parent_link_to_child_joints[key].update(value)

        # Add the connecting joint, which also updates the caches.
        self.add_joint(new_joint)
        logger.info(
            f"Successfully concatenated another robot, connecting via joint '{new_joint.name}'."
        )
