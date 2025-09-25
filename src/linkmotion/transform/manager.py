import logging
from collections import defaultdict, deque
from typing import Dict, Set, Optional, List

from .transform import Transform

logger = logging.getLogger(__name__)


class TransformManager:
    """Manages a hierarchy of transforms for a scene graph.

    This class handles parent-child relationships between nodes, stores their
    local transformations, and calculates their world transformations on demand.
    It uses a dirtying mechanism to avoid redundant calculations.
    """

    def __init__(self):
        """Initializes a new TransformManager."""
        # Maps a node's ID to its parent's ID.
        self.parent_map: Dict[int, int] = {}
        # Maps a node's ID to a set of its children's IDs.
        self.children_map: defaultdict[int, Set[int]] = defaultdict(set)

        # Stores the default local transform for each node.
        self.default_local_transforms: Dict[int, Transform] = {}
        # Stores the current local transform for each node.
        self.local_transforms: Dict[int, Transform] = {}
        # Caches the calculated world transform for each node.
        self.world_transforms: Dict[int, Transform] = {}
        # A set of node IDs whose world transforms need recalculation.
        self.dirty_nodes: Set[int] = set()

    def __repr__(self) -> str:
        """Returns a concise string representation for debugging."""
        return "TransformManager()"

    def _node_exists(self, node_id: int) -> bool:
        """Checks if a node exists in the manager.

        Args:
            node_id: The ID of the node to check.

        Returns:
            True if the node exists, False otherwise.
        """
        return node_id in self.default_local_transforms

    def _would_create_cycle(self, node_id: int, parent_id: int) -> bool:
        """Checks if adding a parent-child relationship would create a cycle.

        Args:
            node_id: The ID of the potential child node.
            parent_id: The ID of the potential parent node.

        Returns:
            True if the relationship would create a cycle, False otherwise.
        """
        # If node_id doesn't exist yet, it can't create a cycle
        if not self._node_exists(node_id):
            return False

        # Traverse up from parent_id to see if we reach node_id
        current = parent_id
        visited = set()
        while current is not None:
            if current == node_id:
                return True
            if current in visited:
                # We found a cycle in the existing structure (shouldn't happen if data is valid)
                logger.warning(
                    f"Detected existing cycle in transform hierarchy involving node {current}"
                )
                return True
            visited.add(current)
            current = self.parent_map.get(current)
        return False

    def add_node(
        self,
        node_id: int,
        parent_id: Optional[int] = None,
        default_transform: Optional[Transform] = None,
    ):
        """Adds a new node to the manager.

        Args:
            node_id: The unique identifier for the new node.
            parent_id: The ID of the parent node. If None, the node is a root.
            default_transform: The default local transform for the node.
                               If None, an identity transform is used.

        Raises:
            ValueError: If the node_id already exists, parent_id does not exist,
                       or adding the node would create a circular dependency.
        """
        if self._node_exists(node_id):
            raise ValueError(f"Node ID {node_id} already exists.")
        if parent_id is not None and not self._node_exists(parent_id):
            raise ValueError(f"Parent node ID {parent_id} does not exist.")

        # Check for circular dependencies
        if parent_id is not None and self._would_create_cycle(node_id, parent_id):
            raise ValueError(
                f"Adding node {node_id} as child of {parent_id} would create a circular dependency."
            )

        # Set up parent-child relationships
        if parent_id is not None:
            self.parent_map[node_id] = parent_id
            self.children_map[parent_id].add(node_id)
            logger.debug(f"Added node {node_id} as child of {parent_id}")
        else:
            logger.debug(f"Added root node {node_id}")

        # Use the provided default transform or create an identity one.
        d_transform = default_transform or Transform()

        # Store the default transform and initialize the current local transform with a copy.
        self.default_local_transforms[node_id] = d_transform
        self.local_transforms[node_id] = d_transform.copy()

        # New nodes are initially dirty and need their world transform calculated.
        self.dirty_nodes.add(node_id)

        logger.info(f"Node {node_id} added to transform hierarchy")

    def set_local_transform(self, node_id: int, transform: Transform):
        """Sets the local transform for a given node.

        This action marks the node and all its descendants as dirty,
        triggering a world transform recalculation on the next request.

        Args:
            node_id: The ID of the node to modify.
            transform: The new local transform.

        Raises:
            ValueError: If the node_id does not exist.
        """
        if not self._node_exists(node_id):
            raise ValueError(f"Node ID {node_id} does not exist.")

        # Only update and mark dirty if the transform actually changed
        if self.local_transforms[node_id] != transform:
            self.local_transforms[node_id] = transform
            self._mark_dirty_descendants(node_id)

    def _mark_dirty_descendants(self, node_id: int):
        """Marks a node and all its descendants as dirty.

        Uses a non-recursive, breadth-first approach to avoid stack
        overflow with deep hierarchies and ensure consistent traversal order.

        Args:
            node_id: The ID of the node to start marking from.
        """
        nodes_to_visit = deque([node_id])
        while nodes_to_visit:
            current_id = nodes_to_visit.popleft()
            if current_id not in self.dirty_nodes:  # Avoid redundant work
                self.dirty_nodes.add(current_id)
                # Add all children to the queue for processing
                nodes_to_visit.extend(self.children_map.get(current_id, set()))

    def get_world_transform(self, node_id: int) -> Transform:
        """Calculates and returns the world transform of a node.

        The result is cached. The transform is only recalculated if the node
        is marked as dirty.

        Args:
            node_id: The ID of the node.

        Returns:
            The calculated world transform of the node.

        Raises:
            ValueError: If the node_id does not exist.
            RecursionError: If there's excessive recursion depth (potential cycle).
        """
        if not self._node_exists(node_id):
            raise ValueError(f"Node ID {node_id} does not exist.")

        # If the node is not dirty, return the cached transform.
        if node_id not in self.dirty_nodes:
            return self.world_transforms[node_id]

        try:
            local_transform = self.local_transforms[node_id]
            parent_id = self.parent_map.get(node_id)

            if parent_id is None:
                # This is a root node; its world transform is its local transform.
                world_transform = local_transform.copy()
            else:
                # Recursively get the parent's world transform and apply the local one.
                parent_world_transform = self.get_world_transform(parent_id)
                world_transform = parent_world_transform.apply(local_transform)

            # Cache the new world transform and mark the node as clean.
            self.world_transforms[node_id] = world_transform
            self.dirty_nodes.discard(node_id)  # More efficient than remove()

            return world_transform

        except RecursionError as e:
            logger.error(
                f"Recursion limit exceeded while calculating world transform for node {node_id}. "
            )
            logger.error(
                "This may indicate a circular dependency in the transform hierarchy."
            )
            raise RecursionError(
                f"Recursion limit exceeded for node {node_id}. Possible circular dependency."
            ) from e

    def reset_node_transform(self, node_id: int):
        """Resets a node's local transform to its default value.

        Args:
            node_id: The ID of the node to reset.

        Raises:
            ValueError: If the node_id does not exist.
        """
        if not self._node_exists(node_id):
            raise ValueError(f"Node ID {node_id} does not exist.")

        # set_local_transform handles copying and dirty marking.
        self.set_local_transform(node_id, self.default_local_transforms[node_id].copy())

    def reset_all_transforms(self):
        """Resets all nodes' local transforms to their default values.

        This method is optimized to avoid redundant dirty marking by batching
        the reset operations.
        """
        node_count = len(self.default_local_transforms)
        logger.info(f"Resetting {node_count} transforms to defaults")

        # Reset all local transforms to their defaults
        for node_id, default_transform in self.default_local_transforms.items():
            self.local_transforms[node_id] = default_transform.copy()

        # Mark all nodes as dirty in one operation
        self.dirty_nodes.update(self.default_local_transforms.keys())

        logger.debug(f"Successfully reset all {node_count} transforms")

    def apply_relative_transform(self, node_id: int, relative_transform: Transform):
        """Applies a relative transform to a node's default transform.

        The new local transform is calculated by applying the relative
        transform to the node's original default transform.

        Args:
            node_id: The ID of the node to apply the transform to.
            relative_transform: The transform to apply relative to the default.

        Raises:
            ValueError: If the node_id does not exist.
        """
        if not self._node_exists(node_id):
            raise ValueError(f"Node ID {node_id} does not exist.")

        # Calculate the new local transform by applying the relative one to the default.
        new_local_transform = self.default_local_transforms[node_id].apply(
            relative_transform
        )
        self.set_local_transform(node_id, new_local_transform)

    def remove_node(self, node_id: int):
        """Removes a node and all its descendants from the manager.

        Args:
            node_id: The ID of the node to remove.

        Raises:
            ValueError: If the node_id does not exist.
        """
        if not self._node_exists(node_id):
            raise ValueError(f"Node ID {node_id} does not exist.")

        # Collect all descendants to remove
        nodes_to_remove = []
        nodes_to_visit = deque([node_id])
        while nodes_to_visit:
            current_id = nodes_to_visit.popleft()
            nodes_to_remove.append(current_id)
            nodes_to_visit.extend(self.children_map.get(current_id, set()))

        logger.info(
            f"Removing node {node_id} and {len(nodes_to_remove) - 1} descendants"
        )

        # Remove all collected nodes
        for node in nodes_to_remove:
            # Remove from parent's children list
            parent_id = self.parent_map.get(node)
            if parent_id is not None:
                self.children_map[parent_id].discard(node)
                del self.parent_map[node]

            # Clean up all data structures
            self.default_local_transforms.pop(node, None)
            self.local_transforms.pop(node, None)
            self.world_transforms.pop(node, None)
            self.dirty_nodes.discard(node)
            # Remove empty children sets
            if node in self.children_map and not self.children_map[node]:
                del self.children_map[node]

        logger.debug(
            f"Successfully removed {len(nodes_to_remove)} nodes from hierarchy"
        )

    def get_node_ids(self) -> List[int]:
        """Returns a list of all node IDs in the manager.

        Returns:
            A list of all registered node IDs.
        """
        return list(self.default_local_transforms.keys())

    def get_root_nodes(self) -> List[int]:
        """Returns a list of all root node IDs (nodes without parents).

        Returns:
            A list of root node IDs.
        """
        return [
            node_id
            for node_id in self.default_local_transforms
            if node_id not in self.parent_map
        ]

    def get_children(self, node_id: int) -> List[int]:
        """Returns a list of children for the given node.

        Args:
            node_id: The ID of the node to get children for.

        Returns:
            A list of child node IDs.

        Raises:
            ValueError: If the node_id does not exist.
        """
        if not self._node_exists(node_id):
            raise ValueError(f"Node ID {node_id} does not exist.")
        return list(self.children_map.get(node_id, set()))
