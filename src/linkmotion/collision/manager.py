import logging
import fcl

from linkmotion.move.manager import MoveManager

logger = logging.getLogger(__name__)


class CollisionManager:
    """Manages collision detection and distance queries between robot links.

    This class uses the Flexible Collision Library (FCL) to perform efficient
    collision and distance checks. It can handle checks between two sets of
    links, or self-collision checks within a single set of links.

    Attributes:
        mm (MoveManager): An instance of MoveManager that provides access to the
            robot's state and collision geometry.
        robot (Robot): The robot model instance, derived from the MoveManager.
    """

    def __init__(self, mm: MoveManager):
        """Initializes the CollisionManager.

        Args:
            mm (MoveManager): A configured MoveManager instance for a robot.
        """
        self.mm = mm
        self.robot = mm.robot
        logger.info("CollisionManager initialized.")

    def __repr__(self) -> str:
        """Returns a string representation of the CollisionManager.

        Returns:
            str: A string representation showing the robot and available links.
        """
        link_count = len(self.robot.links())
        return f"CollisionManager(links={link_count})"

    def copy(self) -> "CollisionManager":
        """Creates a copy of the CollisionManager.

        MoveManager is copied to ensure the original state remains unchanged.
        Robot is shallow copied as it is typically immutable.

        Returns:
            CollisionManager: A new instance with a copied MoveManager.
        """
        return CollisionManager(self.mm.copy())

    def _create_broad_phase_manager(
        self, link_names: set[str]
    ) -> fcl.DynamicAABBTreeCollisionManager:
        """Creates and configures a broad-phase collision manager for a set of links."""
        manager = fcl.DynamicAABBTreeCollisionManager()
        try:
            # Retrieve collision objects for the specified link names
            objects = [self.mm.get_link_collision_obj(n) for n in link_names]
        except ValueError as e:
            # Re-raise with more context if a link name is not found
            logger.error(f"Failed to create broad-phase manager: {e}")
            raise ValueError(
                f"Invalid link name provided for collision check. Original error: {e}"
            ) from e

        manager.registerObjects(objects)
        manager.setup()
        return manager

    def distance(
        self,
        names1: set[str],
        names2: set[str],
        enable_nearest_points: bool = False,
    ) -> fcl.DistanceResult:
        """Computes the minimum distance between two sets of robot links.

        This method supports two modes:
        1.  A direct, single-pair check if both sets contain exactly one link.
        2.  A broad-phase check using AABB trees for multi-link comparisons.

        Args:
            names1 (set[str]): A set of link names for the first collision group.
            names2 (set[str]): A set of link names for the second collision group.
            enable_nearest_points (bool): If True, computes the nearest points
                on the two closest objects.

        Returns:
            fcl.DistanceResult: An object containing the minimum distance and,
                optionally, the nearest points.

        Raises:
            ValueError: If either `names1` or `names2` is an empty list.
        """
        if not names1 or not names2:
            raise ValueError("Link name sets for distance checking must not be empty.")

        logger.debug(
            f"Computing distance between group 1 ({len(names1)} links) and group 2 ({len(names2)} links)."
        )

        request = fcl.DistanceRequest(enable_nearest_points=enable_nearest_points)
        result = fcl.DistanceResult()

        # Optimization: Use a direct check if it's a single pair comparison.
        if len(names1) == 1 and len(names2) == 1:
            o1 = self.mm.get_link_collision_obj(next(iter(names1)))
            o2 = self.mm.get_link_collision_obj(next(iter(names2)))
            fcl.distance(o1, o2, request, result)
            return result
        else:
            # Use broad-phase managers for multi-link comparisons.
            manager1 = self._create_broad_phase_manager(names1)
            manager2 = self._create_broad_phase_manager(names2)
            data = fcl.DistanceData(request=request, result=result)
            manager1.distance(manager2, data, fcl.defaultDistanceCallback)
            return data.result

    def collide(
        self,
        names1: set[str],
        names2: set[str],
        num_max_contacts: int = 1,
    ) -> fcl.CollisionResult:
        """Checks for collision between two sets of robot links.

        This method supports two modes:
        1.  A direct, single-pair check if both sets contain exactly one link.
        2.  A broad-phase check using AABB trees for multi-link comparisons.

        Args:
            names1 (set[str]): A set of link names for the first collision group.
            names2 (set[str]): A set of link names for the second collision group.
            num_max_contacts (int): The maximum number of contact points to report.

        Returns:
            fcl.CollisionResult: An object indicating if a collision occurred and
                containing contact information.

        Raises:
            ValueError: If either `names1` or `names2` is an empty list.
        """
        if not names1 or not names2:
            raise ValueError("Link name sets for collision checking must not be empty.")

        logger.debug(
            f"Checking collision between group 1 ({len(names1)} links) and group 2 ({len(names2)} links)."
        )

        request = fcl.CollisionRequest(
            num_max_contacts=num_max_contacts, enable_contact=True
        )
        result = fcl.CollisionResult()

        # Optimization: Use a direct check if it's a single pair comparison.
        if len(names1) == 1 and len(names2) == 1:
            o1 = self.mm.get_link_collision_obj(next(iter(names1)))
            o2 = self.mm.get_link_collision_obj(next(iter(names2)))
            fcl.collide(o1, o2, request, result)
            return result
        else:
            # Use broad-phase managers for multi-link comparisons.
            manager1 = self._create_broad_phase_manager(names1)
            manager2 = self._create_broad_phase_manager(names2)
            data = fcl.CollisionData(request=request, result=result)
            manager1.collide(manager2, data, fcl.defaultCollisionCallback)
            return data.result

    def self_collide(
        self,
        names: set[str],
        num_max_contacts: int = 1,
    ) -> fcl.CollisionResult:
        """Checks for self-collisions within a single set of robot links.

        Args:
            names (set[str]): A set of link names to check for self-collision.
            num_max_contacts (int): The maximum number of contact points to report.

        Returns:
            fcl.CollisionResult: An object indicating if a collision occurred.

        Raises:
            ValueError: If `names` is an empty list.
        """
        if not names:
            raise ValueError(
                "Link name set for self-collision checking must not be empty."
            )

        logger.debug(f"Checking self-collision for a group of {len(names)} links.")

        manager = self._create_broad_phase_manager(names)
        request = fcl.CollisionRequest(
            num_max_contacts=num_max_contacts, enable_contact=True
        )
        result = fcl.CollisionResult()
        data = fcl.CollisionData(request=request, result=result)

        # Perform self-collision check
        manager.collide(data, fcl.defaultCollisionCallback)
        return data.result
