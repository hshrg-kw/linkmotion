"""
Basic Collision Detection Demo

A simple example demonstrating the core collision detection features of CollisionManager.
Shows distance calculation, collision checking, and self-collision detection.

Run with: uv run python examples/collision/basic_collision_detection.py
"""

import logging
import numpy as np

from linkmotion import (
    Robot,
    Link,
    MoveManager,
    CollisionManager,
    Joint,
    JointType,
    Transform,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_robot_arm() -> Robot:
    """Create a simple robot arm with multiple links for collision testing."""
    robot = Robot()

    # Base (fixed)
    base = Link.from_cylinder(
        name="base",
        radius=0.1,
        height=0.05,
        default_transform=Transform(translate=np.array([0, 0, 0])),
    )
    robot.add_link(base)

    # Upper arm
    upper_arm = Link.from_box(
        name="upper_arm",
        extents=np.array([0.05, 0.05, 0.3]),
        default_transform=Transform(translate=np.array([0, 0, 0.15])),
    )
    robot.add_link(upper_arm)

    # Lower arm
    lower_arm = Link.from_box(
        name="lower_arm",
        extents=np.array([0.04, 0.04, 0.25]),
        default_transform=Transform(translate=np.array([0, 0, 0.125])),
    )
    robot.add_link(lower_arm)

    # End effector
    end_effector = Link.from_sphere(
        name="end_effector", radius=0.03, center=np.array([0, 0, 0])
    )
    robot.add_link(end_effector)

    # Obstacle (separate from robot)
    obstacle = Link.from_box(
        name="obstacle",
        extents=np.array([0.1, 0.1, 0.2]),
        default_transform=Transform(translate=np.array([0.2, 0, 0.1])),
    )
    robot.add_link(obstacle)

    # Joints
    shoulder = Joint(
        name="shoulder",
        type=JointType.REVOLUTE,
        parent_link_name="base",
        child_link_name="upper_arm",
        center=np.array([0.0, 0.0, 0.025]),
        direction=np.array([0.0, 1.0, 0.0]),  # Y-axis rotation
    )
    robot.add_joint(shoulder)

    elbow = Joint(
        name="elbow",
        type=JointType.REVOLUTE,
        parent_link_name="upper_arm",
        child_link_name="lower_arm",
        center=np.array([0.0, 0.0, 0.3]),
        direction=np.array([0.0, 1.0, 0.0]),  # Y-axis rotation
    )
    robot.add_joint(elbow)

    wrist = Joint(
        name="wrist",
        type=JointType.REVOLUTE,
        parent_link_name="lower_arm",
        child_link_name="end_effector",
        center=np.array([0.0, 0.0, 0.25]),
        direction=np.array([1.0, 0.0, 0.0]),  # X-axis rotation
    )
    robot.add_joint(wrist)

    print(
        f"Created robot with {len(robot.links())} links and {len(robot._joint_dict)} joints"
    )
    return robot


def demonstrate_distance_calculation(cm: CollisionManager):
    """Demonstrate distance calculation between links."""
    print("\n=== Distance Calculation Demo ===")

    # Distance between non-adjacent links
    result = cm.distance({"base"}, {"lower_arm"})
    print(f"Distance between base and lower_arm: {result.min_distance:.4f}")

    # Distance between end effector and obstacle
    result = cm.distance({"end_effector"}, {"obstacle"})
    print(f"Distance between end_effector and obstacle: {result.min_distance:.4f}")

    # Distance between multiple links (using sets)
    result = cm.distance({"upper_arm", "lower_arm"}, {"obstacle"})
    print(f"Distance between arm links and obstacle: {result.min_distance:.4f}")


def demonstrate_collision_detection(cm: CollisionManager):
    """Demonstrate collision detection between links."""
    print("\n=== Collision Detection Demo ===")

    # Check collision between adjacent links (should not collide in default pose)
    result = cm.collide({"base"}, {"upper_arm"})
    print(f"Collision between base and upper_arm: {result.is_collision}")

    # Check collision between end effector and obstacle
    result = cm.collide({"end_effector"}, {"obstacle"})
    print(f"Collision between end_effector and obstacle: {result.is_collision}")

    # Check collision between arm and obstacle
    result = cm.collide({"upper_arm", "lower_arm"}, {"obstacle"})
    print(f"Collision between arm and obstacle: {result.is_collision}")


def demonstrate_self_collision(cm: CollisionManager):
    """Demonstrate self-collision detection within robot links."""
    print("\n=== Self-Collision Detection Demo ===")

    # Check self-collision among arm links
    robot_links = {"base", "upper_arm", "lower_arm", "end_effector"}
    result = cm.self_collide(robot_links)
    print(f"Self-collision in robot links: {result.is_collision}")

    # Check self-collision in just arm segments
    arm_links = {"upper_arm", "lower_arm", "end_effector"}
    result = cm.self_collide(arm_links)
    print(f"Self-collision in arm links: {result.is_collision}")


def demonstrate_with_movement(mm: MoveManager, cm: CollisionManager):
    """Demonstrate collision detection with robot movement."""
    print("\n=== Collision Detection with Movement ===")

    # Move arm towards obstacle
    print("\n--- Moving shoulder joint 60 degrees ---")
    mm.move("shoulder", np.pi / 3)  # 60 degrees

    # Check distances after movement
    result = cm.distance({"end_effector"}, {"obstacle"})
    print(
        f"Distance between end_effector and obstacle after shoulder move: {result.min_distance:.4f}"
    )

    result = cm.collide({"lower_arm"}, {"obstacle"})
    print(f"Collision between lower_arm and obstacle: {result.is_collision}")

    # Move elbow to create potential collision
    print("\n--- Moving elbow joint -90 degrees ---")
    mm.move("elbow", -np.pi / 2)  # -90 degrees

    result = cm.distance({"lower_arm"}, {"upper_arm"})
    print(
        f"Distance between lower_arm and upper_arm after elbow move: {result.min_distance:.4f}"
    )

    # Check for self-collision after extreme movement
    robot_links = {"base", "upper_arm", "lower_arm", "end_effector"}
    result = cm.self_collide(robot_links)
    print(f"Self-collision after extreme movement: {result.is_collision}")

    # Reset position
    print("\n--- Resetting to initial position ---")
    mm.reset_move()


def demonstrate_collision_manager():
    """Main demonstration of CollisionManager features."""
    print("=== CollisionManager Demo ===")

    # Create robot and managers
    robot = create_robot_arm()
    mm = MoveManager(robot)
    cm = CollisionManager(mm)

    print(f"CollisionManager: {cm}")
    print(f"Available links: {list(mm.link_name_to_id.keys())}")

    # Show initial link positions
    print("\n--- Initial Link Positions ---")
    for link_name in mm.link_name_to_id.keys():
        transform = mm.get_transform(link_name)
        pos = transform.position
        print(f"{link_name}: position=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # Demonstrate various collision detection features
    demonstrate_distance_calculation(cm)
    demonstrate_collision_detection(cm)
    demonstrate_self_collision(cm)
    demonstrate_with_movement(mm, cm)


def demonstrate_simple_shapes():
    """Demonstrate collision detection with simple geometric shapes."""
    print("\n=== Simple Shapes Collision Demo ===")

    robot = Robot()

    # Create two spheres
    sphere1 = Link.from_sphere("sphere1", radius=1.0, center=np.array([0, 0, 0]))
    sphere2 = Link.from_sphere("sphere2", radius=1.0, center=np.array([1.5, 0, 0]))

    # Create two boxes
    box1 = Link.from_box(
        "box1",
        extents=np.array([1, 1, 1]),
        default_transform=Transform(translate=np.array([0, 3, 0])),
    )
    box2 = Link.from_box(
        "box2",
        extents=np.array([1, 1, 1]),
        default_transform=Transform(translate=np.array([0.5, 3, 0])),
    )

    robot.add_link(sphere1)
    robot.add_link(sphere2)
    robot.add_link(box1)
    robot.add_link(box2)

    mm = MoveManager(robot)
    cm = CollisionManager(mm)

    print("--- Sphere Collision Tests ---")
    # Overlapping spheres (should collide)
    result = cm.collide({"sphere1"}, {"sphere2"})
    distance = cm.distance({"sphere1"}, {"sphere2"}).min_distance
    print(f"Spheres collision: {result.is_collision}, distance: {distance:.4f}")

    print("\n--- Box Collision Tests ---")
    # Overlapping boxes (should collide)
    result = cm.collide({"box1"}, {"box2"})
    distance = cm.distance({"box1"}, {"box2"}).min_distance
    print(f"Boxes collision: {result.is_collision}, distance: {distance:.4f}")

    print("\n--- Mixed Shape Tests ---")
    # Sphere vs box (no collision)
    result = cm.collide({"sphere1"}, {"box1"})
    distance = cm.distance({"sphere1"}, {"box1"}).min_distance
    print(f"Sphere vs box collision: {result.is_collision}, distance: {distance:.4f}")


def main():
    """Main demonstration function."""
    print("LinkMotion CollisionManager Demo")
    print("===============================")

    try:
        # Demonstrate with robot arm
        demonstrate_collision_manager()

        # Demonstrate with simple shapes
        demonstrate_simple_shapes()

        print("\nCollision detection demo completed successfully!")

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
