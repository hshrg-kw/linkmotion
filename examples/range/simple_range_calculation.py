#!/usr/bin/env python3
"""Simple example demonstrating range calculation features.

This example shows how to use the RangeCalculator to compute collision-free
ranges across multiple joint axes of a robot. The robot is defined manually
using primitive shapes.
"""

import logging
import numpy as np

from linkmotion.robot.robot import Robot
from linkmotion.robot.link import Link
from linkmotion.robot.joint import Joint, JointType
from linkmotion.range.range_cal import RangeCalculator

# Configure logging to see calculation progress
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def create_simple_robot() -> Robot:
    """Create a simple 3-link robot for range calculation demonstration.

    Creates a robot with:
    - Base platform (box)
    - First arm segment (cylinder)
    - Second arm segment (cylinder)
    - End effector (sphere)

    Returns:
        Robot: A simple robot with 3 revolute joints.
    """
    robot = Robot()

    # Create base platform
    base_link = Link.from_box(
        name="base_platform",
        extents=np.array([0.4, 0.4, 0.1]),  # 40mmx 40mmx 10cm
        color=np.array([0.5, 0.5, 0.5, 1.0]),  # Gray
    )
    robot.add_link(base_link)

    # Create first arm segment
    arm1_link = Link.from_cylinder(
        name="arm1",
        radius=0.05,  # 5cm radius
        height=0.3,  # 30cm height
        color=np.array([0.8, 0.2, 0.2, 1.0]),  # Red
    )
    robot.add_link(arm1_link)

    # Create second arm segment
    arm2_link = Link.from_cylinder(
        name="arm2",
        radius=0.04,  # 4cm radius
        height=0.25,  # 25cm height
        color=np.array([0.2, 0.8, 0.2, 1.0]),  # Green
    )
    robot.add_link(arm2_link)

    # Create end effector
    end_effector_link = Link.from_sphere(
        name="end_effector",
        radius=0.06,  # 6cm radius
        color=np.array([0.2, 0.2, 0.8, 1.0]),  # Blue
    )
    robot.add_link(end_effector_link)

    # Create joints
    # Base to first arm - revolute around Z-axis
    base_to_arm1 = Joint(
        name="base_to_arm1",
        type=JointType.REVOLUTE,
        parent_link_name="base_platform",
        child_link_name="arm1",
        center=np.array([0.0, 0.0, 0.05]),  # Top of base
        direction=np.array([0.0, 0.0, 1.0]),  # Z-axis rotation
        min_=-np.pi,
        max_=np.pi,
    )
    robot.add_joint(base_to_arm1)

    # First arm to second arm - revolute around Y-axis
    arm1_to_arm2 = Joint(
        name="arm1_to_arm2",
        type=JointType.REVOLUTE,
        parent_link_name="arm1",
        child_link_name="arm2",
        center=np.array([0.0, 0.0, 0.15]),  # Top of arm1
        direction=np.array([0.0, 1.0, 0.0]),  # Y-axis rotation
        min_=-np.pi / 2,  # -90 degrees
        max_=np.pi / 2,  # +90 degrees
    )
    robot.add_joint(arm1_to_arm2)

    # Second arm to end effector - revolute around Y-axis
    arm2_to_effector = Joint(
        name="arm2_to_effector",
        type=JointType.REVOLUTE,
        parent_link_name="arm2",
        child_link_name="end_effector",
        center=np.array([0.0, 0.0, 0.125]),  # Top of arm2
        direction=np.array([0.0, 1.0, 0.0]),  # Y-axis rotation
        min_=-np.pi / 2,  # -90 degrees
        max_=np.pi / 2,  # +90 degrees
    )
    robot.add_joint(arm2_to_effector)

    return robot


def main():
    """Demonstrate basic range calculation functionality."""
    print("Creating simple robot for range calculation...")

    # Create robot manually
    robot = create_simple_robot()
    print(f"Created robot: {robot}")

    # Define link groups to check for collisions
    # Check for self-collision between arm segments and base
    arm_links = {"arm1", "arm2"}
    base_links = {"base_platform"}

    # Create range calculator
    calculator = RangeCalculator(robot, arm_links, base_links)

    # Add joint axes for calculation
    # Use coarse sampling for faster demonstration
    joint1_points = np.linspace(-np.pi, np.pi, 12)  # 12 points for base rotation
    joint2_points = np.linspace(
        -np.pi / 2, np.pi / 2, 8
    )  # 8 points for arm1-arm2 joint

    calculator.add_axis("base_to_arm1", joint1_points)
    calculator.add_axis("arm1_to_arm2", joint2_points)

    print(f"Calculator setup: {calculator}")
    print(f"Axis names: {calculator.get_axis_names()}")
    print(
        f"Total configurations to evaluate: {len(joint1_points) * len(joint2_points)}"
    )

    # Execute range calculation
    print("\nStarting collision range calculation...")
    calculator.execute()

    # Access and analyze results
    if calculator.results is not None:
        print(f"\nResults shape: {calculator.results.shape}")

        # Count collision-free configurations
        collision_free = np.sum(calculator.results == 0.0)
        total_configs = calculator.results.size
        collision_free_percentage = (collision_free / total_configs) * 100

        print(
            f"Collision-free configurations: {collision_free}/{total_configs} "
            f"({collision_free_percentage:.1f}%)"
        )

        # Find collision-free configurations
        free_indices = np.where(calculator.results == 0.0)
        if len(free_indices[0]) > 0:
            print(f"\nFound {len(free_indices[0])} collision-free configurations:")
            # Show first few examples
            for i in range(min(5, len(free_indices[0]))):
                idx1, idx2 = free_indices[0][i], free_indices[1][i]
                joint1_value = joint1_points[idx1]
                joint2_value = joint2_points[idx2]
                print(
                    f"  Configuration {i + 1}: base_to_arm1={joint1_value:.3f} rad, "
                    f"arm1_to_arm2={joint2_value:.3f} rad"
                )
        else:
            print("\nNo collision-free configurations found!")
            print("This might indicate:")
            print("- Robot links are always in collision with these link groups")
            print("- Joint limits are too restrictive")
            print("- Survey points are too sparse")

    print("\nRange calculation demonstration completed!")


if __name__ == "__main__":
    main()
