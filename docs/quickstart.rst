Quick Start
===========

This guide will get you up and running with LinkMotion in minutes.

Basic Robot Construction
------------------------

Create your first robot:

.. code-block:: python

   import numpy as np
   from scipy.spatial.transform import Rotation as R
   from linkmotion import Robot, Link, Joint, JointType, Transform

   # Create a new robot
   robot = Robot()

   # Add a base link
   base = Link.from_box("base", extents=np.array([2, 2, 0.5]))
   robot.add_link(base)

   # Add an arm link with transform
   arm_transform = Transform(translate=np.array([0, 0, 1]))
   arm = Link.from_cylinder("arm", radius=0.2, height=2, default_transform=arm_transform)
   robot.add_link(arm)

   # Add a joint to connect them
   joint = Joint(
      "arm_joint",
      JointType.REVOLUTE,
      parent_link_name="base",
      child_link_name="arm",
      center=np.array([0, 0, 0.25]),
      direction=np.array([0, 0, 1]),
      min_=-np.pi,
      max_=np.pi
   )
   robot.add_joint(joint)

   print(f"Robot has {len(robot.links)} links and {len(robot.joints)} joints")

Robot Movement
--------------

Control your robot's joints:

.. code-block:: python

   from linkmotion import MoveManager

   # Create movement manager
   move_manager = MoveManager(robot)

   # Move the arm joint to 45 degrees
   move_manager.move("arm_joint", np.pi/4)

Collision Detection
-------------------

Check for collisions:

.. code-block:: python

   from linkmotion import CollisionManager

   # Create collision manager
   collision_manager = CollisionManager(move_manager)

   # Add an obstacle
   obstacle = Link.from_sphere(
      "obstacle",
      radius=0.5,
      default_transform=Transform(translate=np.array([1, 0, 1])),
   )
   robot.add_link(obstacle)

   # Check collision distance
   distance = collision_manager.distance({"arm"}, {"obstacle"})
   print(f"Distance between arm and obstacle: {distance.min_distance}")

3D Visualization
----------------

Visualize your robot (in Jupyter notebooks):

.. code-block:: python

   from linkmotion.visual import RobotVisualizer

   RobotVisualizer(robot)

URDF Import/Export
------------------

Work with URDF files:

.. code-block:: python

   # Import from URDF
   robot_from_urdf = Robot.from_urdf_file("path/to/robot.urdf")

   # Export to URDF
   robot.to_urdf_file("my_robot.urdf")

What's Next?
------------

* Check the :doc:`complete API reference <api/modules>`
* Browse `example scripts <https://github.com/hshrg-kw/linkmotion/tree/main/examples>`_
* Try `interactive notebooks <https://github.com/hshrg-kw/linkmotion/tree/main/notebooks>`_

Key Concepts
------------

**Robot**: The main container for your robot model

**Link**: Physical components with geometry (box, sphere, cylinder, etc.)

**Joint**: Connections between links that allow movement

**Transform**: 3D spatial transformations (translation, rotation)

**MoveManager**: Controls joint positions and robot state

**CollisionManager**: Handles collision detection and safety checking