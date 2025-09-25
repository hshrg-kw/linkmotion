import numpy as np
from scipy.spatial.transform import Rotation as R

from linkmotion import (
    Robot,
    Link,
    MoveManager,
    CollisionManager,
    Transform,
    Joint,
    JointType,
)


def test_collision():
    rot = R.from_rotvec(np.radians(30) * np.array([1, 0, 0]))
    pos = np.array([10, 0, 0])
    transform = Transform(rot, pos)

    pos2 = np.array([10, 5, 3])
    radius = 5
    height = 10

    box = Link.from_box("box", pos2, transform)
    sphere = Link.from_sphere("sphere", radius, pos2)
    cylinder = Link.from_cylinder("cylinder", radius, height, transform)
    cone = Link.from_cone("cone", radius, height, transform)
    capsule = Link.from_capsule("capsule", radius, height, transform)
    sphere_joint = Joint(
        name="sphere_joint",
        type=JointType.FLOATING,
        parent_link_name="box",
        child_link_name="sphere",
    )

    robot = Robot()
    robot.add_link(box)
    robot.add_link(sphere)
    robot.add_link(cylinder)
    robot.add_link(cone)
    robot.add_link(capsule)
    robot.add_joint(sphere_joint)

    mm = MoveManager(robot)
    mm.move("sphere_joint", Transform(translate=np.array([0, -0, 20])))

    cm = CollisionManager(mm)
    res_box = cm.distance({"sphere"}, {"box"}).min_distance
    assert abs(res_box - 15.762793936372404) < 1.0e-4
    res_cyl = cm.distance({"sphere"}, {"cylinder"}).min_distance
    assert abs(res_cyl - 11.477648100999037) < 1.0e-4
    res_con = cm.distance({"sphere"}, {"cone"}).min_distance
    assert abs(res_con - 15.11999494749214) < 1.0e-4
    res_cap = cm.distance({"sphere"}, {"capsule"}).min_distance
    assert abs(res_cap - 10.119993964451854) < 1.0e-4
