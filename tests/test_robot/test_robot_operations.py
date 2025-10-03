import pytest

from linkmotion.robot import Robot, Link, Joint, JointType


@pytest.fixture
def simple_robot():
    """Provides a simple robot with a single kinematic chain."""
    robot = Robot()
    links = [Link.from_sphere(f"L{i}", 1) for i in range(4)]
    joints = [
        Joint(
            f"J{i + 1}",
            JointType.FLOATING,
            parent_link_name=f"L{i}",
            child_link_name=f"L{i + 1}",
        )
        for i in range(3)
    ]
    for link in links:
        robot.add_link(link)
    for joint in joints:
        robot.add_joint(joint)
    return robot


def test_rename_link(simple_robot: Robot):
    """Tests renaming a link and verifies all connections are updated."""
    simple_robot.rename_link("L2", "L2_renamed")

    # Check if link was renamed
    assert simple_robot.link("L2_renamed").name == "L2_renamed"
    with pytest.raises(ValueError):
        simple_robot.link("L2")

    # Check connections
    assert simple_robot.joint("J2").child_link_name == "L2_renamed"
    assert simple_robot.joint("J3").parent_link_name == "L2_renamed"
    l2_renamed_parent_joint = simple_robot.parent_joint("L2_renamed")
    assert l2_renamed_parent_joint is not None
    assert l2_renamed_parent_joint.name == "J2"
    assert simple_robot.child_joints("L2_renamed")[0].name == "J3"


def test_divide_link(simple_robot: Robot):
    """Tests dividing a link into two new links and a new joint."""
    simple_robot.divide_link(
        "L2",
        parent_link=Link.from_sphere("L2_parent", 1),
        child_link=Link.from_sphere("L2_child", 1),
        new_joint=Joint(
            "J_new",
            JointType.FLOATING,
            parent_link_name="L2_parent",
            child_link_name="L2_child",
        ),
    )

    # Old link should be gone
    with pytest.raises(ValueError):
        simple_robot.link("L2")

    # New components should exist
    assert simple_robot.link("L2_parent")
    assert simple_robot.link("L2_child")
    assert simple_robot.joint("J_new")

    # Check wiring
    assert simple_robot.joint("J2").child_link_name == "L2_parent"
    assert simple_robot.joint("J3").parent_link_name == "L2_child"
    l3_parent_joint = simple_robot.parent_joint("L3")
    assert l3_parent_joint is not None
    assert l3_parent_joint.parent_link_name == "L2_child"


def test_concatenate_robot(simple_robot: Robot):
    """Tests merging two robots together."""
    robot2 = Robot()
    robot2.add_link(Link.from_sphere("tool_base", 1))
    robot2.add_link(Link.from_sphere("tool_tip", 1))
    robot2.add_joint(
        Joint(
            "tool_joint",
            JointType.FLOATING,
            parent_link_name="tool_base",
            child_link_name="tool_tip",
        )
    )

    # Connect tool to the end of simple_robot's arm
    connecting_joint = Joint(
        "wrist", JointType.FLOATING, parent_link_name="L3", child_link_name="tool_base"
    )
    simple_robot.concatenate_robot(robot2, connecting_joint)

    assert len(simple_robot.links()) == 6
    assert simple_robot.joint("wrist")
    tool_base_parent_joint = simple_robot.parent_joint("tool_base")
    assert tool_base_parent_joint is not None
    assert tool_base_parent_joint.name == "wrist"
    assert simple_robot.child_joints("L3")[0].name == "wrist"


def test_remove_joint(simple_robot: Robot):
    """Tests removing a joint from the robot."""
    # Remove J2 joint
    simple_robot.remove_joint("J2")

    # Joint should be gone
    with pytest.raises(ValueError):
        simple_robot.joint("J2")

    # Link L2 should no longer have a parent joint
    assert simple_robot.parent_joint("L2") is None

    # Link L1 should no longer have L2 as a child
    assert len(simple_robot.child_joints("L1")) == 0

    # Other joints should still exist
    assert simple_robot.joint("J1")
    assert simple_robot.joint("J3")


def test_remove_joint_nonexistent(simple_robot: Robot):
    """Tests removing a non-existent joint raises ValueError."""
    with pytest.raises(ValueError, match="Joint 'nonexistent' not found"):
        simple_robot.remove_joint("nonexistent")


def test_remove_link_with_descendants(simple_robot: Robot):
    """Tests removing a link and all its descendants."""
    # Remove L1 and all its descendants (L2, L3)
    simple_robot.remove_link_with_descendants("L1")

    # L1, L2, L3 should be gone
    with pytest.raises(ValueError):
        simple_robot.link("L1")
    with pytest.raises(ValueError):
        simple_robot.link("L2")
    with pytest.raises(ValueError):
        simple_robot.link("L3")

    # Joints connecting them should be gone
    with pytest.raises(ValueError):
        simple_robot.joint("J1")
    with pytest.raises(ValueError):
        simple_robot.joint("J2")
    with pytest.raises(ValueError):
        simple_robot.joint("J3")

    # Only L0 should remain as a root
    assert len(simple_robot.links()) == 1
    assert simple_robot.link("L0")
    assert len(simple_robot.joints()) == 0


def test_remove_link_with_descendants_root(simple_robot: Robot):
    """Tests removing a root link with descendants."""
    simple_robot.remove_link_with_descendants("L0")

    # All links should be gone
    assert len(simple_robot.links()) == 0
    assert len(simple_robot.joints()) == 0


def test_remove_link_with_descendants_leaf(simple_robot: Robot):
    """Tests removing a leaf link (no descendants)."""
    simple_robot.remove_link_with_descendants("L3")

    # Only L3 should be gone
    with pytest.raises(ValueError):
        simple_robot.link("L3")

    # J3 should be gone
    with pytest.raises(ValueError):
        simple_robot.joint("J3")

    # Other links should still exist
    assert simple_robot.link("L0")
    assert simple_robot.link("L1")
    assert simple_robot.link("L2")
    assert len(simple_robot.links()) == 3


def test_solidify_link():
    """Tests solidifying a link and its descendants into a single mesh."""
    from linkmotion.robot.shape.mesh import MeshShape
    import trimesh

    # Create a robot with mesh links
    robot = Robot()
    mesh1 = trimesh.creation.box(extents=[1, 1, 1])
    mesh2 = trimesh.creation.box(extents=[1, 1, 1])
    mesh3 = trimesh.creation.box(extents=[1, 1, 1])

    link1 = Link.from_mesh("mesh_link1", mesh1)
    link2 = Link.from_mesh("mesh_link2", mesh2)
    link3 = Link.from_mesh("mesh_link3", mesh3)

    robot.add_link(link1)
    robot.add_link(link2)
    robot.add_link(link3)

    joint1 = Joint(
        "joint1",
        JointType.FIXED,
        parent_link_name="mesh_link1",
        child_link_name="mesh_link2",
    )
    joint2 = Joint(
        "joint2",
        JointType.FIXED,
        parent_link_name="mesh_link2",
        child_link_name="mesh_link3",
    )

    robot.add_joint(joint1)
    robot.add_joint(joint2)

    # Solidify link1 and its descendants
    robot.solidify_link("mesh_link1")

    # Should have only one link now
    assert len(robot.links()) == 1
    assert robot.link("mesh_link1")

    # The link should be a MeshShape
    assert isinstance(robot.link("mesh_link1").shape, MeshShape)

    # Should have no joints
    assert len(robot.joints()) == 0


def test_solidify_link_with_non_mesh_link():
    """Tests that solidifying a link with non-mesh links raises ValueError."""
    robot = Robot()
    link1 = Link.from_sphere("sphere_link1", 1)
    link2 = Link.from_sphere("sphere_link2", 1)

    robot.add_link(link1)
    robot.add_link(link2)

    joint = Joint(
        "joint1",
        JointType.FIXED,
        parent_link_name="sphere_link1",
        child_link_name="sphere_link2",
    )
    robot.add_joint(joint)

    # Should raise ValueError because links are not MeshShapes
    with pytest.raises(
        ValueError, match="does not have a mesh shape and cannot be solidified"
    ):
        robot.solidify_link("sphere_link1")


def test_solidify_link_with_parent_joint():
    """Tests solidifying a link that has a parent joint."""
    import trimesh

    robot = Robot()
    mesh1 = trimesh.creation.box(extents=[1, 1, 1])
    mesh2 = trimesh.creation.box(extents=[1, 1, 1])
    mesh3 = trimesh.creation.box(extents=[1, 1, 1])

    link0 = Link.from_mesh("mesh_link0", mesh1)
    link1 = Link.from_mesh("mesh_link1", mesh2)
    link2 = Link.from_mesh("mesh_link2", mesh3)

    robot.add_link(link0)
    robot.add_link(link1)
    robot.add_link(link2)

    joint0 = Joint(
        "joint0",
        JointType.FIXED,
        parent_link_name="mesh_link0",
        child_link_name="mesh_link1",
    )
    joint1 = Joint(
        "joint1",
        JointType.FIXED,
        parent_link_name="mesh_link1",
        child_link_name="mesh_link2",
    )

    robot.add_joint(joint0)
    robot.add_joint(joint1)

    # Solidify link1 and its descendants
    robot.solidify_link("mesh_link1")

    # Should have two links: link0 and the solidified link1
    assert len(robot.links()) == 2
    assert robot.link("mesh_link0")
    assert robot.link("mesh_link1")

    # Should have one joint connecting them
    assert len(robot.joints()) == 1
    assert robot.joint("joint0")
    assert robot.joint("joint0").parent_link_name == "mesh_link0"
    assert robot.joint("joint0").child_link_name == "mesh_link1"
