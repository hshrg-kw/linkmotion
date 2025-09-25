import pytest
import numpy as np

from linkmotion.robot import Link, Joint, JointType
from linkmotion.transform import Transform
from linkmotion.const import LARGE_NUM


@pytest.fixture
def link1():
    return Link.from_sphere("link1", 1)


@pytest.fixture
def link2():
    return Link.from_sphere("link2", 1)


@pytest.fixture
def revolute_joint(link1: Link, link2: Link):
    return Joint(
        name="joint1",
        type=JointType.REVOLUTE,
        parent_link_name=link1.name,
        child_link_name=link2.name,
        center=np.array([0.0, 0.0, 0.0]),
    )


class TestJoint:
    def test_init(self, revolute_joint: Joint):
        assert revolute_joint.name == "joint1"
        assert np.allclose(revolute_joint.direction, [0.0, 0.0, 1.0])

    def test_init_zero_direction_fails(self):
        with pytest.raises(ValueError, match="cannot be a zero vector"):
            Joint(
                "j", JointType.PRISMATIC, "c", "p", direction=np.array([0.0, 0.0, 0.0])
            )

    def test_init_revolute_no_center_fails(self):
        with pytest.raises(ValueError, match="requires a 'center'"):
            Joint("j", JointType.REVOLUTE, "c", "p", center=None)

    def test_init_continuous_no_center_fails(self):
        with pytest.raises(ValueError, match="requires a 'center'"):
            Joint("j", JointType.CONTINUOUS, "c", "p", center=None)

    def test_init_with_limits(self):
        joint = Joint(
            "limited_joint",
            JointType.REVOLUTE,
            "child",
            "parent",
            center=np.array([0, 0, 0]),
            min_=-1.57,
            max_=1.57,
        )
        assert joint.min == -1.57
        assert joint.max == 1.57

    def test_init_default_limits(self):
        joint = Joint(
            "default_limits",
            JointType.REVOLUTE,
            "child",
            "parent",
            center=np.array([0, 0, 0]),
        )
        assert joint.min == -LARGE_NUM
        assert joint.max == LARGE_NUM

    def test_direction_normalization(self):
        direction = np.array([3.0, 4.0, 0.0])  # Length = 5
        joint = Joint(
            "normalized", JointType.PRISMATIC, "child", "parent", direction=direction
        )
        expected = direction / np.linalg.norm(direction)
        assert np.allclose(joint.direction, expected)

    def test_str_representation(self, revolute_joint: Joint):
        str_repr = str(revolute_joint)
        assert "Joint" in str_repr
        assert "joint1" in str_repr
        assert "REVOLUTE" in str_repr
        assert "parent='link1'" in str_repr
        assert "child='link2'" in str_repr

    def test_from_other_creates_copy(self, revolute_joint: Joint):
        copied_joint = Joint.from_other(revolute_joint, "copied_joint")
        assert copied_joint.name == "copied_joint"
        assert copied_joint.type == revolute_joint.type
        assert copied_joint.parent_link_name == revolute_joint.parent_link_name
        assert copied_joint.child_link_name == revolute_joint.child_link_name
        assert np.allclose(copied_joint.direction, revolute_joint.direction)
        if revolute_joint.center is not None and copied_joint.center is not None:
            assert np.allclose(copied_joint.center, revolute_joint.center)
        assert copied_joint is not revolute_joint

    def test_from_other_with_transform(self, revolute_joint: Joint):
        # Test with transform - the direction will be transformed as a point
        transform = Transform(translate=np.array([1.0, 2.0, 3.0]))
        copied_joint = Joint.from_other(revolute_joint, "transformed_joint", transform)
        assert copied_joint.name == "transformed_joint"

        # The transform.apply() method transforms the direction as a point
        expected_direction = transform.apply(revolute_joint.direction)
        # Direction gets normalized in Joint.__init__, so we need to normalize expected too
        expected_direction = expected_direction / np.linalg.norm(expected_direction)
        assert np.allclose(copied_joint.direction, expected_direction)

        if revolute_joint.center is not None and copied_joint.center is not None:
            expected_center = transform.apply(revolute_joint.center)
            assert np.allclose(copied_joint.center, expected_center)

    def test_all_joint_types_creation(self, link1: Link, link2: Link):
        center = np.array([0, 0, 0])
        direction = np.array([1, 0, 0])

        # Test all joint types can be created
        joint_types = [
            (JointType.FIXED, {}),
            (JointType.REVOLUTE, {"center": center}),
            (JointType.CONTINUOUS, {"center": center}),
            (JointType.PRISMATIC, {"direction": direction}),
            (JointType.PLANAR, {}),
            (JointType.FLOATING, {}),
        ]

        for joint_type, kwargs in joint_types:
            joint = Joint(
                f"test_{joint_type.name.lower()}",
                joint_type,
                link2.name,
                link1.name,
                **kwargs,
            )
            assert joint.type == joint_type
