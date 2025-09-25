import pytest
import numpy as np

from linkmotion import Robot, Link, MoveManager, CollisionManager, Transform


@pytest.fixture
def collision_manager_setup():
    """Sets up a real Robot, MoveManager, and CollisionManager for testing."""
    # Build a simple robot
    robot = Robot()
    robot.add_link(Link.from_sphere(name="link1", radius=1, center=np.array([0, 0, 0])))
    robot.add_link(Link.from_sphere(name="link2", radius=1, center=np.array([0, 0, 1])))
    robot.add_link(Link.from_sphere(name="link3", radius=1, center=np.array([0, 0, 8])))

    mm = MoveManager(robot)
    cm = CollisionManager(mm)
    return cm


def test_init(collision_manager_setup: CollisionManager):
    """Tests that the CollisionManager initializes correctly."""
    cm = collision_manager_setup
    assert cm is not None
    assert isinstance(cm.mm, MoveManager)
    assert isinstance(cm.robot, Robot)


def test_distance_single_pair_check(collision_manager_setup: CollisionManager):
    """Tests the distance method's optimized path for a single pair of links."""
    cm = collision_manager_setup
    result = cm.distance({"link1"}, {"link2"})
    assert result.min_distance == -1.0


def test_distance_broad_phase_check(collision_manager_setup: CollisionManager):
    """Tests the distance method's broad-phase path for multiple links."""
    cm = collision_manager_setup
    # This case triggers the broad-phase logic (1 vs 2 links)
    result = cm.distance({"link1"}, {"link2", "link3"})
    assert result.min_distance == -1.0


def test_collide_single_pair_check(collision_manager_setup: CollisionManager):
    """Tests the collide method's optimized path for a single pair of links."""
    cm = collision_manager_setup
    result = cm.collide({"link1"}, {"link2"})
    assert result.is_collision is True


def test_collide_broad_phase_check(collision_manager_setup: CollisionManager):
    """Tests the collide method's broad-phase path for multiple links."""
    cm = collision_manager_setup
    result = cm.collide({"link1", "link2"}, {"link3"})
    assert result.is_collision is False


def test_self_collide(collision_manager_setup: CollisionManager):
    """Tests the self-collide method."""
    cm = collision_manager_setup
    result = cm.self_collide({"link1", "link2", "link3"})
    assert result.is_collision is True


def test_error_on_empty_lists(collision_manager_setup: CollisionManager):
    """Tests that methods raise ValueError for empty link lists."""
    cm = collision_manager_setup
    with pytest.raises(ValueError, match="must not be empty"):
        cm.distance(set(), {"link2"})
    with pytest.raises(ValueError, match="must not be empty"):
        cm.collide({"link1"}, set())
    with pytest.raises(ValueError, match="must not be empty"):
        cm.self_collide(set())


def test_error_on_invalid_link_name(collision_manager_setup: CollisionManager):
    """Tests that an invalid link name raises a descriptive ValueError."""
    cm = collision_manager_setup
    with pytest.raises(ValueError, match="Invalid link name provided"):
        cm.distance({"link1", "invalid_link"}, {"link2"})


# Defines collision and non-collision scenarios for each shape.
# Tuple structure for each test case:
# ( (Link1 definition), (Link2 definition), expected collision result (bool), expected distance if not colliding )
# Link definition: (Link factory, link name, arguments dict, position coordinates)
SHAPE_INTERACTION_TEST_CASES = [
    # --- Box vs Box ---
    # Colliding case
    (
        (Link.from_box, "box1", {"extents": np.array([1, 1, 1])}, [0, 0, 0]),
        (Link.from_box, "box2", {"extents": np.array([1, 1, 1])}, [0.5, 0, 0]),
        True,
        -0.5,
    ),
    # Non-colliding case (1.0 unit apart on the x-axis)
    (
        (Link.from_box, "box1", {"extents": np.array([1, 1, 1])}, [0, 0, 0]),
        (Link.from_box, "box2", {"extents": np.array([1, 1, 1])}, [2, 0, 0]),
        False,
        1.0,
    ),
    # --- Capsule vs Capsule ---
    # Colliding case
    (
        (Link.from_capsule, "cap1", {"radius": 0.5, "height": 2}, [0, 0, 0]),
        (Link.from_capsule, "cap2", {"radius": 0.5, "height": 2}, [0, 0, 2.9]),
        True,
        -0.1,
    ),
    # Non-colliding case (1.0 unit apart on the z-axis)
    (
        (Link.from_capsule, "cap1", {"radius": 0.5, "height": 2}, [0, 0, 0]),
        (Link.from_capsule, "cap2", {"radius": 0.5, "height": 2}, [0, 0, 4]),
        False,
        1.0,
    ),
    # --- Cylinder vs Cylinder ---
    # Colliding case
    (
        (Link.from_cylinder, "cyl1", {"radius": 1, "height": 2}, [0, 0, 0]),
        (Link.from_cylinder, "cyl2", {"radius": 1, "height": 2}, [0, 0, 1.9]),
        True,
        -0.1,
    ),
    # Non-colliding case (1.0 unit apart on the z-axis)
    (
        (Link.from_cylinder, "cyl1", {"radius": 1, "height": 2}, [0, 0, 0]),
        (Link.from_cylinder, "cyl2", {"radius": 1, "height": 2}, [0, 0, 3]),
        False,
        1.0,
    ),
    # --- Cone vs Cone ---
    # Colliding case (apex and base overlap)
    (
        (Link.from_cone, "cone1", {"radius": 1, "height": 2}, [0, 0, 0]),
        (Link.from_cone, "cone2", {"radius": 1, "height": 2}, [0, 0, 1.5]),
        True,
        -0.5,
    ),
    # Non-colliding case (apex and base are 1.0 unit apart)
    (
        (Link.from_cone, "cone1", {"radius": 1, "height": 2}, [0, 0, 0]),
        (Link.from_cone, "cone2", {"radius": 1, "height": 2}, [0, 0, 3]),
        False,
        1.0,
    ),
    # --- Tests for different shape pairs ---
    # Sphere vs Box (non-colliding, distance 1.0)
    (
        (Link.from_sphere, "sphere1", {"radius": 1}, [0, 0, 0]),
        (Link.from_box, "box1", {"extents": np.array([2, 2, 2])}, [3, 0, 0]),
        False,
        1.0,
    ),
    # Sphere vs Box (colliding)
    (
        (Link.from_sphere, "sphere1", {"radius": 1}, [0, 0, 0]),
        (Link.from_box, "box1", {"extents": np.array([2, 2, 2])}, [1.5, 0, 0]),
        True,
        -0.5,
    ),
    # Sphere vs Capsule (non-colliding, distance 1.0)
    (
        (Link.from_sphere, "sphere1", {"radius": 1}, [0, 0, 0]),
        (Link.from_capsule, "cap1", {"radius": 1, "height": 2}, [0, 0, 4]),
        False,
        1.0,
    ),
    # Box vs Cylinder (non-colliding, distance 1.0)
    (
        (Link.from_box, "box1", {"extents": np.array([2, 2, 2])}, [0, 0, 0]),
        (Link.from_cylinder, "cyl1", {"radius": 1, "height": 2}, [3, 0, 0]),
        False,
        1.0,
    ),
    # Capsule vs Cone (colliding)
    (
        (Link.from_capsule, "cap1", {"radius": 1, "height": 2}, [0, 0, 0]),
        (Link.from_cone, "cone1", {"radius": 1, "height": 2}, [0, 0, 1]),
        True,
        -1.0,
    ),
]


def _create_link_from_params(params) -> Link:
    """Helper function to create a Link object from test parameters."""
    factory, name, kwargs, xyz = params
    # Sphere uses the 'center' argument for position, while others use 'default_transform'.
    if factory == Link.from_sphere:
        kwargs["center"] = np.array(xyz)
        return factory(name=name, **kwargs)
    else:
        transform = Transform(translate=np.array(xyz))
        return factory(name=name, default_transform=transform, **kwargs)


@pytest.mark.parametrize(
    "link1_params, link2_params, is_colliding, expected_distance",
    SHAPE_INTERACTION_TEST_CASES,
)
def test_various_shape_interactions(
    link1_params, link2_params, is_colliding, expected_distance
):
    """
    Tests collision detection and distance calculation for various shape pairs,
    including Box, Capsule, Cylinder, and Cone.
    """
    # 1. Set up the robot for each test case
    robot = Robot()
    link1 = _create_link_from_params(link1_params)
    link2 = _create_link_from_params(link2_params)
    robot.add_link(link1)
    robot.add_link(link2)

    mm = MoveManager(robot)
    cm = CollisionManager(mm)

    # 2. Test collision detection (collide method)
    collide_result = cm.collide({link1.name}, {link2.name})
    assert collide_result.is_collision == is_colliding, (
        f"Collision check failed for {link1.name} vs {link2.name}"
    )

    # 3. Test distance calculation (distance method)
    distance_result = cm.distance({link1.name}, {link2.name})
    if is_colliding:
        # If colliding, the distance should be less than or equal to 0.
        assert distance_result.min_distance <= 0, (
            f"Expected non-positive distance for colliding pair {link1.name} vs {link2.name}"
        )
    else:
        # If not colliding, verify that it matches the expected minimum distance.
        assert distance_result.min_distance == pytest.approx(expected_distance), (
            f"Distance check failed for {link1.name} vs {link2.name}"
        )


def _create_mesh_link_from_params(params) -> Link:
    """Helper function to create a Link object from test parameters."""
    factory, name, kwargs, xyz = params
    # Sphere uses the 'center' argument for position, while others use 'default_transform'.
    if factory == Link.from_sphere:
        kwargs["center"] = np.array(xyz)
        link = factory(name=name, **kwargs)
    else:
        transform = Transform(translate=np.array(xyz))
        link = factory(name=name, default_transform=transform, **kwargs)
    link: Link
    return Link.from_mesh(name=name, mesh=link.visual_mesh())


@pytest.mark.parametrize(
    "link1_params, link2_params, is_colliding, expected_distance",
    SHAPE_INTERACTION_TEST_CASES,
)
def test_various_mesh_interactions(
    link1_params, link2_params, is_colliding, expected_distance
):
    """
    Tests collision detection and distance calculation for various shape pairs,
    including Box, Capsule, Cylinder, and Cone.
    """
    # 1. Set up the robot for each test case
    robot = Robot()
    link1 = _create_mesh_link_from_params(link1_params)
    link2 = _create_mesh_link_from_params(link2_params)
    robot.add_link(link1)
    robot.add_link(link2)

    mm = MoveManager(robot)
    cm = CollisionManager(mm)

    # 2. Test collision detection (collide method)
    collide_result = cm.collide({link1.name}, {link2.name})
    assert collide_result.is_collision == is_colliding, (
        f"Collision check failed for {link1.name} vs {link2.name}"
    )

    # 3. Test distance calculation (distance method)
    distance_result = cm.distance({link1.name}, {link2.name})
    if is_colliding:
        # If colliding, the distance should be less than or equal to 0.
        assert distance_result.min_distance <= 0, (
            f"Expected non-positive distance for colliding pair {link1.name} vs {link2.name}"
        )
    else:
        # If not colliding, verify that it matches the expected minimum distance.
        assert distance_result.min_distance == pytest.approx(expected_distance), (
            f"Distance check failed for {link1.name} vs {link2.name}"
        )
