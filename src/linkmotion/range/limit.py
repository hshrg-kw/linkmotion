import logging


from linkmotion.collision.manager import CollisionManager
from linkmotion.move.manager import JointLimitError
from linkmotion.robot.joint import JointType

logger = logging.getLogger(__name__)


def calc_limit_from_current_state(
    cm: CollisionManager,
    joint_name: str,
    link_names1: set[str],
    link_names2: set[str],
    collision_tolerance: float,
    step: float = 0.1,
    max_try: int = 1000,
) -> float:
    """Check how far the joint can translate from current state without colliding.

    Args:
        manager (CollisionManager): The collision manager with the current robot state.
        joint_name (str): The name of the joint to check.
        link_names1 (set[str]): The set of link names attached to the joint to be checked.
        link_names2 (set[str]): The set of link names to check against for collisions.
        collision_tolerance (float): The minimum allowed distance between links before considering it a collision.
        step (float, optional): The step size for checking. Defaults to 0.1. Negative value means checking in the negative direction. This arg means the accuracy of return value.
        max_try (int, optional): The maximum number of iterations to prevent infinite loops. Defaults to 1000.

    Returns:
        float: The maximum distance the joint can translate without colliding.

    Raises:
        RuntimeError: If the maximum number of tries is exceeded.
    """
    joint_type = cm.mm.robot.joint(joint_name).type
    if joint_type not in [
        JointType.PRISMATIC,
        JointType.REVOLUTE,
        JointType.CONTINUOUS,
    ]:
        raise ValueError(f"Joint {joint_name} is not prismatic or revolute.")
    if step == 0:
        raise ValueError("Step size cannot be zero.")
    # copy MoveManager to avoid modifying the original state
    joint_value_map = cm.mm.joint_values_map.copy()

    # store initial joint value
    previous_value = cm.mm.joint_value(joint_name)
    logger.debug(f"Starting joint {joint_name} at value {previous_value}")

    is_collide = False
    try_count = 1

    while not is_collide:
        # expensive distance calculation
        safe_distance = cm.distance(link_names1, link_names2).min_distance

        # consider collision tolerance
        safe_distance = safe_distance - collision_tolerance
        # collision detection
        if safe_distance < 0.0:
            is_collide = True
            safe_distance = 0.0
            return previous_value
        # determine next step
        next_step = step
        # if joint type is prismatic
        # link can move with safe distance exceeding step size
        if joint_type == JointType.PRISMATIC:
            if step > 0:
                next_step = max(step, safe_distance * 0.9999999)
            elif step < 0:
                next_step = min(step, -safe_distance * 0.9999999)
        # store previous value
        previous_value = cm.mm.joint_value(joint_name)
        # update joint state
        try:
            cm.mm.move(joint_name, previous_value + next_step)
            logger.debug(
                f"Joint {joint_name} moved to {previous_value + next_step} (step: {next_step})"
            )
        # stop if joint limit is reached
        except JointLimitError as e:
            logger.info(f"Reached joint limit for {joint_name} at value {e.value}")
            return (
                cm.mm.robot.joint(joint_name).max
                if step > 0
                else cm.mm.robot.joint(joint_name).min
            )

        # avoid infinite loop
        if try_count > max_try:
            raise RuntimeError("Exceeded maximum number of tries.")
        try_count += 1

    for joint_name, joint_value in joint_value_map.items():
        cm.mm.move(joint_name, joint_value)

    raise RuntimeError("Unreachable code reached while checking joint limits.")
