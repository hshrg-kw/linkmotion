import pytest
import numpy as np

from linkmotion.transform import Transform
from linkmotion.transform.manager import TransformManager


# --- Pytest Fixtures ---
@pytest.fixture
def manager() -> TransformManager:
    """Provides a clean TransformManager instance for each test function."""
    return TransformManager()


@pytest.fixture
def populated_manager(manager: TransformManager) -> TransformManager:
    """
    Provides a manager with a pre-built node hierarchy for testing.
    Hierarchy: 0 -> 1 -> 2
    Positions:
    - Node 0 (root): [10, 0, 0]
    - Node 1 (child): [5, 0, 0] (local)
    - Node 2 (grandchild): [2, 0, 0] (local)
    """
    manager.add_node(0, default_transform=Transform(translate=np.array([10, 0, 0])))
    manager.add_node(
        1, parent_id=0, default_transform=Transform(translate=np.array([5, 0, 0]))
    )
    manager.add_node(
        2, parent_id=1, default_transform=Transform(translate=np.array([2, 0, 0]))
    )
    return manager


# --- Test Cases ---


def test_add_node(manager: TransformManager):
    """Tests basic node addition."""
    t_root = Transform(translate=np.array([100, 50, 0]))
    manager.add_node(0, default_transform=t_root)
    assert 0 in manager.local_transforms
    assert manager.local_transforms[0] == t_root
    assert 0 in manager.dirty_nodes

    manager.add_node(1, parent_id=0)
    assert manager.parent_map[1] == 0
    assert 1 in manager.children_map[0]


def test_add_node_errors(manager: TransformManager):
    """Tests error conditions when adding nodes."""
    manager.add_node(0)
    with pytest.raises(ValueError, match="already exists"):
        manager.add_node(0)  # Duplicate ID
    with pytest.raises(ValueError, match="does not exist"):
        manager.add_node(1, parent_id=99)  # Non-existent parent


def test_get_world_transform_calculation(populated_manager: TransformManager):
    """Tests the correctness of world transform calculation with translations."""
    manager = populated_manager
    # Root (0): local is [10, 0, 0] -> world is [10, 0, 0]
    expected_t0 = Transform(translate=np.array([10, 0, 0]))
    assert manager.get_world_transform(0) == expected_t0

    # Child (1): parent_world([10,0,0]) + local([5,0,0]) -> world is [15, 0, 0]
    expected_t1 = Transform(translate=np.array([15, 0, 0]))
    assert manager.get_world_transform(1) == expected_t1

    # Grandchild (2): parent_world([15,0,0]) + local([2,0,0]) -> world is [17, 0, 0]
    expected_t2 = Transform(translate=np.array([17, 0, 0]))
    assert manager.get_world_transform(2) == expected_t2


def test_transform_caching(populated_manager: TransformManager):
    """Tests that world transforms are cached and nodes become 'clean'."""
    manager = populated_manager
    # Calculate all transforms to make them 'clean'
    manager.get_world_transform(0)
    manager.get_world_transform(1)
    manager.get_world_transform(2)
    assert not manager.dirty_nodes

    # Manually change a local transform without using the proper method
    # to verify that the cached value is returned.
    manager.local_transforms[0] = Transform(translate=np.array([999, 999, 999]))

    # The returned value should still be the old, cached one.
    expected_t0 = Transform(translate=np.array([10, 0, 0]))
    assert manager.get_world_transform(0) == expected_t0


def test_set_local_transform_dirties_descendants(populated_manager: TransformManager):
    """Tests that changing a transform dirties the node and all its descendants."""
    manager = populated_manager
    # First, clean all nodes by getting their transforms
    manager.get_world_transform(2)
    assert not manager.dirty_nodes

    # Change the transform of the middle node (1)
    new_local_t1 = Transform(translate=np.array([50, 0, 0]))
    manager.set_local_transform(1, new_local_t1)

    # Node 1 and its descendant 2 should be dirty, but not its ancestor 0.
    assert 1 in manager.dirty_nodes
    assert 2 in manager.dirty_nodes
    assert 0 not in manager.dirty_nodes

    # Check if the new world transforms are calculated correctly
    # Root(0) is still at [10, 0, 0]
    assert manager.get_world_transform(0).position == pytest.approx([10, 0, 0])
    # Child(1) world = [10,0,0] + new_local([50,0,0]) = [60,0,0]
    assert manager.get_world_transform(1).position == pytest.approx([60, 0, 0])
    # Grandchild(2) world = [60,0,0] + local([2,0,0]) = [62,0,0]
    assert manager.get_world_transform(2).position == pytest.approx([62, 0, 0])


def test_reset_node_transform(populated_manager: TransformManager):
    """Tests resetting a single node's transform to its default."""
    manager = populated_manager
    # Change the local transform first
    manager.set_local_transform(1, Transform(translate=np.array([99, 99, 99])))
    assert manager.local_transforms[1].position == pytest.approx([99, 99, 99])

    manager.reset_node_transform(1)

    # It should be back to the default value of [5, 0, 0]
    expected_default_t1 = Transform(translate=np.array([5, 0, 0]))
    assert manager.local_transforms[1] == expected_default_t1
    assert 1 in manager.dirty_nodes


def test_reset_all_transforms(populated_manager: TransformManager):
    """Tests resetting all nodes to their default transforms."""
    manager = populated_manager
    manager.set_local_transform(0, Transform(translate=np.array([111, 0, 0])))
    manager.set_local_transform(1, Transform(translate=np.array([222, 0, 0])))

    manager.reset_all_transforms()

    assert manager.local_transforms[0].position == pytest.approx([10, 0, 0])
    assert manager.local_transforms[1].position == pytest.approx([5, 0, 0])
    assert manager.dirty_nodes == {0, 1, 2}


def test_apply_relative_transform(populated_manager: TransformManager):
    """Tests applying a transform relative to the node's default transform."""
    manager = populated_manager  # Default for node 1 is position [5, 0, 0]

    relative_transform = Transform(translate=np.array([3, 1, 0]))
    manager.apply_relative_transform(1, relative_transform)

    # New local should be default([5,0,0]) + relative([3,1,0]) = [8, 1, 0]
    assert manager.local_transforms[1].position == pytest.approx([8, 1, 0])
    assert 1 in manager.dirty_nodes

    # World transform should be updated accordingly
    # parent_world([10,0,0]) + new_local([8,1,0]) = [18, 1, 0]
    assert manager.get_world_transform(1).position == pytest.approx([18, 1, 0])


# Error handling tests for manager.py
def test_set_local_transform_nonexistent_node():
    manager = TransformManager()
    with pytest.raises(ValueError, match="Node ID 999 does not exist"):
        manager.set_local_transform(999, Transform())


# Tests for circular dependency detection
def test_circular_dependency_prevention():
    """Tests that circular dependencies are prevented when adding nodes."""
    manager = TransformManager()

    # Create a simple chain: 0 -> 1 -> 2
    manager.add_node(0)
    manager.add_node(1, parent_id=0)
    manager.add_node(2, parent_id=1)

    # Try to make 0 a child of 2 (would create cycle) - but 0 already exists
    with pytest.raises(ValueError, match="already exists"):
        manager.add_node(0, parent_id=2)

    # Test the _would_create_cycle method directly
    assert manager._would_create_cycle(0, 2)  # 0 is ancestor of 2
    assert manager._would_create_cycle(1, 2)  # 1 is ancestor of 2
    assert not manager._would_create_cycle(
        2, 0
    )  # This would be valid (child to ancestor)
    assert not manager._would_create_cycle(3, 0)  # Non-existent node


def test_would_create_cycle_with_nonexistent_node():
    """Tests cycle detection when node doesn't exist yet."""
    manager = TransformManager()
    manager.add_node(0)

    # Should not create cycle since node 1 doesn't exist
    assert not manager._would_create_cycle(1, 0)


def test_set_local_transform_no_change_optimization():
    """Tests that setting the same transform doesn't dirty descendants unnecessarily."""
    manager = TransformManager()
    transform = Transform(translate=np.array([1, 2, 3]))

    manager.add_node(0, default_transform=transform)
    manager.add_node(1, parent_id=0)

    # Clear dirty nodes
    manager.get_world_transform(0)
    manager.get_world_transform(1)
    assert not manager.dirty_nodes

    # Setting the same transform should not dirty nodes
    manager.set_local_transform(0, transform)
    assert not manager.dirty_nodes


# Tests for new TransformManager methods
def test_remove_node_single():
    """Tests removing a single node without children."""
    manager = TransformManager()
    manager.add_node(0)
    manager.add_node(1, parent_id=0)

    manager.remove_node(1)

    assert not manager._node_exists(1)
    assert 1 not in manager.children_map[0]
    assert len(manager.get_node_ids()) == 1


def test_remove_node_with_descendants():
    """Tests removing a node with multiple descendants."""
    manager = TransformManager()
    # Create hierarchy: 0 -> 1 -> 2, 0 -> 3
    manager.add_node(0)
    manager.add_node(1, parent_id=0)
    manager.add_node(2, parent_id=1)
    manager.add_node(3, parent_id=0)

    # Remove node 1 (should also remove node 2)
    manager.remove_node(1)

    assert not manager._node_exists(1)
    assert not manager._node_exists(2)
    assert manager._node_exists(0)
    assert manager._node_exists(3)
    assert 1 not in manager.children_map[0]
    assert 3 in manager.children_map[0]


def test_remove_node_nonexistent():
    """Tests error when removing non-existent node."""
    manager = TransformManager()
    with pytest.raises(ValueError, match="Node ID 999 does not exist"):
        manager.remove_node(999)


def test_get_node_ids():
    """Tests getting all node IDs."""
    manager = TransformManager()
    assert manager.get_node_ids() == []

    manager.add_node(5)
    manager.add_node(10)
    manager.add_node(1)

    node_ids = manager.get_node_ids()
    assert set(node_ids) == {1, 5, 10}
    assert len(node_ids) == 3


def test_get_root_nodes():
    """Tests getting root node IDs."""
    manager = TransformManager()
    assert manager.get_root_nodes() == []

    # Create hierarchy: 0, 1 -> 2, 3
    manager.add_node(0)
    manager.add_node(1)
    manager.add_node(2, parent_id=1)
    manager.add_node(3)

    root_nodes = manager.get_root_nodes()
    assert set(root_nodes) == {0, 1, 3}


def test_get_children():
    """Tests getting children of a node."""
    manager = TransformManager()
    manager.add_node(0)
    manager.add_node(1, parent_id=0)
    manager.add_node(2, parent_id=0)
    manager.add_node(3, parent_id=1)

    assert set(manager.get_children(0)) == {1, 2}
    assert manager.get_children(1) == [3]
    assert manager.get_children(2) == []
    assert manager.get_children(3) == []


def test_get_children_nonexistent_node():
    """Tests error when getting children of non-existent node."""
    manager = TransformManager()
    with pytest.raises(ValueError, match="Node ID 999 does not exist"):
        manager.get_children(999)


def test_reset_node_transform_nonexistent():
    """Tests error when resetting non-existent node."""
    manager = TransformManager()
    with pytest.raises(ValueError, match="Node ID 999 does not exist"):
        manager.reset_node_transform(999)


def test_apply_relative_transform_nonexistent():
    """Tests error when applying relative transform to non-existent node."""
    manager = TransformManager()
    with pytest.raises(ValueError, match="Node ID 999 does not exist"):
        manager.apply_relative_transform(999, Transform())


def test_get_world_transform_nonexistent():
    """Tests error when getting world transform of non-existent node."""
    manager = TransformManager()
    with pytest.raises(ValueError, match="Node ID 999 does not exist"):
        manager.get_world_transform(999)
