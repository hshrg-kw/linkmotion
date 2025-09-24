# Transform System Examples

This directory contains comprehensive examples demonstrating how to use the LinkMotion transform system for robotics applications. Each example is self-contained and can be run independently.

## Examples Overview

### 1. Basic Transform Usage (`basic_transform_usage.py`)
**Purpose**: Learn the fundamentals of the Transform class
**Topics Covered**:
- Creating transforms (identity, translation, rotation, combined)
- Transform operations (copy, in-place modifications)
- Applying transforms to vectors, meshes, and other transforms
- Transform composition and matrix conversions
- Basic robotics pose calculations

**Run with**: `uv run python examples/basic_transform_usage.py`

### 2. TransformManager Hierarchy (`transform_manager_hierarchy.py`)
**Purpose**: Master hierarchical transform management
**Topics Covered**:
- Creating kinematic chains and scene graphs
- Parent-child relationships
- World coordinate calculations
- Joint movements and cascading updates
- Node management (add, remove, query)
- Reset operations and relative transforms

**Run with**: `uv run python examples/transform_manager_hierarchy.py`

## Example Structure

Each example follows a consistent structure:
- **Imports and setup**: Required libraries and logging configuration
- **Main functions**: Focused demonstrations of specific concepts
- **Practical scenarios**: Real-world applications and use cases
- **Comprehensive logging**: Detailed output explaining what's happening
- **Error handling**: Robust examples that handle edge cases

## Running Examples

### Individual Examples
```bash
# Run a specific example
uv run python examples/basic_transform_usage.py

# Run with different log levels
PYTHONPATH=. python examples/robotics_applications.py
```

### All Examples
```bash
# Run all examples in sequence
for example in examples/*.py; do
    echo "Running $example..."
    uv run python "$example"
    echo "Completed $example"
    echo "---"
done
```

## Key Concepts Demonstrated

### Transform Fundamentals
- **Identity transforms**: No rotation or translation
- **Composition**: Combining multiple transforms
- **Inverse transforms**: Reversing transformations
- **Matrix representation**: 4x4 homogeneous matrices

### Hierarchical Systems
- **Parent-child relationships**: How transforms propagate
- **World coordinates**: Converting from local to global frames
- **Kinematic chains**: Serial robot arm structures
- **Scene graphs**: Complex multi-object hierarchies
