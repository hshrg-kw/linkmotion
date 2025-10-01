# CLAUDE.md
This file provides guidance for Claude Code (claude.ai/code) when working with the code in this repository. Its purpose is to ensure consistency, maintainability, and high-quality code.

## 1. Project Documentation

**LinkMotion** is a comprehensive robotics library that provides tools for robot modeling, joint control, collision detection, visualization, and URDF import/export functionality.

**Note**: The `README.md` file currently contains minimal content and should be expanded to provide a proper project overview.

### Supplementary Documentation

Detailed explanations of specific components are documented in the following locations:

-   `examples/robot/README.md` - Robot system usage examples and patterns
-   `examples/transform/README.md` - Transform system usage examples
-   `examples/move/README.md` - Robot movement and joint control examples

## 2. Development Commands

Use the following `uv` commands to manage common development tasks.

-   **Format Code**: `uv run ruff format`
    -   Formats all Python code according to the project's style configuration.
-   **Lint Code**: `uv run ruff check`
    -   Analyzes the code for potential errors, bugs, and stylistic issues.
-   **Auto-fix Lint Issues**: `uv run ruff check --fix`
    -   Automatically fixes all fixable linting issues.
-   **Run All Tests**: `uv run pytest`
    -   Executes the entire test suite using pytest.
-   **Run a Specific Test**: `uv run pytest tests/test_filename.py`
    -   Executes a single test file.
-   **Run Tests Verbose**: `uv run pytest -v`
    -   Runs tests with detailed, verbose output for debugging.
-   **Run Tests with Coverage**: `uv run pytest --cov`
    -   Runs tests with coverage reporting.

## 3. Project Architecture

This is a comprehensive robotics library built in Python. Below is an overview of the key functional components and modules.

### Project Directory Structure

The project follows a standard Python project layout with modular organization:

-   `src/linkmotion/` - Main source code organized by functional modules
-   `tests/` - Comprehensive test suite mirroring the `src/` structure
-   `docs/` - Auto-generated Sphinx documentation (do not edit manually)
-   `examples/` - Standalone example scripts with detailed README files
-   `notebooks/` - Jupyter notebooks for visualization and interactive experimentation
-   `models/` - Sample robot model files (URDF, mesh files)
-   `scripts/` - Development and automation scripts

### Core Modules

The `src/linkmotion/` directory contains the following functional modules:

#### Robot System (`robot/`)
Core robotics modeling and manipulation functionality:
-   `robot.py` - Main Robot class for building and manipulating robot models
-   `link.py` - Link class representing robot components
-   `joint.py` - Joint class and JointType enum for robot connections
-   `custom.py` - Custom robot construction utilities
-   `shape/` - Geometric shapes (box, sphere, cylinder, cone, capsule, mesh, convex)

#### Transform System (`transform/`)
Spatial transformations and coordinate frame management:
-   `transform.py` - Core Transform class for spatial calculations
-   `manager.py` - Transform management and hierarchy handling

#### Robot Movement (`move/`)
Robot joint manipulation and state management:
-   `manager.py` - MoveManager class for joint control and robot state management

#### Collision Detection (`collision/`)
Collision detection and safety checking:
-   `manager.py` - CollisionManager class for collision detection operations

#### Range Analysis (`range/`)
Workspace and reachability analysis:
-   `range.py` - Range calculation and analysis tools
-   `limit.py` - Joint and workspace limit handling

#### Modeling Tools (`modeling/`)
Advanced geometric modeling operations:
-   `sweep.py` - Sweep operations for complex geometry generation
-   `remove.py` - Geometry removal and modification operations

#### URDF Support (`urdf/`)
URDF (Unified Robot Description Format) import/export:
-   `parser.py` - URDF file parsing and import functionality
-   `writer.py` - URDF file generation and export functionality

#### Visualization (`visual/`)
3D visualization and rendering capabilities:
-   `base.py` - Base visualization classes and utilities
-   `robot.py` - Robot visualization and rendering
-   `mesh.py` - Mesh visualization and display
-   `move.py` - Motion visualization and animation
-   `collision.py` - Collision visualization
-   `range.py` - Range and workspace visualization

#### Type System (`typing/`)
Type definitions and utilities:
-   `numpy.py` - NumPy-based type definitions for arrays and matrices

#### Constants (`const/`)
Project-wide constants and configuration:
-   `const.py` - Mathematical constants and default values

### Example Organization

Examples are organized by functional area in `examples/`:
-   `robot/` - Robot construction, kinematics, and advanced operations
-   `transform/` - Transform system usage and hierarchy management
-   `move/` - Robot joint manipulation and state management
-   `range/` - Workspace analysis and reachability calculations
-   `collision/` - Collision detection and safety checking
-   `modeling/` - Advanced geometric modeling operations

### Notebook Organization

Jupyter notebooks are organized in `notebooks/`:
-   `visual/` - Interactive visualization examples and tutorials
-   `urdf/` - URDF import/export demonstrations with visual examples

## 4. Coding Guidelines

Follow these guidelines strictly to maintain code quality and consistency.

### Comments and Documentation

-   **Language**: All code comments, docstrings, and commit messages **must be written in English**.
-   **Clarity**: Write clear and descriptive comments to explain complex logic, business rules, or non-obvious implementations.
-   **Docstrings**: Write **Google-style docstrings** for all public modules, classes, methods, and functions. The docstring should describe the purpose, arguments (`Args:`), and return values (`Returns:`).

### Code Style and Best Practices

-   **Formatting**: Adhere to the `Ruff` formatter configuration. Always run the formatter before committing code.
-   **Idiomatic Naming**: Use clear and idiomatic Python naming conventions (e.g., `snake_case` for variables and functions, `PascalCase` for classes).
-   **Single Responsibility**: Aim for one primary class per file. Decompose complex modules into smaller, more focused files.
-   **Type Hinting**: **Always** use type hints for function arguments and return values. This is critical for static analysis and IDE support.
-   **Error Handling**: Implement robust error handling using `try...except` blocks where exceptions (e.g., `IOError`, `ValueError`) are likely to occur. Avoid catching generic `Exception`. However, the error handling for issues that can be prevented by type hints is not needed because that would be excessive.
-   **Logging**: Use the `logging` module for all output, including debugging, status information, and errors. Do not use `print()` statements in the library/application code.
-   **Custom Class `__repr__`**: All custom-defined classes MUST implement the `__repr__` special method for effective debugging and meaningful logging.

### Python Logging Specifics

-   **Logger Setup**: In each module, get a logger instance with `import logging` and `logger = logging.getLogger(__name__)`.
-   **No Cross-Module Imports**: Do not import logger instances from other modules. Each module should create its own to preserve the correct logger hierarchy.

## 5. Document Maintenance

-   **Stay Synchronized**: If you implement changes that conflict with or extend the information in this document (e.g., adding a new development command or architectural component), you **must** update this file accordingly.
-   **Update README**: If the project's setup, installation, or basic usage changes, update both `README.md` and this file. Documentation must always reflect the current state of the codebase.
