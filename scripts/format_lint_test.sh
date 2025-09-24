#!/bin/bash

# format_lint_test.sh
# Script to format code, run linting, and execute tests

set -e  # Exit on any error

echo "=' Formatting code..."
uv run ruff format

echo "=
 Checking code with linter..."
uv run ruff check --fix

echo "Running tests..."
uv run pytest -v

echo "All checks completed successfully!"