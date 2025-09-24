#!/bin/bash

# run_all_examples.sh
# Script to run all example files in the examples directory

set -e  # Exit on any error

EXAMPLES_DIR="examples"

if [ ! -d "$EXAMPLES_DIR" ]; then
    echo "L Examples directory not found!"
    exit 1
fi

echo "Running all examples..."

# Find all Python files in examples directory and subdirectories
for example in $(find "$EXAMPLES_DIR" -name "*.py" -type f | sort); do
    echo "Running: $example"
    if uv run python "$example"; then
        echo "$example completed successfully"
    else
        echo "$example failed"
        exit 1
    fi
    echo ""
done

echo "All examples completed successfully!"