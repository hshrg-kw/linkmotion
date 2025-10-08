"""Range manager for handling multiple range readers."""

import logging
from pathlib import Path
from typing import Iterator

import numpy as np

from .range_read import RangeReader

logger = logging.getLogger(__name__)


class RangeManager:
    """Manager class for handling multiple range readers.

    This class provides a centralized interface for managing multiple RangeReader
    instances, allowing selection of appropriate range readers based on point validity
    and performing range operations across multiple range configurations.

    Note:
        When multiple ranges are registered, first_valid_range() returns ranges in
        insertion order (the order they were added via add_range()).
    """

    def __init__(self) -> None:
        """Initialize an empty RangeManager."""
        self.ranges: dict[str, RangeReader] = {}

    def __repr__(self) -> str:
        """Return string representation of RangeManager.

        Returns:
            String representation showing registered range names.
        """
        range_names = list(self.ranges.keys())
        return f"RangeManager(ranges={range_names})"

    def __len__(self) -> int:
        """Return the number of registered ranges.

        Returns:
            Number of ranges in the manager.
        """
        return len(self.ranges)

    def __contains__(self, name: str) -> bool:
        """Check if a range name exists in the manager.

        Args:
            name: Name of the range to check.

        Returns:
            True if the range exists, False otherwise.
        """
        return name in self.ranges

    def __iter__(self) -> Iterator[str]:
        """Iterate over range names.

        Returns:
            Iterator of range names.
        """
        return iter(self.ranges)

    def add_range(
        self, name: str, range_reader: Path | RangeReader, overwrite: bool = False
    ) -> None:
        """Add a range reader to the manager.

        Args:
            name: Unique name for the range reader.
            range_reader: Either a Path to a range file or a RangeReader instance.
            overwrite: If True, allow overwriting existing ranges. If False, raise
                ValueError when attempting to add a range with an existing name.

        Raises:
            TypeError: If range_reader is not a Path or RangeReader instance.
            ValueError: If name already exists and overwrite is False.
        """
        is_overwriting = name in self.ranges

        if is_overwriting and not overwrite:
            raise ValueError(
                f"Range '{name}' already exists. Use overwrite=True to replace it."
            )

        if isinstance(range_reader, Path):
            self.ranges[name] = RangeReader.import_from_file(range_reader)
        elif isinstance(range_reader, RangeReader):
            self.ranges[name] = range_reader
        else:
            raise TypeError("range_reader must be a Path or RangeReader instance")

        if is_overwriting:
            logger.info(f"Overwrote existing range '{name}'")
        else:
            logger.debug(f"Added range '{name}' to manager")

    def remove_range(self, name: str) -> None:
        """Remove a range reader from the manager.

        Args:
            name: Name of the range reader to remove.

        Raises:
            KeyError: If the range name is not found.
        """
        if name not in self.ranges:
            raise KeyError(
                f"Range '{name}' not found in RangeManager. "
                f"Available ranges: {list(self.ranges.keys())}"
            )
        del self.ranges[name]
        logger.debug(f"Removed range '{name}' from manager")

    def clear(self) -> None:
        """Remove all range readers from the manager."""
        self.ranges.clear()
        logger.debug("Cleared all ranges from manager")

    def list_ranges(self) -> list[str]:
        """Get a list of all registered range names.

        Returns:
            List of range names in insertion order.
        """
        return list(self.ranges.keys())

    def get_range(self, name: str) -> RangeReader:
        """Get a specific range reader by name.

        Args:
            name: Name of the range reader to retrieve.

        Returns:
            The requested RangeReader instance.

        Raises:
            KeyError: If the range name is not found.
        """
        if name not in self.ranges:
            raise KeyError(
                f"Range '{name}' not found in RangeManager. "
                f"Available ranges: {list(self.ranges.keys())}"
            )
        return self.ranges[name]

    def _first_calculable_range(self, *points: float) -> RangeReader:
        """Find the first range reader where the given points are calculable.

        Ranges are checked in insertion order (the order they were added).

        Args:
            *points: Coordinates to check for calculability.

        Returns:
            The first RangeReader instance where points are calculable.

        Raises:
            ValueError: If no ranges are registered or no calculable range is found.
        """
        if not self.ranges:
            raise ValueError(
                "No ranges registered in RangeManager. Add ranges using add_range()."
            )

        for range_name, range_reader in self.ranges.items():
            if range_reader.is_calculable(*points):
                logger.debug(f"Point {points} is calculable in '{range_name}'")
                return range_reader

        raise ValueError(
            f"No calculable range found for points {points}. "
            f"Checked ranges: {list(self.ranges.keys())}"
        )

    def calculable_ranges(self, *points: float) -> list[RangeReader]:
        """Get all range readers where the given points are calculable.

        Args:
            *points: Coordinates to check for calculability.

        Returns:
            List of RangeReader instances where points are calculable.
        """
        return [
            range_reader
            for range_reader in self.ranges.values()
            if range_reader.is_calculable(*points)
        ]

    def is_out(self, *points: float) -> bool:
        """Check if points are outside all valid ranges.

        A point is considered "out" if it is out of bounds in all calculable ranges.

        Args:
            *points: Coordinates to check.

        Returns:
            True if points are outside all calculable ranges, False otherwise.
        """
        calculable = self.calculable_ranges(*points)
        if not calculable:
            return True
        return all(range_.is_out(*points) for range_ in calculable)

    def is_valid(self, *points: float) -> bool:
        """Check if points are valid in any calculable range.

        The definition of "valid" depends on the RangeReader implementation.
        A point is considered valid if it is valid in at least one calculable range.

        Args:
            *points: Coordinates to check.

        Returns:
            True if points are valid in at least one calculable range, False otherwise.
        """
        calculable = self.calculable_ranges(*points)
        if not calculable:
            return False
        return any(range_.is_valid(*points) for range_ in calculable)

    def get_corner_values(
        self, *points: float
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Get corner values for the given points.

        Args:
            *points: Coordinates to query.

        Returns:
            Array of corner values.

        Raises:
            ValueError: If no valid range is found for the given points.
        """
        range_reader = self._first_calculable_range(*points)
        return range_reader.get_corner_values(*points)

    def get_corner_max_value(self, *points: float) -> float:
        """Get maximum corner value for the given points.

        Args:
            *points: Coordinates to query.

        Returns:
            Maximum corner value.

        Raises:
            ValueError: If no valid range is found for the given points.
        """
        range_reader = self._first_calculable_range(*points)
        return range_reader.get_corner_max_value(*points)

    def get_corner_min_value(self, *points: float) -> float:
        """Get minimum corner value for the given points.

        Args:
            *points: Coordinates to query.

        Returns:
            Minimum corner value.

        Raises:
            ValueError: If no valid range is found for the given points.
        """
        range_reader = self._first_calculable_range(*points)
        return range_reader.get_corner_min_value(*points)

    def get_nearest_value(self, *points: float) -> float:
        """Get nearest value for the given points.

        Args:
            *points: Coordinates to query.

        Returns:
            Nearest value.

        Raises:
            ValueError: If no valid range is found for the given points.
        """
        range_reader = self._first_calculable_range(*points)
        return range_reader.get_nearest_value(*points)

    def interpolate(self, *points: float) -> float:
        """Interpolate value at the given points.

        Args:
            *points: Coordinates to interpolate at.

        Returns:
            Interpolated value.

        Raises:
            ValueError: If no valid range is found for the given points.
        """
        range_reader = self._first_calculable_range(*points)
        return range_reader.interpolate(*points)

    def interpolate_max(self, *points: float) -> float:
        """Get maximum interpolated value at the given points across all calculable ranges.

        Args:
            *points: Coordinates to interpolate at.

        Returns:
            Maximum interpolated value.

        Raises:
            ValueError: If no calculable range is found for the given points.
        """
        calculable = self.calculable_ranges(*points)
        if not calculable:
            raise ValueError(
                f"No calculable range found for points {points}. "
                f"Available ranges: {list(self.ranges.keys())}"
            )
        return max(range_.interpolate(*points) for range_ in calculable)

    def interpolate_min(self, *points: float) -> float:
        """Get minimum interpolated value at the given points across all calculable ranges.

        Args:
            *points: Coordinates to interpolate at.

        Returns:
            Minimum interpolated value.

        Raises:
            ValueError: If no calculable range is found for the given points.
        """
        calculable = self.calculable_ranges(*points)
        if not calculable:
            raise ValueError(
                f"No calculable range found for points {points}. "
                f"Available ranges: {list(self.ranges.keys())}"
            )
        return min(range_.interpolate(*points) for range_ in calculable)
