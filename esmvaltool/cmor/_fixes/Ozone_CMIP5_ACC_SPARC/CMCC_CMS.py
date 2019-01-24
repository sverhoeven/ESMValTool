"""Fixes for the project Ozone_CMIP5_ACC_SPARC."""
from . import fix_time_coordinate
from ..fix import Fix


class tro3(Fix):
    """Fixes for tro3."""

    def fix_metadata(self, cube):
        """Fix metadata."""
        return fix_time_coordinate(cube)
