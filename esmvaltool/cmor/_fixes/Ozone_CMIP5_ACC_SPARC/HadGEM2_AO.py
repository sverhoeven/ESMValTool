# pylint: disable=invalid-name, no-self-use, too-few-public-methods
"""Fixes for the project Ozone_CMIP5_ACC_SPARC."""
from . import fix_time_coordinate
from ..fix import Fix


class tro3(Fix):
    """Fixes for tro3."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = self.get_cube_from_list(cubes, short_name='O3')
        cube.var_name = 'tro3'
        return [fix_time_coordinate(cube)]
