# pylint: disable=invalid-name, no-self-use, too-few-public-methods
"""Fixes for the project Ozone_CMIP5_ACC_SPARC."""
from . import fix_time_coordinate, remove_cell_method
from ..fix import Fix


class tro3(Fix):
    """Fixes for tro3."""

    def fix_file(self, filepath, output_dir):
        """Fix file."""
        new_path = Fix.get_fixed_filepath(output_dir, filepath)
        return remove_cell_method(filepath, new_path)

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = self.get_cube_from_list(cubes, short_name='O3')
        return [fix_time_coordinate(cube)]
