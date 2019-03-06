"""Derivation of variable `gpp_grid`."""

import logging

from ._derived_variable_base import DerivedVariableBase
from ._shared import grid_area_correction

logger = logging.getLogger(__name__)


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `gpp_grid`."""

    # Required variables
    _required_variables = {
        'vars': [{
            'short_name': 'gpp',
            'field': 'T2{frequency}s'
        }],
        'fx_files': ['areacella', 'sftlf']
    }

    def calculate(self, cubes):
        """Compute gross primary production per grid cell.

        Note
        ----
        By default, `gpp` is defined relative to land area. For spatial
        integration, the original quantity is multiplied by the land area
        fraction (`sftlf`), so that the resuting derived variable is defined
        relative to the grid cell area. This correction is only relevant for
        coastal regions.

        """
        return grid_area_correction(cubes,
                                    'gross_primary_productivity_of_carbon')
