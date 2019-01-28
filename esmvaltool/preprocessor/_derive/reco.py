"""Derivation of variable `reco`."""

import logging

import cf_units
import iris
from iris import Constraint

from ._derived_variable_base import DerivedVariableBase

logger = logging.getLogger(__name__)


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `reco`."""

    # Required variables
    _required_variables = {
        'vars': [{
            'short_name': 'ra',
            'field': 'T2{frequency}s'
        }, {
            'short_name': 'rh',
            'field': 'T2{frequency}s'
        }],
        'fx_files': ['sftlf']
    }

    def calculate(self, cubes):
        """Compute ecosystem respiration relative to grid cell area.

        Note
        ----
        By default, `reco` is defined relative to land area. For easy spatial
        integration, the original quantity is multiplied by the land area
        fraction (`sftlf`), so that the resuting derived variable is defined
        relative to the grid cell area. This correction is only relevant for
        coastal regions.

        """
        # Calculate reco
        ra_cube = cubes.extract_strict(
            Constraint(name='plant_respiration_carbon_flux'))
        rh_cube = cubes.extract_strict(
            Constraint(name='heterotrophic_respiration_carbon_flux'))
        reco_cube = ra_cube + rh_cube
        reco_cube.units = cf_units.Unit('kg m^-2 s^-1')

        try:
            sftlf_cube = cubes.extract_strict(
                Constraint(name='land_area_fraction'))
            reco_cube.data *= sftlf_cube.data / 100.0
        except iris.exceptions.ConstraintMismatchError:
            pass
        return reco_cube
