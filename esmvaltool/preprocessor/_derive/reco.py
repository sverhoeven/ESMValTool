"""Derivation of variable `reco`."""

import logging

import cf_units
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
    }

    def calculate(self, cubes):
        """Compute ecosystem respiration."""
        ra_cube = cubes.extract_strict(
            Constraint(name='plant_respiration_carbon_flux'))
        rh_cube = cubes.extract_strict(
            Constraint(name='heterotrophic_respiration_carbon_flux'))
        reco_cube = ra_cube + rh_cube
        reco_cube.units = cf_units.Unit('kg m-2 s-1')
        return reco_cube
