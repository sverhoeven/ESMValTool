#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnostic script to calculte Austral Jet position.

Description
-----------
Allows the calculation of the Austral Jet position in the recipe as diagnostic
script from 'ua' ('eastward_wind') data given via the 'ancestors' key.

Author
------
Manuel Schlund (DLR, Germany)

Project
-------
CRESCENDO

"""

import logging
import os

import iris

from esmvaltool.diag_scripts.shared import run_diagnostic
from esmvaltool.preprocessor._derive.uajet import DerivedVariable as Uajet

logger = logging.getLogger(os.path.basename(__file__))


def main(cfg):
    """Run the diagnostic."""
    input_dirs = [
        d for d in cfg['input_files'] if not d.endswith('metadata.yml')
    ]
    if not input_dirs:
        logging.error("No input files given, use 'ancestors' key in recipe")
        return

    # Iterate over all directories
    for input_dir in input_dirs:
        for (root, _, files) in os.walk(input_dir):
            for filename in files:
                if '.nc' not in filename:
                    continue
                path = os.path.join(root, filename)
                cubes = iris.load(
                    path, constraints=iris.Constraint(name='eastward_wind'))
                if not cubes:
                    continue

                # Calculate jet position via preprocessor function
                uajet = Uajet('ua')
                jet_position_cube = uajet.calculate(cubes)
                description = cubes[0].attributes.get('description', filename)
                units = jet_position_cube.units.name
                logger.info("%s: %.2f %s", description, jet_position_cube.data,
                            units)


if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
