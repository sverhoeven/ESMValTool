"""Calculate trends to use them for GBRT models."""

import logging
import os

import iris
from scipy import stats

from esmvaltool.diag_scripts.shared import run_diagnostic, gbrt

logger = logging.getLogger(os.path.basename(__file__))


def main(cfg):
    """Run the diagnostic."""
    if cfg['write_netcdf']:
        for (path, data) in cfg['input_data'].items():
            cube = iris.load_cube(path)

            # Calculate trends
            if cfg.get('yearly_trend'):
                cube = cube.aggregated_by('year', iris.analysis.MEAN)
                temp_units = ' yr-1'
            else:
                temp_units = ' mon-1'
            reg = stats.linregress(cube.coord('year').points, cube.data)

            # Save new cube
            cube = cube.collapsed('time', iris.analysis.MEAN)
            cube.data = reg.slope
            new_path = os.path.join(cfg['work_dir'], os.path.basename(path))
            data['filename'] = new_path
            data['units'] += temp_units
            data['short_name'] += '_trend'
            data['long_name'] += ' (trend)'
            gbrt.write_cube(cube, data, new_path, cfg)
    else:
        logger.warning("Cannot save netcdf files because 'write_netcdf' is "
                       "set to 'False' in user configuration file.")


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
