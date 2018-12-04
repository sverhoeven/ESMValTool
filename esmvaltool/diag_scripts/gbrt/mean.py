"""Calculate means to use them for GBRT models."""

import logging
import os

import iris

from esmvaltool.diag_scripts.gbrt import write_cube
from esmvaltool.diag_scripts.shared import run_diagnostic

logger = logging.getLogger(os.path.basename(__file__))


def main(cfg):
    """Run the diagnostic."""
    if cfg['write_netcdf']:
        for (path, data) in cfg['input_data'].items():
            cube = iris.load_cube(path)

            # Calculate desired means
            if cfg.get('global_mean'):
                weights = iris.analysis.cartography.area_weights(cube)
                cube = cube.collapsed(['latitude', 'longitude'],
                                      iris.analysis.MEAN,
                                      weights=weights)
            if cfg.get('temporal_mean'):
                cube = cube.collapsed('time', iris.analysis.MEAN)

            # Save new cube
            new_path = os.path.join(cfg['work_dir'], os.path.basename(path))
            data['filename'] = new_path
            if 'tag' in cfg:
                data['tag'] = cfg['tag']
            write_cube(cube, data, new_path, cfg)
    else:
        logger.warning("Cannot save netcdf files because 'write_netcdf' is "
                       "set to 'False' in user configuration file.")


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
