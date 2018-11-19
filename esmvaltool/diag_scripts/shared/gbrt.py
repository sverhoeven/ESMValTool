"""Convenience functions for GBRT diagnostics."""

import logging
import os

import iris

from ._base import save_iris_cube

logger = logging.getLogger(os.path.basename(__file__))

VAR_KEYS = [
    'long_name',
    'standard_name',
    'units',
]
NECESSARY_KEYS = VAR_KEYS + [
    'dataset',
    'filename',
    'short_name',
    'var_type',
]


def get_ancestor_data(cfg):
    """Get ancestor datasets.

    Parameters
    ----------
    cfg : dict
        Diagnostic script configuration.

    Returns
    -------
    list of dict
        Information for every ancestor dataset similar to `cfg['input_data']`.

    """
    input_dirs = [
        d for d in cfg['input_files'] if not d.endswith('metadata.yml')
    ]
    if not input_dirs:
        logger.debug("Skipping loading ancestor datasets, 'ancestors' key "
                     "not given")
        return []

    # Extract datasets
    datasets = []
    for input_dir in input_dirs:
        for (root, _, files) in os.walk(input_dir):
            for filename in files:
                if '.nc' not in filename:
                    continue
                path = os.path.join(root, filename)
                cube = iris.load_cube(path)
                dataset_info = dict(cube.attributes)
                for var_key in VAR_KEYS:
                    dataset_info[var_key] = getattr(cube, var_key)
                dataset_info['short_name'] = getattr(cube, 'var_name')

                # Check if necessary keys are available
                valid_data = True
                for key in NECESSARY_KEYS:
                    if key not in dataset_info:
                        logger.debug("Skipping %s, attribute '%s' not given",
                                     path, key)
                        valid_data = False
                        break
                if valid_data:
                    datasets.append(dataset_info)

    return datasets


def write_cube(cube, attributes, path, cfg):
    """Write cube with all necessary information for GBRT models.

    Parameters
    ----------
    cube : iris.cube.Cube
        Cube which should be written.
    attributes : dict
        Attributes for the cube (needed for GBRT models).
    path : str
        Path to the new file.
    cfg : dict
        Diagnostic script configuration.

    """
    for key in NECESSARY_KEYS:
        if key not in attributes:
            logger.warning(
                "Cannot save cube to %s, attribute '%s' "
                "not given", path, key)
            return
    for var_key in VAR_KEYS:
        setattr(cube, var_key, attributes.pop(var_key))
    setattr(cube, 'var_name', attributes.pop('short_name'))
    cube.attributes.update(attributes)
    save_iris_cube(cube, path, cfg)
