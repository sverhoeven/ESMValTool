"""Convenience functions for GBRT diagnostics."""

import logging
import os

import iris
import numpy as np

from ._base import (select_metadata, save_iris_cube)

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


class GBRTBase():
    """Base class for GBRT model diagnostics."""

    def __init__(self, cfg):
        """Initialize class members.

        Parameters
        ----------
        cfg : dict
            Diagnostic script configuration.

        """
        self._cfg = cfg
        self._input_datasets = self._get_input_datasets()

    def _get_ancestor_datasets(self):
        """Get ancestor datasets."""
        input_dirs = [
            d for d in self._cfg['input_files']
            if not d.endswith('metadata.yml')
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
                            logger.debug("Skipping %s, attribute '%s' not "
                                         "given", path, key)
                            valid_data = False
                            break
                    if valid_data:
                        datasets.append(dataset_info)
        return datasets

    def _get_broadcasted_data(self, datasets, target_shape):
        """Get broadcasted data."""
        new_data = []
        names = []
        if not datasets:
            return (new_data, names)
        var_type = datasets[0]['var_type']
        if target_shape is None:
            raise ValueError("Expected at least one '{}' dataset without the "
                             "option 'broadcast_from'".format(var_type))
        for data in datasets:
            cube_to_broadcast = iris.load_cube(data['filename'])
            data_to_broadcast = cube_to_broadcast.data
            name = data.get('label', data['short_name'])
            try:
                new_axis_pos = np.delete(np.arange(len(target_shape)),
                                         data['broadcast_from'])
            except IndexError:
                raise ValueError("Broadcasting failed for '{}', index out of "
                                 "bounds".format(name))
            logger.info("Broadcasting %s '%s' from %s to %s", var_type, name,
                        data_to_broadcast.shape, target_shape)
            for idx in new_axis_pos:
                data_to_broadcast = np.expand_dims(data_to_broadcast, idx)
            data_to_broadcast = np.broadcast_to(data_to_broadcast,
                                                target_shape)
            if not self._cfg.get('use_only_coords_as_features'):
                new_data.append(data_to_broadcast.ravel())
                names.append(name)
        return (new_data, names)

    def _get_coordinate_data(self, cube):
        """Get coordinate variables to be used as x data."""
        new_data = []
        names = []
        if cube is None:
            return (new_data, names)
        for (coord, coord_idx) in self._cfg.get(
                'use_coords_as_feature', {}).items():
            coord_array = cube.coord(coord).points
            try:
                new_axis_pos = np.delete(np.arange(len(cube.shape)), coord_idx)
            except IndexError:
                raise ValueError("'use_coords_as_feature' failed, index '{}'"
                                 "is out of bounds for coordinate "
                                 "'{}'".format(coord_idx, coord))
            for idx in new_axis_pos:
                coord_array = np.expand_dims(coord_array, idx)
            coord_array = np.broadcast_to(coord_array, cube.shape)
            new_data.append(coord_array.ravel())
            names.append(coord)
        return (new_data, names)

    def _get_input_datasets(self):
        """Get input data (including ancestors)."""
        input_datasets = list(self._cfg['input_data'].values())
        input_datasets.extend(self._get_ancestor_datasets())
        return input_datasets
