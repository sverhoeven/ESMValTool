#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnostic script to create one GBRT for multiple climate models.

Description
-----------
This diagnostic creates one "Gradient Boosted Regression Trees" (GBRT) model to
predict future climate for all climate model given in the recipe.

Notes
-----
All datasets must have the attribute 'var_type' which specifies this dataset.
Possible values are 'feature' (independent variables used for
training/testing), 'label' (dependent variables, y-axis) or 'prediction_input'
(independent variables used for prediction of dependent variables, usually
observational data). All 'feature' and 'label' datasets must have the same
shape, except the attribute 'broadcast_from' is set to a list of suitable
coordinate indices (must be done for each feature/label). This also applies to
the 'prediction_input' data sets.

Author
------
Manuel Schlund (DLR, Germany)

Project
-------
CRESCENDO

Configuration options in recipe
-------------------------------
parameters : dict, optional
    Paramter used in the classifier, more information is available here:
    https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting.

"""

import logging
import os

import iris
import numpy as np

from esmvaltool.diag_scripts.shared.gbrt import GBRTBase
from esmvaltool.diag_scripts.shared import (group_metadata, run_diagnostic,
                                            sorted_metadata)

logger = logging.getLogger(os.path.basename(__file__))


class IntraModelGBRT(GBRTBase):
    """Intra-model GBRT diagnostic."""

    def __init__(self, cfg):
        """Initialize class members.

        Parameters
        ----------
        cfg : dict
            Diagnostic script configuration.

        """
        super().__init__(cfg)
        self.classes['climate_models'] = None

    def _append_ensemble_to_name(self, datasets):  # noqa
        """Append ensemble to dataset name."""
        for dataset in datasets:
            dataset['dataset'] += '_{}'.format(dataset['ensemble'])
        return datasets

    def _check_climate_models(self, climate_models):
        """Check if `climate_models` match with already saved data."""
        if self.classes['climate_models'] is None:
            self.classes['climate_models'] = climate_models
        else:
            if climate_models != self.classes['climate_models']:
                raise ValueError("Got different climate models for different "
                                 "var_types, '{}' and '{}'".format(
                                     climate_models,
                                     self.classes['climate_models']))

    def _check_cube_shape(self, cube, climate_model):  # noqa
        """Check shape of a given cube."""
        if cube.shape != (1, ):
            raise ValueError("Expected only cubes with shape (1,), got {} "
                             "from climate model {}".format(
                                 cube.shape, climate_model))

    def _collect_x_data(self, datasets, var_type):
        """Collect x data from `datasets`."""
        datasets = self._append_ensemble_to_name(datasets)
        climate_models = []
        x_data = []
        feature_names = []
        cube = None

        # Iterate over all climate models:
        for (climate_model, model_datasets) in group_metadata(
                datasets, 'dataset', sort=True).values():
            climate_models.append(climate_model)
            model_datasets = sorted_metadata(model_datasets, 'label')
            model_data = []
            names = []
            for dataset in model_datasets:
                cube = iris.load_cube(dataset['filename'])
                self._check_cube_shape(cube, climate_model)
                model_data.append(cube.data)
                names.append(dataset['label'])

            # Check features
            if not feature_names:
                feature_names = names
            else:
                if feature_names != names:
                    raise ValueError("Expected identical features from all "
                                     "climate models, got '{}' from {} and "
                                     "'{}' from {}".format(
                                         feature_names, climate_models[0],
                                         names, climate_model))
            x_data.append(model_data)

        # Check if data was found
        if cube is None:
            raise ValueError("No '{}' datasets found".format(var_type))

        # Convert data to numpy array with correct shape
        x_data = np.array(x_data)
        return (x_data, feature_names, cube)

    def _collect_y_data(self, datasets):
        """Collect y data from `datasets`."""
        datasets = self._append_ensemble_to_name(datasets)
        climate_models = []
        y_data = []
        label_name = None
        cube = None

        # Iterate over all climate models
        for (climate_model, dataset) in group_metadata(
                datasets, 'dataset', sort=True).items():
            if len(dataset) > 1:
                raise ValueError("Expected exactly one 'label' dataset, got "
                                 "{}".format(len(dataset)))
            dataset = dataset[0]
            climate_models.append(climate_model)

            # Save data
            cube = iris.load_cube(dataset['filename'])
            self._check_cube_shape(cube, climate_model)
            y_data.append(cube.data)

            # Check label
            if label_name is None:
                label_name = dataset['label']
            else:
                if dataset['label'] != label_name:
                    raise ValueError("Expected exactly one label for var_type "
                                     "'label', got '{}' and '{}'".format(
                                         label_name, dataset['label']))

        # Check if data was found
        if cube is None:
            raise ValueError("No 'label' datasets found")

        # Check climate models
        self._check_climate_models(climate_models)

        # Return data
        y_data = np.array(y_data)
        return (y_data.ravel(), label_name, cube)

    def _group_training_datasets(self, datasets):
        """Group input datasets (one GBRT model for every climate model)."""
        return {None: datasets}


def main(cfg):
    """Run the diagnostic."""
    gbrt = IntraModelGBRT(cfg)
    logger.info("Initialized GBRT model with parameters %s", gbrt.parameters)

    # Fit and predict
    gbrt.fit()
    gbrt.predict()

    # Plots
    gbrt.plot_feature_importance()
    gbrt.plot_partial_dependence()
    gbrt.plot_prediction_error()


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
