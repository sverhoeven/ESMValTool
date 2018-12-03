#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnostic script to create one GBRT model for multiple climate models.

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
from pprint import pformat

import iris
import matplotlib  # noqa
import matplotlib.pyplot as plt  # noqa
import numpy as np

from esmvaltool.diag_scripts.gbrt import GBRTBase
from esmvaltool.diag_scripts.shared import (group_metadata, run_diagnostic,
                                            sorted_metadata)

matplotlib.use('Agg')

logger = logging.getLogger(os.path.basename(__file__))


class InterModelGBRT(GBRTBase):
    """Inter-model GBRT diagnostic."""

    def __init__(self, cfg):
        """Initialize class members.

        Parameters
        ----------
        cfg : dict
            Diagnostic script configuration.

        """
        super().__init__(cfg)
        self.classes['climate_models'] = None

    def plot_scatterplot(self, filename=None):
        """Plot scatterplot label vs. feature for every feature."""
        if not self._is_ready_for_plotting():
            return
        if filename is None:
            filename = 'scatterplot_{feature}'
        (_, axes) = plt.subplots()

        # Plot scatterplot for every feature
        for (f_idx, feature) in enumerate(self.classes['features']):
            for (m_idx, model) in enumerate(self.classes['climate_models']):
                x_data = self._data[None]['x_data'][m_idx, f_idx]
                y_data = self._data[None]['y_data'][m_idx]
                axes.scatter(x_data, y_data, label=model)
            axes.set_title(feature)
            axes.set_xlabel(feature)
            axes.set_ylabel(self.classes['label'])
            new_filename = (filename.format(feature=feature) + '.' +
                            self._cfg['output_file_type'])
            new_path = os.path.join(self._cfg['plot_dir'], new_filename)
            legend = axes.legend(
                loc='center left',
                ncol=2,
                bbox_to_anchor=[1.05, 0.5],
                borderaxespad=0.0)
            plt.savefig(
                new_path,
                orientation='landscape',
                bbox_inches='tight',
                additional_artists=[legend])
            axes.clear()
        plt.close()

    def _append_ensemble_to_name(self, datasets):  # noqa
        """Append ensemble to dataset name."""
        for dataset in datasets:
            if 'ensemble' in dataset:
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
        allowed_shape = ()
        if cube.shape != allowed_shape:
            raise ValueError("Expected only cubes with shape {}, got {} "
                             "from climate model {}".format(
                                 allowed_shape, cube.shape, climate_model))

    def _collect_x_data(self, datasets, var_type):
        """Collect x data from `datasets`."""
        datasets = self._append_ensemble_to_name(datasets)
        climate_models = []
        x_data = []
        feature_names = []
        cube = None

        # Iterate over all climate models:
        for (climate_model, model_datasets) in group_metadata(
                datasets, 'dataset', sort=True).items():
            climate_models.append(climate_model)
            model_datasets = sorted_metadata(model_datasets, 'label')
            model_data = []
            names = []
            for dataset in model_datasets:
                cube = iris.load_cube(dataset['filename'])

                # FIXME
                # cube = cube.collapsed('time', iris.analysis.MEAN)

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

        # Check climate models
        if var_type == 'feature':
            self._check_climate_models(climate_models)

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

            # FIXME
            # cube = cube.collapsed('time', iris.analysis.MEAN)

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
    gbrt = InterModelGBRT(cfg)
    logger.info("Initialized GBRT model with parameters %s", gbrt.parameters)

    # Fit and predict
    gbrt.fit()
    predictions = gbrt.predict()
    logger.info("Predictions:")
    logger.info("%s", pformat(predictions))

    # Plots
    gbrt.plot_scatterplot()
    gbrt.plot_feature_importance()
    gbrt.plot_partial_dependence()
    gbrt.plot_prediction_error()


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
