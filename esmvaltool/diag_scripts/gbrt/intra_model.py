#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Diagnostic script to create GBRT models for every climate model.

Description
-----------
This diagnostic performs the machine learning algorithm "Gradient Boosting
Regression Trees" for climate predictions for every individual climate model
given in the recipe.

Notes
-----
All datasets must have the attribute 'var_type' which specifies this dataset.
Possible values are 'feature' (independent variables used for
training/testing), 'label' (dependent variables, y-axis) or 'prediction_input'
(independent variables used for prediction of dependent variables, usually
observational data). All 'feature' and 'label' datasets must have the same
shape, also all 'prediction_input' datasets, but these two might be different.

Author
------
Manuel Schlund (DLR, Germany)

Project
-------
CRESCENDO

Configuration options in recipe
-------------------------------
test : str
    This is a test option.

"""


import logging
import os
from pprint import pprint

import iris
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb

from esmvaltool.diag_scripts.shared import (group_metadata, run_diagnostic,
                                            select_metadata)
from esmvaltool.preprocessor._derive.uajet import DerivedVariable as Uajet

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(os.path.basename(__file__))


def main(cfg):
    """Run the diagnostic."""
    input_data = cfg['input_data'].values()

    # Extract datasets and variables
    all_features = select_metadata(input_data, var_type='feature')
    all_labels = select_metadata(input_data, var_type='label')
    all_prediction_input = select_metadata(input_data,
                                           var_type='prediction_input')

    # GBRT for every dataset (climate model)
    for (model_name, features) in group_metadata(
            all_features, 'dataset').items():
        logger.info("Processing %s", model_name)

        # Extract features
        x_data = []
        feature_names = []
        for feature in features:
            feature_cube = iris.load_cube(feature['filename'])
            x_data.append(feature_cube.data.ravel())
            feature_names.append(feature.get('label', feature['short_name']))
        x_data = np.array(x_data)
        x_data = np.swapaxes(x_data, 0, 1)
        feature_names = np.array(feature_names)

        # Extract labels
        label = select_metadata(all_labels, dataset=model_name)
        if len(label) == 1:
            label = label[0]
        else:
            raise ImportError("Expected exactly one dataset with var_type "
                              "'label' for model %s in recipe", model_name)
        label_cube = iris.load_cube(label['filename'])
        y_data = label_cube.data.ravel()

        # Separate training and test data
        (x_train, x_test, y_train, y_test) = train_test_split(x_data, y_data)

        # Create regression model
        params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
                  'learning_rate': 0.01, 'loss': 'ls'}
        clf = GradientBoostingRegressor(**params)
        clf.fit(x_train, y_train)

        # Plot training deviance
        if cfg['write_plots']:
            (fig, axes) = plt.subplots()

            # Compute test set deviance
            test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
            for (idx, y_pred) in enumerate(clf.staged_predict(x_test)):
                test_score[idx] = clf.loss_(y_test, y_pred)

            # Plot
            axes.plot(np.arange(params['n_estimators']) + 1, clf.train_score_,
                      'b-', label='Training Set Deviance')
            axes.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
                      label='Test Set Deviance')
            axes.legend(loc='upper right')
            axes.set_title('Deviance')
            axes.set_xlabel('Boosting Iterations')
            axes.set_ylabel('Deviance')
            new_path = os.path.join(cfg['plot_dir'],
                                    '{}_prediction_error.{}'.format(
                                        model_name, cfg['output_file_type']))
            plt.savefig(new_path)
            logger.info("Wrote %s", new_path)
            plt.close()

        # Plot feature importance
        if cfg['write_plots']:
            (fig, axes) = plt.subplots()
            feature_importance = clf.feature_importances_
            sorted_idx = np.argsort(feature_importance)
            pos = np.arange(sorted_idx.shape[0]) + 0.5
            axes.barh(pos, feature_importance[sorted_idx], align='center')
            axes.set_title('Variable Importance')
            axes.set_yticks(pos)
            axes.set_yticklabels(feature_names[sorted_idx])
            axes.set_xlabel('Relative Importance')
            new_path = os.path.join(cfg['plot_dir'],
                                    '{}_relative_importance.{}'.format(
                                        model_name, cfg['output_file_type']))
            plt.savefig(new_path)
            logger.info("Wrote %s", new_path)
            plt.close()

        # Prediction
        x_pred = []
        for feature_name in feature_names:
            data = select_metadata(all_prediction_input,
                                   label=feature_name)
            if len(data) == 1:
                data = data[0]
            else:
                raise ImportError("Expected exactly one dataset with "
                                  "var_type 'prediction_input' for "
                                  "feature %s", feature_name)
            cube = iris.load_cube(data['filename'])
            original_shape = cube.shape
            x_pred.append(cube.data.ravel())
        x_pred = np.array(x_pred)
        x_pred = np.swapaxes(x_pred, 0, 1)
        prediction = clf.predict(x_pred)
        cube.data = prediction.reshape(original_shape)
        cube.attributes.update(params)
        logger.info("Prediction successful:")
        logger.info(cube)

        # Save cube if desired
        if cfg['write_netcdf']:
            new_path = os.path.join(cfg['work_dir'],
                                    '{}_prediction.nc'.format(model_name))
            iris.save(cube, new_path)
            logger.info("Wrote %s", new_path)

        # Calculate jet positions
        uajet = Uajet('ua')
        cubes = iris.cube.CubeList([cube])
        predicted_uajet = uajet.calculate(cubes)
        label = select_metadata(all_labels, dataset=model_name)[0]
        cubes = iris.cube.CubeList([iris.load_cube(label['filename'])])
        label_uajet = uajet.calculate(cubes)
        logger.info("Predicted jet position: %f", predicted_uajet.data)
        logger.info("Model projected jet position: %f", label_uajet.data)


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
