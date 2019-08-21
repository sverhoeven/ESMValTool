#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple evaluation of residuals (coming from MLR model output).

Description
-----------
This diagnostic evaluates residuals created by MLR models.

Author
------
Manuel Schlund (DLR, Germany)

Project
-------
CRESCENDO

Configuration options in recipe
-------------------------------
pattern : str, optional
    Pattern matched against ancestor files.
seaborn_settings : dict, optional
    Options for seaborn's `set()` method (affects all plots), see
    <https://seaborn.pydata.org/generated/seaborn.set.html>.

"""

import logging
import os

import iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import esmvaltool.diag_scripts.shared.iris_helpers as ih
from esmvaltool.diag_scripts import mlr
from esmvaltool.diag_scripts.shared import (get_plot_filename, group_metadata,
                                            plot, run_diagnostic,
                                            select_metadata)

logger = logging.getLogger(os.path.basename(__file__))


def get_grouped_datasets(cfg):
    """Get grouped datasets (by tag)."""
    input_data = mlr.get_input_data(cfg, pattern=cfg.get('pattern'))
    logger.debug("Extracting 'prediction_output' and 'prediction_residual'")
    out_data = select_metadata(input_data, var_type='prediction_output')
    res_data = select_metadata(input_data, var_type='prediction_residual')
    input_data = out_data + res_data
    return group_metadata(input_data, 'tag')


def plot_boxplot(cfg, datasets, tag):
    """Plot boxplot."""
    mlr_models_rmse = []
    datasets = select_metadata(datasets, var_type='prediction_residual')
    grouped_datasets = group_metadata(datasets, 'mlr_model_name')

    # Collect data of for every MLR model
    for (model_name, model_datasets) in grouped_datasets.items():
        rmse_data = []
        for dataset in model_datasets:
            cube = iris.load_cube(dataset['filename'])
            rmse_data.append(np.ma.sqrt(np.ma.mean(cube.data**2)))
        data_frame = pd.DataFrame(rmse_data, columns=[model_name])
        mlr_models_rmse.append(data_frame)
    boxplot_data = pd.concat(mlr_models_rmse, axis=1)

    # Plot
    boxplot_kwargs = {
        'data': boxplot_data,
        'showfliers': False,
        'showmeans': True,
        'meanprops': {
            'marker': 'x',
            'markeredgecolor': 'black',
            'markerfacecolor': 'black',
            'markersize': 10,
        },
        'whis': 'range',
    }
    sns.boxplot(**boxplot_kwargs)
    sns.stripplot(data=boxplot_data, color='black')

    # Plot appearance
    plt.title('RMSE for different statistical models')
    plt.ylabel(f"{tag} / {datasets[0]['units']}")
    plt.ylim(0.0, plt.ylim()[1])

    # Save plot
    plot_path = get_plot_filename(f'{tag}_boxplot', cfg)
    plt.savefig(plot_path, bbox_inches='tight', orientation='landscape')
    logger.info("Wrote %s", plot_path)
    plt.close()


def plot_relative_errors(cfg, datasets, tag):
    """Plot relative errors for every MLR model."""
    grouped_datasets = group_metadata(datasets, 'mlr_model_name')
    for (model_name, model_datasets) in grouped_datasets.items():
        logger.debug("Plotting relative error of MLR model '%s'", model_name)
        res_datasets = select_metadata(model_datasets,
                                       var_type='prediction_residual')
        res_datasets = group_metadata(res_datasets, 'prediction_name')
        abs_datasets = select_metadata(model_datasets,
                                       var_type='prediction_output')
        abs_datasets = group_metadata(abs_datasets, 'prediction_name')
        rel_error_cubes = iris.cube.CubeList()

        # Calculate relative errors for every prediction
        for pred_name in res_datasets:
            if pred_name is None:
                pred_name = 'unnamed_prediction'
            if pred_name not in abs_datasets:
                logger.warning(
                    "No 'prediction_output' for '%s' of MLR model '%s' "
                    "available, skipping it for relative error calculation",
                    pred_name, model_name)
                continue
            if len(res_datasets[pred_name]) > 1:
                logger.warning(
                    "Multiple 'prediction_residual' for '%s' of MLR model "
                    "'%s' given, using only first one (%s)", pred_name,
                    model_name, res_datasets[pred_name][0]['filename'])
            if len(abs_datasets[pred_name]) > 1:
                logger.warning(
                    "Multiple 'prediction_output' for '%s' of MLR model '%s' "
                    "given, using only first one (%s)", pred_name, model_name,
                    abs_datasets[pred_name][0]['filename'])
            res_cube = iris.load_cube(res_datasets[pred_name][0]['filename'])
            abs_cube = iris.load_cube(abs_datasets[pred_name][0]['filename'])
            res_cube.data = np.ma.abs(res_cube.data)
            res_cube.data /= abs_cube.data
            ih.preprocess_cube_before_merging(res_cube, pred_name)
            rel_error_cubes.append(res_cube)

        # Merge cubes if possible
        if not rel_error_cubes:
            logger.warning(
                "No relative errors available for MLR model '%s', skipping "
                "relative error plotting", model_name)
            continue
        rel_error_cube = rel_error_cubes.merge_cube()
        if len(rel_error_cubes) > 1:
            rel_error_cube = rel_error_cube.collapsed(['cube_label'],
                                                      iris.analysis.MEAN)

        # Create plot
        rel_error_plot = plot.global_contourf(rel_error_cube)
        plot_path = get_plot_filename(f'{tag}_{model_name}_relative_error',
                                      cfg)
        plt.savefig(plot_path, bbox_inches='tight', orientation='landscape')
        logger.info("Wrote %s", plot_path)
        plt.close()


def main(cfg):
    """Run the diagnostic."""
    sns.set(**cfg.get('seaborn_settings', {}))
    grouped_datasets = get_grouped_datasets(cfg)

    # Process data
    for (tag, datasets) in grouped_datasets.items():
        logger.info("Processing tag '%s'", tag)

        # Plots
        plot_boxplot(cfg, datasets, tag)
        plot_relative_errors(cfg, datasets, tag)


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
