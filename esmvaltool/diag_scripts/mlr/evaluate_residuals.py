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
box_plot : dict, optional
    Specify additional keyword arguments for :func:`seaborn.boxplot` by
    ``plot_kwargs`` and plot appearance options by ``pyplot_kwargs`` (processed
    as functions of :mod:`matplotlib.pyplot`).
pattern : str, optional
    Pattern matched against ancestor files.
residual_plot : dict, optional
    Specify additional keyword arguments for the residual plot function by
    ``plot_kwargs`` and plot appearance options by ``pyplot_kwargs`` (processed
    as functions of :mod:`matplotlib.pyplot`).
savefig_kwargs : dict, optional
    Keyword arguments for :func:`matplotlib.pyplot.savefig`.
seaborn_settings : dict, optional
    Options for :func:`seaborn.set` (affects all plots), see
    <https://seaborn.pydata.org/generated/seaborn.set.html>.

"""

import logging
import os

import iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import esmvaltool.diag_scripts.mlr.plot as mlr_plot
import esmvaltool.diag_scripts.shared.iris_helpers as ih
from esmvaltool.diag_scripts import mlr
from esmvaltool.diag_scripts.shared import (get_plot_filename, group_metadata,
                                            plot, run_diagnostic,
                                            select_metadata)

logger = logging.getLogger(os.path.basename(__file__))


def plot_boxplot(cfg, input_data):
    """Plot boxplot."""
    logger.info("Creating box plot")
    mlr_models_rmse = []
    input_data = select_metadata(input_data, var_type='prediction_residual')
    if not input_data:
        logger.warning(
            "Creating box plot not possible, no 'prediction_residual' data "
            "found")
        return
    grouped_datasets = group_metadata(input_data, 'mlr_model_name')

    # Collect data of for every MLR model
    for (model_name, datasets) in grouped_datasets.items():
        rmse_data = []
        for dataset in datasets:
            cube = iris.load_cube(dataset['filename'])
            weights = mlr.get_area_weights(cube)
            mse = np.ma.average(cube.data**2, weights=weights)
            rmse_data.append(np.ma.sqrt(mse))
        data_frame = pd.DataFrame(rmse_data, columns=[model_name])
        mlr_models_rmse.append(data_frame)
    boxplot_data = pd.concat(mlr_models_rmse, axis=1)

    # Plot
    boxplot_kwargs = {
        'color': 'b',
        'data': boxplot_data,
        'showfliers': False,
        'showmeans': True,
        'meanprops': {
            'marker': 'x',
            'markeredgecolor': 'k',
            'markerfacecolor': 'k',
            'markersize': 10,
        },
        'whis': 'range',
    }
    boxplot_kwargs.update(mlr_plot.get_plot_kwargs(cfg, 'box_plot'))
    sns.boxplot(**boxplot_kwargs)
    sns.swarmplot(data=boxplot_data, color='k', alpha='0.6')

    # Plot appearance
    plt.title('RMSE for different statistical models')
    plt.ylabel(f"{input_data[0]['tag']} / {input_data[0]['units']}")
    plt.ylim(0.0, plt.ylim()[1])
    mlr_plot.process_pyplot_kwargs(cfg, 'box_plot')

    # Save plot
    plot_path = get_plot_filename('boxplot', cfg)
    plt.savefig(plot_path, **mlr_plot.get_savefig_kwargs(cfg))
    logger.info("Wrote %s", plot_path)
    plt.close()


def plot_residuals(cfg, input_data):
    """Plot relative errors for every MLR model."""
    logger.info("Creating residual plots")
    input_data = select_metadata(input_data, var_type='prediction_residual')
    if not input_data:
        logger.warning(
            "Creating residual plot not possible, no 'prediction_residual' "
            "data found")
        return
    plot_kwargs = mlr_plot.get_plot_kwargs(cfg, 'residual_plot')
    grouped_datasets = group_metadata(input_data, 'mlr_model_name')
    for (model_name, datasets) in grouped_datasets.items():
        logger.debug("Plotting residual plots for MLR model '%s'", model_name)
        filename = model_name.lower().replace(' ', '_')
        cubes = iris.cube.CubeList()

        # Plot residuals for every prediction
        pred_groups = group_metadata(datasets, 'prediction_name')
        for (pred_name, pred_datasets) in pred_groups.items():
            if pred_name is None:
                pred_name = 'unnamed_prediction'
            if len(pred_datasets) > 1:
                logger.warning(
                    "Multiple 'prediction_residual' datasets for prediction "
                    "'%s' of MLR model '%s' found, using only first one (%s)",
                    pred_name, model_name, pred_datasets[0]['filename'])
            cube = iris.load_cube(pred_datasets[0]['filename'])

            # Create plot
            plot.global_contourf(cube, **plot_kwargs)
            mlr_plot.process_pyplot_kwargs(cfg, 'residual_plot')
            plot_path = get_plot_filename(f'{filename}_{pred_name}_residual',
                                          cfg)
            plt.savefig(plot_path, **mlr_plot.get_savefig_kwargs(cfg))
            logger.info("Wrote %s", plot_path)
            plt.close()

            # Append cube for merging
            ih.preprocess_cube_before_merging(cube, pred_name)
            cubes.append(cube)

        # Merge cubes to create mean
        mean_cube = cubes.merge_cube()
        if len(cubes) > 1:
            mean_cube = mean_cube.collapsed(['cube_label'], iris.analysis.MEAN)

        # Create mean plot
        plot.global_contourf(mean_cube, **plot_kwargs)
        mlr_plot.process_pyplot_kwargs(cfg, 'residual_plot')
        plot_path = get_plot_filename(
            f'{filename}_mean_of_predictions_residual', cfg)
        plt.savefig(plot_path, **mlr_plot.get_savefig_kwargs(cfg))
        logger.info("Wrote %s", plot_path)
        plt.close()


def main(cfg):
    """Run the diagnostic."""
    sns.set(**cfg.get('seaborn_settings', {}))
    input_data = mlr_plot.get_input_datasets(cfg)

    # Plots
    plot_boxplot(cfg, input_data)
    plot_residuals(cfg, input_data)


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
