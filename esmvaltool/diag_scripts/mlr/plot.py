#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple plots for MLR output (absolute values and biases).

Description
-----------
This diagnostic creates simple plots for MLR model output.

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
    grouped_datasets = group_metadata(datasets, 'mlr_model_name')

    # Collect data of for every MLR model
    for (model_name, model_datasets) in grouped_datasets.items():
        res_datasets = select_metadata(model_datasets,
                                       var_type='prediction_residual')
        rmse_data = []
        for dataset in res_datasets:
            cube = iris.load_cube(dataset['filename'])
            rmse_data.append(np.ma.sqrt(np.ma.mean(cube.data**2)))
        data_frame = pd.DataFrame(rmse_data, columns=[model_name])
        mlr_models_rmse.append(data_frame)
    boxplot_data = pd.concat(mlr_models_rmse, axis=1)

    path = get_plot_filename('map', cfg)
    map_plot = plot.global_contourf(cube)
    print(list(map_plot.__dict__.keys()))
    plt.savefig(path, bbox_inches='tight', orientation='landscape')
    plt.close()

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


def main(cfg):
    """Run the diagnostic."""
    sns.set(**cfg.get('seaborn_settings', {}))
    grouped_datasets = get_grouped_datasets(cfg)

    # Process data
    for (tag, datasets) in grouped_datasets.items():
        logger.info("Processing tag '%s'", tag)

        # Plot boxplot
        plot_boxplot(cfg, datasets, tag)


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
