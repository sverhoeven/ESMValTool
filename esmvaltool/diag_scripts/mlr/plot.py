#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple plots for MLR output (absolute values and biases).

Description
-----------
This diagnostic creates simple plots for MLR model output (absolute plots and
biases).

Author
------
Manuel Schlund (DLR, Germany)

Project
-------
CRESCENDO

Configuration options in recipe
-------------------------------
abs_plot : dict, optional
    Specify additional keyword arguments for the absolute plotting function by
    `plot_kwargs` and plot appearance options by `pyplot_kwargs` (processed as
    functions of :mod:`matplotlib.pyplot`).
bias_plot : dict, optional
    Specify additional keyword arguments for the absolute plotting function by
    `plot_kwargs` and plot appearance options by `pyplot_kwargs` (processed as
    functions of :mod:`matplotlib.pyplot`).
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
import seaborn as sns

import esmvaltool.diag_scripts.shared.iris_helpers as ih
from esmvaltool.diag_scripts import mlr
from esmvaltool.diag_scripts.shared import (get_plot_filename, group_metadata,
                                            plot, run_diagnostic,
                                            select_metadata)

logger = logging.getLogger(os.path.basename(__file__))


def get_cube_dict(input_data):
    """Get dictionary of mean cubes (values) with `var_type` (keys)."""
    grouped_datasets = group_metadata(input_data, 'var_type')
    cube_dict = {}
    for (var_type, datasets) in grouped_datasets.items():
        logger.info("Found var_type '%s'", var_type)
        try:
            cube = ih.get_mean_cube(datasets)
        except iris.exceptions.MergeError:
            logger.warning("Merging of var_type '%s' data failed, skipping it",
                           var_type)
            continue
        dataset_names = list({d['dataset'] for d in datasets})
        projects = list({d['project'] for d in datasets})
        start_years = list({d['start_year'] for d in datasets})
        end_years = list({d['end_year'] for d in datasets})
        cube.attributes.update({
            'dataset': '|'.join(dataset_names),
            'end_year': min(end_years),
            'project': '|'.join(projects),
            'start_year': min(start_years),
            'tag': datasets[0]['tag'],
            'var_type': var_type,
        })
        cube_dict[var_type] = cube
    return cube_dict


def get_input_datasets(cfg):
    """Get grouped datasets (by tag)."""
    input_data = mlr.get_input_data(cfg, pattern=cfg.get('pattern'))
    tags = list(group_metadata(input_data, 'tag').keys())
    if len(tags) > 1:
        logger.warning(
            "Got multiple tags %s, processing only first one ('%s')", tags,
            tags[0])
    return select_metadata(input_data, tag=tags[0])


def get_plot_kwargs(cfg, option):
    """Get keyword arguments for desired plot function."""
    return cfg.get(option, {}).get('plot_kwargs', {})


def process_pyplot_kwargs(cfg, option):
    """Process functions for :mod:`matplotlib.pyplot`."""
    for (key, val) in cfg.get(option, {}).get('pyplot_kwargs', {}).items():
        getattr(plt, key)(val)


def plot_abs(cfg, cube_dict):
    """Plot absolute values of datasets."""
    logger.info("Creating absolute plots")
    for (var_type, cube) in cube_dict.items():
        logger.debug("Plotting absolute plot for var_type '%s'", var_type)
        attrs = cube.attributes
        plot_kwargs = {
            'cbar_label': f"{attrs['tag']} / cube.units",
            'cmap': 'YlOrBr',
        }
        plot_kwargs.update(get_plot_kwargs(cfg, 'abs_plot'))
        plot.global_contourf(cube, **plot_kwargs)
        plt.title(f"{var_type} ({attrs['start_year']} - {attrs['end_year']})")
        process_pyplot_kwargs(cfg, 'abs_plot')
        plot_path = get_plot_filename(f'abs_{var_type}', cfg)
        plt.savefig(plot_path, bbox_inches='tight', orientation='landscape')
        logger.info("Wrote %s", plot_path)
        plt.close()


def main(cfg):
    """Run the diagnostic."""
    sns.set(**cfg.get('seaborn_settings', {}))
    input_data = get_input_datasets(cfg)
    cube_dict = get_cube_dict(input_data)

    # Plots
    plot_abs(cfg, cube_dict)


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
