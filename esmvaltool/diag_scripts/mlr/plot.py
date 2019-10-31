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
    ``plot_kwargs`` and plot appearance options by ``pyplot_kwargs`` (processed
    as functions of :mod:`matplotlib.pyplot`).
alias : dict, optional
    :obj:`str` to :obj:`str` mapping for nicer plot labels (e.g.
    ``{'feature': 'Historical CMIP5 data'}``.
bias_plot : dict, optional
    Specify additional keyword arguments for the absolute plotting function by
    ``plot_kwargs`` and plot appearance options by ``pyplot_kwargs`` (processed
    as functions of :mod:`matplotlib.pyplot`).
ignore_var_types : list of str, optional
    Ignore specific ``var_type``s.
pattern : str, optional
    Pattern matched against ancestor files.
savefig_kwargs : dict, optional
    Keyword arguments for :func:`matplotlib.pyplot.savefig`.
seaborn_settings : dict, optional
    Options for :func:`seaborn.set` (affects all plots), see
    <https://seaborn.pydata.org/generated/seaborn.set.html>.

"""

import itertools
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


def _get_alias(cfg, name):
    """Get alias for given ``name``."""
    return cfg.get('aliases', {}).get(name, name)


def get_cube_dict(cfg, input_data):
    """Get dictionary of mean cubes (values) with ``var_type`` (keys)."""
    cube_dict = {}
    for (var_type, datasets) in group_metadata(input_data, 'var_type').items():
        if var_type in cfg.get('ignore_var_types', []):
            logger.debug("Ignored var_type '%s'", var_type)
            continue
        grouped_datasets = group_metadata(datasets, 'mlr_model_name')
        for (model_name, model_datasets) in grouped_datasets.items():
            if model_name is not None:
                key = f'{var_type}_{model_name}'
            else:
                key = var_type
            logger.info("Found cube for '%s'", key)
            try:
                cube = ih.get_mean_cube(model_datasets)
            except iris.exceptions.MergeError as exc:
                logger.warning("Merging of var_type '%s' data failed: %s",
                               var_type, str(exc))
                continue
            dataset_names = list({d['dataset'] for d in model_datasets})
            projects = list({d['project'] for d in model_datasets})
            start_years = list({d['start_year'] for d in model_datasets})
            end_years = list({d['end_year'] for d in model_datasets})
            cube.attributes.update({
                'dataset': '|'.join(dataset_names),
                'end_year': min(end_years),
                'project': '|'.join(projects),
                'start_year': min(start_years),
                'tag': model_datasets[0]['tag'],
                'var_type': var_type,
            })
            if model_name is not None:
                cube.attributes['mlr_model_name'] = model_name
            cube_dict[key] = cube
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


def get_savefig_kwargs(cfg):
    """Get keyword arguments for :func:`matplotlib.pyplot.savefig`."""
    if 'savefig_kwargs' in cfg:
        return cfg['savefig_kwargs']
    savefig_kwargs = {
        'bbox_inches': 'tight',
        'dpi': 300,
        'orientation': 'landscape',
    }
    return savefig_kwargs


def process_pyplot_kwargs(cfg, option):
    """Process functions for :mod:`matplotlib.pyplot`."""
    for (key, val) in cfg.get(option, {}).get('pyplot_kwargs', {}).items():
        getattr(plt, key)(val)


def plot_abs(cfg, cube_dict):
    """Plot absolute values of datasets."""
    logger.info("Creating absolute plots")
    for (key, cube) in cube_dict.items():
        logger.debug("Plotting absolute plot for '%s'", key)
        attrs = cube.attributes

        # Plot
        plot_kwargs = {
            'cbar_label': f"{attrs['tag']} / {cube.units}",
            'cmap': 'YlOrBr',
        }
        plot_kwargs.update(get_plot_kwargs(cfg, 'abs_plot'))
        plot.global_contourf(cube, **plot_kwargs)

        # Plot appearance
        alias = _get_alias(cfg, key)
        plt.title(f"{alias} ({attrs['start_year']}-{attrs['end_year']})")
        process_pyplot_kwargs(cfg, 'abs_plot')

        # Save plot
        plot_path = get_plot_filename(f'abs_{key}', cfg)
        plt.savefig(plot_path, **get_savefig_kwargs(cfg))
        logger.info("Wrote %s", plot_path)
        plt.close()


def plot_biases(cfg, cube_dict):
    """Plot biases of datasets."""
    logger.info("Creating bias plots")
    for (key_1, key_2) in itertools.permutations(cube_dict, 2):
        logger.debug("Plotting bias plot '%s' - '%s'", key_1, key_2)
        cube_1 = cube_dict[key_1]
        cube_2 = cube_dict[key_2]
        attrs_1 = cube_1.attributes
        attrs_2 = cube_2.attributes
        alias_1 = _get_alias(cfg, key_1)
        alias_2 = _get_alias(cfg, key_2)

        # Plot
        diff_cube = cube_1.copy()
        diff_cube.data -= cube_2.data
        plot_kwargs = {
            'cbar_label': f"Δ{attrs_1['tag']} / {diff_cube.units}",
            'cmap': 'bwr',
        }
        plot_kwargs.update(get_plot_kwargs(cfg, 'bias_plot'))
        plot.global_contourf(diff_cube, **plot_kwargs)

        # Plot appearance
        if (attrs_1['start_year'] == attrs_2['start_year']
                and attrs_1['end_year'] == attrs_2['end_year']):
            title = (f"{alias_1} - {alias_2} ({attrs_1['start_year']}-"
                     f"{attrs_1['end_year']})")
        else:
            title = (f"{alias_1} ({attrs_1['start_year']}-"
                     f"{attrs_1['end_year']}) - {alias_2} ("
                     f"{attrs_2['start_year']}-{attrs_2['end_year']})")
        plt.title(title)
        process_pyplot_kwargs(cfg, 'bias_plot')

        # Save plot
        plot_path = get_plot_filename(f'bias_{key_1}-{key_2}', cfg)
        plt.savefig(plot_path, **get_savefig_kwargs(cfg))
        logger.info("Wrote %s", plot_path)
        plt.close()


def main(cfg):
    """Run the diagnostic."""
    sns.set(**cfg.get('seaborn_settings', {}))
    input_data = get_input_datasets(cfg)
    cube_dict = get_cube_dict(cfg, input_data)

    # Plots
    plot_abs(cfg, cube_dict)
    plot_biases(cfg, cube_dict)


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
