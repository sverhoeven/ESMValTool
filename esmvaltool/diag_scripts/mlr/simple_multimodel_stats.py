#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple multi-model statistics to evaluate MLR models.

Description
-----------
This diagnostic applies simple multi-model statistics to evaluate MLR models.

Author
------
Manuel Schlund (DLR, Germany)

Project
-------
CRESCENDO

Configuration options in recipe
-------------------------------
convert_units_to : str, optional
    Convert units of the input data. Can also be given as dataset option.
unweighted_mean : bool, optional (default: False)
    Calculate unweighted multi-model mean.
median : bool, optional (default: False)
    Calculate multi-model median.
std : bool, optional (default: False)
    Calculate multi-model standard deviation.
pattern : str, optional
    Pattern matched against ancestor files to retrieve variables.
pattern_errors : str, optional
    Pattern matched against ancestor files to retrieve variable errors.
plot : bool, optional (default: True)
    Plot results.
var : bool, optional (default: False)
    Calculate multi-model variance.

"""

import logging
import os
from pprint import pformat

import iris
import matplotlib.pyplot as plt
import numpy as np

from esmvaltool.diag_scripts.shared import (get_diagnostic_filename,
                                            get_plot_filename, io,
                                            run_diagnostic, select_metadata)

logger = logging.getLogger(os.path.basename(__file__))

STATS = {
    'unweighted_mean': iris.analysis.MEAN,
    'median': iris.analysis.MEDIAN,
    'std': iris.analysis.STD_DEV,
    'var': iris.analysis.VARIANCE,
}


def _extract_error_cube(cfg, error_data, dataset):
    """Extract error cube for specific dataset."""
    err_data = select_metadata(error_data, dataset=dataset)
    if not err_data:
        logger.warning("No errors for dataset '%s' available, skipping",
                       dataset)
        return None
    if len(err_data) > 1:
        logger.warning(
            "Got multiple files (%s) for error of '%s', using first "
            "one", [d['filename'] for d in err_data], dataset)
    err_data = err_data[0]
    err_cube = iris.load_cube(err_data['filename'])
    convert_units(cfg, err_cube, err_data)
    preprocess_cube(err_cube, dataset)
    return err_cube


def _get_plot_elements(cubes, error_cubes, datasets):
    """Get elements needed for the plots."""
    cubes_to_plot = {}
    error_cubes_to_plot = {}
    for (idx, dataset) in enumerate(datasets):
        if dataset not in STATS.keys():
            cubes_to_plot[dataset] = cubes[idx]
            if error_cubes is None:
                error_cubes_to_plot[dataset] = None
            else:
                error_cubes_to_plot[dataset] = error_cubes[idx]
        elif dataset == 'median':
            cubes_to_plot[dataset] = cubes[idx]
            error_cubes_to_plot[dataset] = None
        elif dataset == 'unweighted_mean':
            if 'std' in datasets:
                cubes_to_plot[dataset] = cubes[idx]
                error_cubes_to_plot[dataset] = cubes[datasets.index('std')]
            elif 'var' in datasets:
                cubes_to_plot[dataset] = cubes[idx]
                error_cubes_to_plot[dataset] = cubes[datasets.index('var')]
                error_cubes_to_plot[dataset].data = np.ma.sqrt(
                    error_cubes_to_plot[dataset].data)
            else:
                cubes_to_plot[dataset] = cubes[idx]
                error_cubes_to_plot[dataset] = None
    return (cubes_to_plot, error_cubes_to_plot)


def _plot_1d_cubes(cubes, error_cubes):
    """Plot 1-dimensional cubes."""
    (_, axes) = plt.subplots()
    x_coord = cubes[list(cubes.keys())[0]].coord(dimensions=(0, ))
    for (dataset, cube) in cubes.items():
        error_cube = error_cubes[dataset]
        if dataset in ('unweighted_mean', 'median'):
            alpha = 1.0
        else:
            alpha = 0.5
        if error_cube is None:
            axes.plot(x_coord.points, cube.data, label=dataset, alpha=alpha)
        else:
            lines = axes.plot(x_coord.points, cube.data, alpha=alpha)
            axes.fill_between(x_coord.points,
                              cube.data - error_cube.data,
                              cube.data + error_cube.data,
                              facecolor=lines[-1].get_color(),
                              alpha=alpha - 0.3,
                              label=f'{dataset} ± std')
    if iris.util.guess_coord_axis(x_coord) == 'T':
        time = x_coord.units
        (x_ticks, _) = plt.xticks()
        x_labels = time.num2date(x_ticks)
        x_labels = [label.strftime('%Y-%m-%d') for label in x_labels]
        plt.xticks(x_ticks, x_labels, rotation=45)
    else:
        x_units = (x_coord.units.symbol
                   if x_coord.units.origin is None else x_coord.units.origin)
        axes.set_xlabel(f'{x_coord.name()} / {x_units}')
    legend = axes.legend(loc='center left',
                         ncol=2,
                         bbox_to_anchor=[1.05, 0.5],
                         borderaxespad=0.0)
    return (axes, legend)


def _plot_scalar_cubes(cubes, error_cubes):
    """Plot scalar cubes."""
    (_, axes) = plt.subplots()
    datasets = []
    for (idx, (dataset, cube)) in enumerate(cubes.items()):
        error_cube = error_cubes[dataset]
        if error_cube is None:
            axes.scatter(idx, cube.data)
            datasets.append(dataset)
        else:
            axes.errorbar(idx,
                          cube.data,
                          yerr=error_cube.data,
                          fmt='o',
                          capsize=5)
            datasets.append(f'{dataset} ± std')
    plt.xticks(np.arange(len(cubes)), datasets, rotation=90)
    legend = None
    return (axes, legend)


def add_mm_cube_attributes(cube, input_data, stat):
    """Add attribute to cube."""
    projects = {d['project'] for d in input_data}
    project = 'Multiple projects'
    if len(projects) == 1:
        project = projects.pop()

    # Modify attributes
    attrs = cube.attributes
    attrs['dataset'] = f'Multi-model {stat}'
    attrs['project'] = project


def convert_units(cfg, cube, data):
    """Convert units if desired."""
    cfg_settings = cfg.get('convert_units_to')
    data_settings = data.get('convert_units_to')
    if cfg_settings or data_settings:
        units_to = cfg_settings
        if data_settings:
            units_to = data_settings
        logger.info("Converting units from '%s' to '%s'", cube.units.symbol,
                    units_to)
        try:
            cube.convert_units(units_to)
        except ValueError:
            logger.warning("Cannot convert units from '%s' to '%s'",
                           cube.units.symbol, units_to)


def extract_data(cfg, input_data, error_data):
    """Extract data."""
    cubes = iris.cube.CubeList()
    error_cubes = iris.cube.CubeList()
    datasets = []
    for data in input_data:
        dataset = data['dataset']
        logger.info("Processing '%s'", dataset)

        # Error data
        if error_data is not None:
            err_cube = _extract_error_cube(cfg, error_data, dataset)
            if err_cube is None:
                continue
            error_cubes.append(err_cube)

        # Regular data
        cube = iris.load_cube(data['filename'])
        convert_units(cfg, cube, data)
        preprocess_cube(cube, dataset)
        datasets.append(dataset)
        cubes.append(cube)
    if not error_cubes:
        error_cubes = None
    return (cubes, error_cubes, datasets)


def get_input_files(cfg):
    """Get input files (including errors if desired)."""
    input_data = list(cfg['input_data'].values())
    input_data.extend(io.netcdf_to_metadata(cfg, pattern=cfg.get('pattern')))
    paths = [d['filename'] for d in input_data]
    logger.debug("Found regular files")
    logger.debug(pformat(paths))

    # Errors
    error_data = None
    if cfg.get('pattern_errors'):
        error_data = io.netcdf_to_metadata(cfg, pattern=cfg['pattern_errors'])
        erro_paths = [d['filename'] for d in error_data]
        logger.debug("Found error files")
        logger.debug(pformat(erro_paths))
    return (input_data, error_data)


def plot(cfg, cubes, error_cubes, datasets):
    """Plot calculated data."""
    if not cfg.get('plot', True):
        return
    if not cfg['write_plots']:
        logger.debug(
            "Plotting not possible, 'write_plots' is set to 'False' in user "
            "configuration file")
        return
    (cubes_to_plot,
     error_cubes_to_plot) = _get_plot_elements(cubes, error_cubes, datasets)

    # Plot
    if cubes[0].ndim == 0:
        plot_func = _plot_scalar_cubes
    elif cubes[0].ndim == 1:
        plot_func = _plot_1d_cubes
    else:
        logger.warning("Plotting of %i-dimensional cubes is not supported yet",
                       cubes[0].ndim)
        return
    (axes, legend) = plot_func(cubes_to_plot, error_cubes_to_plot)

    # Set plot appearance and save it
    y_units = (cubes[0].units.symbol
               if cubes[0].units.origin is None else cubes[0].units.origin)
    axes.set_ylabel(f'{cubes[0].var_name} / {y_units}')
    axes.set_title(f'{cubes[0].var_name} for multiple datasets')
    path = get_plot_filename('multi-model_stats', cfg)
    plt.savefig(path,
                orientation='landscape',
                bbox_inches='tight',
                additional_artists=[legend])
    logger.info("Wrote %s", path)
    plt.close()


def preprocess_cube(cube, dataset):
    """Preprocess single cubes."""
    cube.attributes = {}
    cube.cell_methods = ()
    for coord in cube.coords(dim_coords=False):
        cube.remove_coord(coord)
    dataset_coord = iris.coords.AuxCoord(dataset,
                                         var_name='dataset',
                                         long_name='dataset')
    cube.add_aux_coord(dataset_coord, [])


def main(cfg):
    """Run the diagnostic."""
    (input_data, error_data) = get_input_files(cfg)
    if not input_data:
        logger.error("No input data found")
        return

    # Extract data
    (cubes, error_cubes, datasets) = extract_data(cfg, input_data, error_data)
    if not cubes:
        logger.error("No input data with regular AND error data available")
        return

    # Merge cubes
    mm_cube = cubes.merge_cube()

    # Calculate desired statistics
    stats = {}
    for (stat, iris_op) in STATS.items():
        if cfg.get(stat):
            stats[stat] = iris_op
    for (stat, iris_op) in stats.items():
        logger.info("Calculating '%s'", stat)
        try:
            new_cube = mm_cube.collapsed('dataset', iris_op)
        except iris.exceptions.CoordinateCollapseError:
            new_cube = mm_cube
        new_path = get_diagnostic_filename(stat, cfg)
        add_mm_cube_attributes(new_cube, input_data, stat)
        io.iris_save(new_cube, new_path)
        datasets.append(stat)
        cubes.append(new_cube)

    # Plot if desired
    plot(cfg, cubes, error_cubes, datasets)


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
