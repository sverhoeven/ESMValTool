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
mean : bool, optional (default: False)
    Calculate multi-model mean.
median : bool, optional (default: False)
    Calculate multi-model median.
std : bool, optional (default: False)
    Calculate multi-model standard deviation.
var : bool, optional (default: False)
    Calculate multi-model variance.
plot : bool, optional (default: True)
    Plot results.
convert_units_to : str, optional
    Convert units of the input data. Can also be given as dataset option.
pattern : str, optional
    Pattern matched against ancestor files.

"""

import logging
import os
from pprint import pformat

import iris
import matplotlib.pyplot as plt
import numpy as np

from esmvaltool.diag_scripts.shared import (get_diagnostic_filename,
                                            get_plot_filename, io,
                                            run_diagnostic)

logger = logging.getLogger(os.path.basename(__file__))

STATS = {
    'mean': iris.analysis.MEAN,
    'median': iris.analysis.MEDIAN,
    'std': iris.analysis.STD_DEV,
    'var:': iris.analysis.VARIANCE,
}


def _plot_1d_cubes(cubes, datasets, error=None):
    """Plot 1-dimensional cubes."""
    (_, axes) = plt.subplots()
    x_coord = cubes[0].coord(dimensions=(0, ))
    for (idx, dataset) in enumerate(datasets):
        cube = cubes[idx]
        if dataset not in ('mean', 'median'):
            axes.plot(x_coord.points, cube.data, label=dataset, alpha=0.4)
        else:
            lines = axes.plot(x_coord.points, cube.data, label=dataset)
            if error is not None:
                axes.fill_between(x_coord.points,
                                  cube.data - error,
                                  cube.data + error,
                                  facecolor=lines[-1].get_color(),
                                  alpha=0.4,
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


def _plot_scalar_cubes(cubes, datasets, error=None):
    """Plot scalar cubes."""
    (_, axes) = plt.subplots()
    x_data = np.arange(len(cubes))
    for (idx, dataset) in enumerate(datasets):
        cube = cubes[idx]
        if dataset not in ('mean', 'median'):
            axes.scatter(x_data[idx], cube.data)
        else:
            axes.errorbar(x_data[idx],
                          cube.data,
                          yerr=error,
                          fmt='ro',
                          capsize=5)
            if error is not None:
                datasets[idx] += ' ± std'
    plt.xticks(x_data, datasets, rotation=45)
    legend = None
    return (axes, legend)


def add_mm_cube_attributes(cube, input_data, stat, path):
    """Add attribute to cube."""
    projects = {d['project'] for d in input_data}
    project = 'Multiple projects'
    if len(projects) == 1:
        project = projects.pop()

    # Modify attributes
    attrs = cube.attributes
    attrs['dataset'] = f'Multi-model {stat}'
    attrs['project'] = project
    attrs['filename'] = path


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
        else:
            data['units'] = units_to
    return (cube, data)


def plot(cfg, cubes, datasets):
    """Plot calculated data."""
    if not cfg.get('plot', True):
        return
    if not cfg['write_plots']:
        logger.debug(
            "Plotting not possible, 'write_plots' is set to 'False' in user "
            "configuration file")
        return
    cubes_to_plot = []
    datasets_to_plot = []
    error = None
    for (idx, dataset) in enumerate(datasets):
        if dataset not in ('std', 'var'):
            datasets_to_plot.append(dataset)
            cubes_to_plot.append(cubes[idx])
        if dataset == 'std':
            error = cubes[idx].data

    # Plot
    if cubes[0].ndim == 0:
        plot_func = _plot_scalar_cubes
    elif cubes[0].ndim == 1:
        plot_func = _plot_1d_cubes
    else:
        logger.error("Plotting of %i-dimensional cubes is not supported yet",
                     cubes[0].ndim)
        return
    (axes, legend) = plot_func(cubes_to_plot, datasets_to_plot, error)

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
    input_data = list(cfg['input_data'].values())
    input_data.extend(io.netcdf_to_metadata(cfg, pattern=cfg.get('pattern')))
    paths = [d['filename'] for d in input_data]
    logger.debug("Found files")
    logger.debug(pformat(paths))
    datasets = []
    cubes = iris.cube.CubeList()
    if not input_data:
        logger.error("No input data found")
        return

    # Iterate over all data
    for data in input_data:
        logger.info("Processing %s", data['filename'])
        data = dict(data)
        cube = iris.load_cube(data['filename'])

        # Convert units
        (cube, data) = convert_units(cfg, cube, data)

        # Add dataset coordinate and append to CubeList
        preprocess_cube(cube, data['dataset'])
        datasets.append(data['dataset'])
        cubes.append(cube)

    # Merge cubes
    mm_cube = cubes.merge_cube()

    # Calculate desired statistics
    stats = {}
    for (stat, iris_op) in STATS.items():
        if cfg.get(stat):
            stats[stat] = iris_op
    for (stat, iris_op) in stats.items():
        try:
            new_cube = mm_cube.collapsed('dataset', iris_op)
        except iris.exceptions.CoordinateCollapseError:
            new_cube = mm_cube
        new_path = get_diagnostic_filename(stat, cfg)
        add_mm_cube_attributes(new_cube, input_data, stat, new_path)
        io.save_iris_cube(new_cube, new_path)
        datasets.append(stat)
        cubes.append(new_cube)

    # Plot if desired
    plot(cfg, cubes, datasets)


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
