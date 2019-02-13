"""General fix for all models of project Ozone_CMIP5_ACC_SPARC."""
import warnings

import iris

import cf_units
import numpy as np


def fix_time_coordinate(cube):
    """Fix time coordinate of Ozone_CMIP5_ACC_SPARC cubes.

    Change units of time axis from `months since ...` to `days since ...` and
    round to months if necessary.

    Parameters
    ----------
    cube: iris.cube.Cube

    Returns
    -------
    iris.cube.Cube

    """
    time = cube.coord('time')
    time.convert_units('days since 1850-01-01 00:00:00')
    new_array = np.copy(time.points)
    for idx in range(time.shape[0]):
        point = time.cell(idx).point
        if point.day >= 15:
            new_month = point.month % 12 + 1
            if new_month == 1:
                new_year = point.year + 1
            else:
                new_year = point.year
            new_time = point.replace(
                year=new_year, month=new_month, day=1, hour=0, minute=0)
            new_array[idx] = cf_units.date2num(new_time, time.units.name,
                                               time.units.calendar)
    time.points = new_array
    cube.var_name = 'tro3'
    return cube


def remove_cell_method(infile, outfile):
    """Apply fixes to the files prior to creating the cube.

    Removes invalid cell method without a warning.

    Parameters
    ----------
    infile : str
        Path to the input file.
    outfile : str
        Path to the output file.

    Returns
    -------
    str
        Path to the output file.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            category=iris.fileformats.netcdf.UnknownCellMethodWarning)
        cube = iris.load_cube(infile)
    cube.cell_methods = ()
    iris.save(cube, outfile)
    return outfile
