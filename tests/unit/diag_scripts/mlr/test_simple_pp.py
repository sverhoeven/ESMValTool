"""Unit tests for the module :mod:`esmvaltool.diag_scripts.mlr.simple_pp`."""

import iris
import numpy as np
import pytest

import esmvaltool.diag_scripts.mlr.simple_pp as simple_pp

LONG_NAME_1 = 'loooong name'
LONG_NAME_2 = ':)'
COORD_1 = iris.coords.DimCoord([-2.0, -1.0, 20.0], long_name=LONG_NAME_1)
COORD_2 = iris.coords.DimCoord([-2.0, -1.0, 20.0], long_name=LONG_NAME_2)
COORD_3 = iris.coords.DimCoord([-42.0], long_name=LONG_NAME_1)
COORD_AUX = iris.coords.AuxCoord(['a', 'b', 'c'], long_name=LONG_NAME_2)
CUBE_1 = iris.cube.Cube(np.arange(3.0), dim_coords_and_dims=[(COORD_1, 0)])
CUBE_2 = iris.cube.Cube(np.arange(3.0), dim_coords_and_dims=[(COORD_2, 0)])
CUBE_3 = iris.cube.Cube(np.arange(1.0), dim_coords_and_dims=[(COORD_3, 0)])
CUBE_4 = iris.cube.Cube(np.arange(3.0),
                        dim_coords_and_dims=[(COORD_1, 0)],
                        aux_coords_and_dims=[(COORD_AUX, 0)])
TEST_CHECK_COORDS = [
    (CUBE_1, [LONG_NAME_1], True),
    (CUBE_2, [LONG_NAME_1], False),
    (CUBE_3, [LONG_NAME_1], False),
    (CUBE_4, [LONG_NAME_1], True),
    (CUBE_1, [LONG_NAME_1, LONG_NAME_2], False),
    (CUBE_2, [LONG_NAME_1, LONG_NAME_2], False),
    (CUBE_3, [LONG_NAME_1, LONG_NAME_2], False),
    (CUBE_4, [LONG_NAME_1, LONG_NAME_2], True),
]


@pytest.mark.parametrize('cube,coords,output', TEST_CHECK_COORDS)
def test_has_valid_coords(cube, coords, output):
    """Test check for valid coords."""
    out = simple_pp._has_valid_coords(cube, coords)
    assert out == output


X_ARR = np.arange(5)
Y_ARR_1 = np.ma.masked_invalid([np.nan, 1.0, 0.0, np.nan, -0.5])
Y_ARR_2 = np.ma.masked_invalid([np.nan, np.nan, np.nan, np.nan, -0.5])
Y_ARR_2x2 = np.ma.masked_invalid(
    [[2.1 * X_ARR, -3.14 * X_ARR, 0.8 * X_ARR],
     [Y_ARR_1.filled(np.nan),
      Y_ARR_1.filled(2.0),
      Y_ARR_2.filled(np.nan)]])
TEST_GET_SLOPE = [
    (X_ARR, 3.14 * X_ARR, 3.14),
    (X_ARR, Y_ARR_2x2,
     np.array([[2.1, -3.14, 0.8], [-0.46428571428571436, -0.4, np.nan]])),
    (np.arange(1.0), np.arange(1.0), np.nan),
    (X_ARR, np.ma.array([X_ARR, Y_ARR_2]), np.array([1.0, np.nan])),
]


@pytest.mark.parametrize('x_arr,y_arr,output', TEST_GET_SLOPE)
def test_get_slope(x_arr, y_arr, output):
    """Test calculation of slope."""
    out = simple_pp._get_slope(x_arr, y_arr)
    assert ((out == output) | (np.isnan(output) & np.isnan(output))).all()
