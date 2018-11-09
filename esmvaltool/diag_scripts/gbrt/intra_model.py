#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Main diagnostic for performing GBRT.

Description
-----------
This diagnostic performs the machine learning algorithm "Gradient Boosting
Regression Trees" for climate predictions.

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
import xgboost as xgb

from esmvaltool.diag_scripts.shared import (group_metadata, run_diagnostic,
                                            select_metadata)

logger = logging.getLogger(os.path.basename(__file__))


def main(cfg):
    """Run the diagnostic."""
    input_data = cfg['input_data'].values()

    # Extract datasets and variables
    all_features = select_metadata(input_data, var_type='feature')
    all_labels = select_metadata(input_data, var_type='label')
    all_prediction_input = select_metadata(input_data,
                                           var_type='prediction_input')

    print("features:")
    pprint(all_features)
    print("labels:")
    pprint(all_labels)
    print("predictants:")
    pprint(all_prediction_input)

    # GBRT for every dataset
    for (model_name, features) in group_metadata(
            all_features, 'dataset').items():
        x_data = []
        for feature in features:
            cube = iris.load_cube(feature['filename'])
            x_data.append(cube.data.ravel())
        x_data = np.array(x_data)
        print(model_name)
        print(x_data.shape)
        print(x_data)


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
