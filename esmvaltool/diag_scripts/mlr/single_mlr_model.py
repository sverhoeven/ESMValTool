#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnostic script to create one MLR model for many climate models.

Description
-----------
This diagnostic creates one "Machine Learning Regression" (MLR) model to
predict future climate for several climate models.

Author
------
Manuel Schlund (DLR, Germany)

Project
-------
CRESCENDO

Configuration options in recipe
-------------------------------
See esmvaltool.mlr module.

"""

import logging
import os
from pprint import pformat

from esmvaltool.diag_scripts.mlr import MLRModel
from esmvaltool.diag_scripts.shared import run_diagnostic

logger = logging.getLogger(os.path.basename(__file__))


def main(cfg):
    """Run the diagnostic."""
    mlr_model = MLRModel(cfg)

    # Fit and predict
    mlr_model.simple_train_test_split()
    mlr_model.export_training_data()
    mlr_model.fit()
    predictions = mlr_model.predict()
    logger.info("Predictions:")
    logger.info("%s", pformat(predictions))

    # Plots
    mlr_model.plot_scatterplots()
    mlr_model.plot_feature_importance()
    mlr_model.plot_partial_dependences()
    mlr_model.plot_prediction_error()


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
