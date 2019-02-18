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
See :mod:`esmvaltool.diag_scripts.mlr.models` module.

"""

import logging
import os
from pprint import pformat

from esmvaltool.diag_scripts.mlr.models import MLRModel
from esmvaltool.diag_scripts.shared import run_diagnostic

logger = logging.getLogger(os.path.basename(__file__))


def main(cfg):
    """Run the diagnostic."""
    model_type = cfg.get('mlr_model', 'gbr')
    mlr_model = MLRModel.create(model_type, cfg)

    # Fit and predict
    if cfg.get('grid_search_cv_param_grid'):
        mlr_model.grid_search_cv()
    else:
        # mlr_model.simple_train_test_split()
        mlr_model.fit()
    mlr_model.export_training_data()
    predictions = mlr_model.predict()
    logger.info("Predictions:")
    logger.info("%s", pformat(predictions))
    mlr_model.export_prediction_data()

    # Plots
    mlr_model.plot_scatterplots()
    if model_type in ('gbr' or 'rfr'):
        mlr_model.plot_feature_importance()
    if model_type == 'gbr':
        mlr_model.plot_partial_dependences()
        mlr_model.plot_prediction_error()


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
