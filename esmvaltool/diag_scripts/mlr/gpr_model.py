#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnostic script to create a single GPR model for many climate models.

Description
-----------
This diagnostic creates one "Gaussian Process Regressor" (MLR) model to predict
future climate for several climate models.

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

from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from esmvaltool.diag_scripts.mlr.models import MLRModel
from esmvaltool.diag_scripts.shared import run_diagnostic

logger = logging.getLogger(os.path.basename(__file__))


def main(cfg):
    """Run the diagnostic."""
    model_type = 'gpr'
    mlr_model = MLRModel.create(model_type, cfg)

    # Kernel
    kernel = (ConstantKernel(1.0, (1e-5, 1e5)) * RBF(1.0, (1e-5, 1e5)) +
              WhiteKernel(1e-1, (1e-5, 1e5)))

    # Fit and predict
    mlr_model.fit(kernel=kernel)
    mlr_model.export_training_data()
    predictions = mlr_model.predict()
    logger.info("Predictions:")
    logger.info("%s", pformat(predictions))
    mlr_model.export_prediction_data()

    # Output
    mlr_model.plot_scatterplots()
    mlr_model.print_kernel_info()


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
