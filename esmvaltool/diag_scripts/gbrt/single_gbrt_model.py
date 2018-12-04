#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnostic script create one GBRT models for multiple climate models.

Description
-----------
This diagnostic creates one "Gradient Boosted Regression Trees" (GBRT) model to
predict future climate for several climate models.

Author
------
Manuel Schlund (DLR, Germany)

Project
-------
CRESCENDO

Configuration options in recipe
-------------------------------
see esmvaltool.gbrt module.

"""

import logging
import os
from pprint import pformat

from esmvaltool.diag_scripts.gbrt import GBRTModel
from esmvaltool.diag_scripts.shared import run_diagnostic

logger = logging.getLogger(os.path.basename(__file__))


def main(cfg):
    """Run the diagnostic."""
    gbrt_model = GBRTModel(cfg)

    # Fit and predict
    gbrt_model.fit()
    predictions = gbrt_model.predict()
    logger.info("Predictions:")
    logger.info("%s", pformat(predictions))

    # Plots
    gbrt_model.plot_scatterplot()
    gbrt_model.plot_feature_importance()
    gbrt_model.plot_partial_dependence()
    gbrt_model.plot_prediction_error()


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
