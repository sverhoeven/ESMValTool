#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnostic script to create multiple MLR models for many climate models.

Description
-----------
This diagnostic creates multiple "Machine Learning Regression" (MLR) models to
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

from esmvaltool.diag_scripts.mlr.models import MLRModel
from esmvaltool.diag_scripts.shared import (group_metadata, run_diagnostic,
                                            select_metadata)

logger = logging.getLogger(os.path.basename(__file__))


def main(cfg):
    """Run the diagnostic."""
    input_data = cfg['input_data'].values()
    model_type = cfg.get('mlr_model', 'gbr')
    preselection = cfg.get('metadata_preselection', {})
    group = preselection.get('group')
    input_data = select_metadata(input_data, **preselection.get('select', {}))
    grouped_datasets = group_metadata(input_data, group)
    for attr in grouped_datasets:
        logger.info("Processing %s", attr)
        if group is not None:
            metadata = {group: attr}
        else:
            metadata = {}
        mlr_model = MLRModel.create(model_type, cfg, root_dir=attr, **metadata)

        # Fit and predict
        if cfg.get('grid_search_cv_param_grid'):
            mlr_model.grid_search_cv()
        else:
            mlr_model.simple_train_test_split()
            mlr_model.fit()
        mlr_model.export_training_data()
        mlr_model.predict()

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
