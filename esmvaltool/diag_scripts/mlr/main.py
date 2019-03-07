#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main Diagnostic script to create MLR models.

Description
-----------
This diagnostic script creates "Machine Learning Regression" (MLR) models to
predict the future climate for multiple climate models.

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
from esmvaltool.diag_scripts.shared import (group_metadata, run_diagnostic,
                                            select_metadata)

logger = logging.getLogger(os.path.basename(__file__))


def get_grouped_datasets(cfg):
    """Group input datasets according to given settings."""
    input_data = cfg['input_data'].values()
    if input_data:
        preselection = cfg.get('metadata_preselection', {})
        if preselection:
            logger.info("Pre-selecting data using")
            logger.info(pformat(preselection))
        input_data = select_metadata(input_data,
                                     **preselection.get('select', {}))
        group = preselection.get('group')
        grouped_datasets = group_metadata(input_data, group)
        if not grouped_datasets:
            logger.error(
                "No input data found for this diagnostic matching the "
                "specified criteria")
            logger.error(pformat(preselection))
    else:
        group = None
        grouped_datasets = {None: None}
    if len(list(grouped_datasets.keys())) == 1 and None in grouped_datasets:
        logger.info("Creating single MLR model")
    return (group, grouped_datasets)


def main(cfg):
    """Run the diagnostic."""
    model_type = cfg.get('mlr_model', 'gbr')
    (group, grouped_datasets) = get_grouped_datasets(cfg)
    for attr in grouped_datasets:
        if attr is not None:
            logger.info("Processing %s", attr)
        metadata = {} if group is None else {group: attr}
        mlr_model = MLRModel.create(model_type, cfg, root_dir=attr, **metadata)

        # Fit and predict
        if cfg.get('grid_search_cv_param_grid'):
            mlr_model.grid_search_cv()
        else:
            mlr_model.simple_train_test_split()
            mlr_model.fit()
        mlr_model.export_training_data()
        mlr_model.predict()
        mlr_model.export_prediction_data()

        # Output
        mlr_model.plot_scatterplots()
        if not cfg.get('accept_only_scalar_data'):
            mlr_model.plot_feature_importance()
            mlr_model.plot_partial_dependences()
        if model_type == 'gbr':
            mlr_model.plot_prediction_error()
        if model_type == 'gpr':
            mlr_model.print_kernel_info()


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
