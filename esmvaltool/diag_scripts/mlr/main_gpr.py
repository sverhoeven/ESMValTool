#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main Diagnostic script to create a GPR models.

Description
-----------
This diagnostic creates "Gaussian Process Regressor" (GPR) models to predict
the future climate for multiple climate models.

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

import numpy as np
from esmvaltool.diag_scripts.mlr.models import MLRModel
from esmvaltool.diag_scripts.shared import (group_metadata, io, run_diagnostic,
                                            select_metadata)
from george import kernels as george_kernels
from sklearn.gaussian_process import kernels as sklearn_kernels

logger = logging.getLogger(os.path.basename(__file__))


def get_grouped_datasets(cfg):
    """Group input datasets according to given settings."""
    input_data = list(cfg['input_data'].values())
    input_data.extend(io.netcdf_to_metadata(cfg))
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
    cfg.setdefault('parameters', {})
    algorithm = cfg.get('algorithm', 'sklearn')
    if algorithm == 'sklearn':
        model_type = 'sklearn_gpr'
        kernel = (sklearn_kernels.ConstantKernel(
            1.0, (1e-5, 1e5)) * sklearn_kernels.RBF(1.0, (1e-5, 1e5)) +
                  sklearn_kernels.WhiteKernel(1e-1, (1e-10, 1e5)))

    elif algorithm == 'george':
        model_type = 'george_gpr'
        kernel = 1.0 * george_kernels.ExpSquaredKernel(1.0)
        # cfg['parameters']['white_noise'] = np.log(1e-2)
        cfg['parameters']['fit_white_noise'] = True
    else:
        logger.error("Got unknown GPR algorithm '%s'", algorithm)
        return

    # Set Kernel
    cfg['parameters']['kernel'] = kernel

    # Preselect data
    (group, grouped_datasets) = get_grouped_datasets(cfg)
    for attr in grouped_datasets:
        if attr is not None:
            logger.info("Processing %s", attr)
        metadata = {} if group is None else {group: attr}
        mlr_model = MLRModel.create(model_type, cfg, root_dir=attr, **metadata)

        # Kernel (dependent on number of features)
        n_features = mlr_model.classes['features'].size
        mlr_model.update_parameters(
            transformed_target_regressor__regressor__fit_mean=False,
            transformed_target_regressor__transformer__with_std=False,
            transformed_target_regressor__regressor__kernel__k1__log_constant=
            -1.0,
        )
        print(pformat(mlr_model.parameters))

        # Fit and predict
        if cfg.get('grid_search_cv_param_grid'):
            mlr_model.grid_search_cv()
        else:
            mlr_model.simple_train_test_split()
            mlr_model.fit()
        mlr_model.export_training_data()
        mlr_model.predict()
        mlr_model.export_prediction_data()
        mlr_model.print_regression_metrics()

        # Output
        mlr_model.plot_scatterplots()
        if not cfg.get('accept_only_scalar_data'):
            mlr_model.plot_feature_importance()
            mlr_model.plot_partial_dependences()
            mlr_model.print_kernel_info()


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
