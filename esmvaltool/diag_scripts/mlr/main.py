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
metadata_preselection : dict, optional
    Pre-select metadata by specifying (key, value) pairs under the key `select`
    and group input data by an attribute given under `group`. For every group
    element, an individual MLR model is calculated.
model_type : str, optional (default: 'gbr_sklearn')
    MLR model type. The given model has to be defined in
    :mod:`esmvaltool.diag_scripts.mlr.models`.

Additional parameters see :mod:`esmvaltool.diag_scripts.mlr.models`.

"""

import logging
import os
from pprint import pformat

from george import kernels as george_kernels
from sklearn.gaussian_process import kernels as sklearn_kernels

from esmvaltool.diag_scripts.mlr.models import MLRModel
from esmvaltool.diag_scripts.shared import (group_metadata, io, run_diagnostic,
                                            select_metadata)

logger = logging.getLogger(os.path.basename(__file__))


def _get_grouped_datasets(cfg):
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
            logger.warning(
                "No input data found for this diagnostic matching the "
                "specified criteria")
            logger.warning(pformat(preselection))
    else:
        logger.warning("No input data found")
        group = None
        grouped_datasets = {None: None}
    if len(list(grouped_datasets.keys())) == 1 and None in grouped_datasets:
        logger.info("Creating single MLR model")
    return (group, grouped_datasets)


def _update_mlr_model(model_type, mlr_model):
    """Update MLR model paramters during run time."""
    if model_type == 'gpr_george':
        n_features = mlr_model.features_after_preprocessing.size
        new_kernel = (
            george_kernels.ExpSquaredKernel(
                1.0, ndim=n_features, metric_bounds=[(-10.0, 10.0)]) *
            george_kernels.ConstantKernel(
                0.0, ndim=n_features, bounds=[(-10.0, 10.0)])
        )
        mlr_model.update_parameters(final__regressor__kernel=new_kernel)


def run_mlr_model(cfg, model_type):
    """Run all MLR model of desired type on input data."""
    (group, grouped_datasets) = _get_grouped_datasets(cfg)
    for attr in grouped_datasets:
        if attr is not None:
            logger.info("Processing %s", attr)
        metadata = {} if group is None else {group: attr}
        mlr_model = MLRModel.create(model_type, cfg, root_dir=attr, **metadata)

        # Update MLR model parameters dynamically
        _update_mlr_model(model_type, mlr_model)

        # Fit and predict
        if cfg.get('grid_search_cv_param_grid'):
            mlr_model.grid_search_cv()
        else:
            mlr_model.fit()
        mlr_model.predict()

        # Output
        mlr_model.export_training_data()
        mlr_model.export_prediction_data()
        mlr_model.print_correlation_matrices()
        mlr_model.print_regression_metrics()

        # Plots
        mlr_model.plot_pairplots()
        mlr_model.plot_scatterplots()
        for idx in range(10):
            mlr_model.plot_lime(idx)
        if not cfg.get('accept_only_scalar_data'):
            mlr_model.plot_feature_importance()
            # mlr_model.plot_partial_dependences()
        if 'gbr' in model_type:
            mlr_model.plot_gbr_feature_importance()
            mlr_model.plot_prediction_error()
        if 'gpr' in model_type and not cfg.get('accept_only_scalar_data'):
            mlr_model.print_kernel_info()


def set_parameters(cfg, model_type):
    """Update MLR paramters."""
    cfg.setdefault('parameters_final_regressor', {})
    if model_type == 'gpr_sklearn':
        kernel = (sklearn_kernels.ConstantKernel(1.0, (1e-5, 1e5)) *
                  sklearn_kernels.RBF(1.0, (1e-5, 1e5)))
        cfg['parameters_final_regressor']['kernel'] = kernel


def main(cfg):
    """Run the diagnostic."""
    if 'mlr_model' not in cfg:
        default = 'gbr_sklearn'
        logger.warning("'mlr_model' not given in recipe, defaulting to '%s'",
                       default)
        model_type = default
    else:
        model_type = cfg['mlr_model']
    set_parameters(cfg, model_type)
    run_mlr_model(cfg, model_type)


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
