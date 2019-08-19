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
group_metadata : str, optional
    Group input data by an attribute. For every group element (set of
    datasets), an individual MLR model is calculated. Only affects `feature`
    and `label` datasets. Cannot be used together with the option
    `pseudo_reality`.
model_type : str, optional (default: 'gbr_sklearn')
    MLR model type. The given model has to be defined in
    :mod:`esmvaltool.diag_scripts.mlr.models`.
pseudo_reality : list of str, optional
    List of dataset attributes which are used to group input data for a pseudo-
    reality test (also known as 'model-as-truth' or 'perfect-model' setup). In
    this setting, all `prediction_input` specified in the recipe is ignored.
    On the contrary, for every element of the group a single MLR model is
    fitted on all data EXCEPT for that of the specified group element.
    This group element is then used as `prediction_input`. This allows a direct
    assessment of the predictive power of the MLR model by comparing the MLR
    prediction output and the true labels (similar to splitting the input data
    in a training and test set, but not dividing the data randomly but using
    specific datasets, e.g. the different climate models). Cannot be used
    together with the option `group_metadata`.
pattern : str, optional
    Pattern matched against ancestor files.
select_metadata : dict, optional
    Pre-select input data by specifying (key, value) pairs. Affects all
    datasets regardless of `var_type`.

Additional parameters see :mod:`esmvaltool.diag_scripts.mlr.models`.

"""

# TODO: Modify description above!

import logging
import os
import warnings
from copy import deepcopy
from pprint import pformat

from george import kernels as george_kernels
from iris.fileformats.netcdf import UnknownCellMethodWarning
from sklearn.gaussian_process import kernels as sklearn_kernels

from esmvaltool.diag_scripts import mlr
from esmvaltool.diag_scripts.mlr.mmm import main as create_mmm_model
from esmvaltool.diag_scripts.mlr.models import MLRModel
from esmvaltool.diag_scripts.shared import (group_metadata, io, run_diagnostic,
                                            select_metadata)

logger = logging.getLogger(os.path.basename(__file__))

WARNINGS_TO_IGNORE = [
    {
        'message': ".* contains unknown cell method 'trend'",
        'category': UnknownCellMethodWarning,
        'module': 'iris',
    },
]


def _get_grouped_data(cfg, input_data):
    """Group input data to create individual MLR models for each group."""
    group_attribute = cfg['group_metadata']
    logger.info(
        "Grouping training data by atribute '%s' and creating individual MLR "
        "model for each group member", group_attribute)

    # Group data using var types
    var_types = group_metadata(input_data, 'var_type')
    training_data = var_types.get('feature', []) + var_types.get('label', [])
    prediction_data = []
    for pred_type in var_types:
        if 'prediction_' in pred_type:
            prediction_data.extend(var_types[pred_type])

    # Create groups of dataset using training data
    grouped_datasets = group_metadata(training_data, group_attribute)
    grouped_input_data = {}
    for (group_val, datasets) in grouped_datasets.items():
        datasets.extend(prediction_data)
        grouped_input_data[group_val] = datasets
    return (group_attribute, grouped_input_data)


def get_input_data(cfg):
    """Get (grouped) input datasets according to given settings."""
    input_data = get_raw_input_data(cfg)
    if cfg.get('group_metadata'):
        return _get_grouped_data(cfg, input_data)
    if cfg.get('pseudo_reality'):
        return _get_pseudo_reality_data(cfg, input_data)
    logger.info("Creating single MLR model")
    return (None, {None: input_data})


def _get_pseudo_reality_data(cfg, input_data):
    """Get input data groups for pseudo-reality experiment."""
    pseudo_reality_attrs = cfg['pseudo_reality']
    logger.info(
        "Grouping input data for pseudo-reality experiment using attributes "
        "%s", pseudo_reality_attrs)

    # Extract training data
    var_types = group_metadata(input_data, 'var_type')
    training_data = var_types.get('feature', []) + var_types.get('label', [])
    prediction_data = []
    for pred_type in var_types:
        if 'prediction_' in pred_type:
            prediction_data.extend(var_types[pred_type])
            # logger.info("Dropped '%s' datasets", pred_type)
            # logger.debug(pformat([d['filename']
            #                       for d in var_types[pred_type]]))

    # Add aliases and group datasets
    for dataset in training_data:
        dataset['pseudo_reality_group'] = mlr.create_alias(
            dataset, pseudo_reality_attrs)
    grouped_datasets = group_metadata(training_data, 'pseudo_reality_group')
    grouped_input_data = {}
    for (group_val, datasets) in grouped_datasets.items():
        logger.debug("Found pseudo reality group '%s'", group_val)
        pred_datasets = deepcopy(datasets)
        for dataset in pred_datasets:
            if dataset['var_type'] == 'feature':
                dataset['var_type'] = 'prediction_input'
            else:
                dataset['var_type'] = 'prediction_reference'
        remaining_datasets = []
        for data in training_data:
            if data['pseudo_reality_group'] != group_val:
                remaining_datasets.append(deepcopy(data))
        grouped_input_data[group_val] = pred_datasets + remaining_datasets
    return ('pseudo-reality', grouped_input_data)


def get_raw_input_data(cfg):
    """Extract all input datasets."""
    input_data = list(cfg['input_data'].values())
    input_data.extend(io.netcdf_to_metadata(cfg, pattern=cfg.get('pattern')))
    paths = [d['filename'] for d in input_data]
    logger.debug("Found files:")
    logger.debug(pformat(paths))
    select_kwargs = cfg.get('select_metadata', {})
    if select_kwargs:
        logger.info("Only selecting files matching %s", select_kwargs)
        input_data = select_metadata(input_data, **select_kwargs)
        paths = [d['filename'] for d in input_data]
        logger.debug("Remaining files:")
        logger.debug(pformat(paths))
    valid_datasets = []
    for dataset in input_data:
        if mlr.datasets_have_mlr_attributes([dataset], log_level='warning'):
            valid_datasets.append(dataset)
        else:
            logger.warning("Skipping ancestor file %s", dataset['filename'])
    if not input_data:
        logger.warning("No input valid data found")
    return input_data


def _update_mlr_model(model_type, mlr_model):
    """Update MLR model paramters during run time."""
    if model_type == 'gpr_george':
        n_features = mlr_model.features_after_preprocessing.size
        exp_squared_kernel = george_kernels.ExpSquaredKernel(
            1.0,
            ndim=n_features,
            metric_bounds=[(-10.0, 10.0)],
        )
        constant_kernel = george_kernels.ConstantKernel(
            0.0,
            ndim=n_features,
            bounds=[(-10.0, 10.0)],
        )
        new_kernel = exp_squared_kernel * constant_kernel
        mlr_model.update_parameters(final__regressor__kernel=new_kernel)
    elif model_type == 'gpr_sklearn':
        new_kernel = (sklearn_kernels.ConstantKernel(1.0, (1e-5, 1e5)) *
                      sklearn_kernels.RBF(1.0, (1e-5, 1e5)))
        mlr_model.update_parameters(final__regressor__kernel=new_kernel)


def check_cfg(cfg):
    """Check recipe configuration for invalid options."""
    if cfg.get('group_metadata') and cfg.get('pseudo_reality'):
        raise ValueError(
            "The options 'group_metadata' and 'pseudo_reality' cannot be used "
            "together")


def run_mlr_model(cfg, model_type, group_attribute, grouped_datasets):
    """Run MLR model(s) of desired type on input data."""
    for (descr, datasets) in grouped_datasets.items():
        if descr is not None:
            attr = '' if group_attribute is None else f'{group_attribute} '
            logger.info("Creating MLR model '%s' for %s'%s'", model_type, attr,
                        descr)
        mlr_model = MLRModel.create(model_type,
                                    cfg,
                                    input_data=datasets,
                                    root_dir=descr)

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
        # mlr_model.plot_pairplots()
        # mlr_model.plot_scatterplots()
        # if not cfg.get('accept_only_scalar_data'):
        #     mlr_model.plot_feature_importance()
        #     # mlr_model.plot_partial_dependences()
        # if 'gbr' in model_type:
        #     mlr_model.plot_gbr_feature_importance()
        #     mlr_model.plot_prediction_error()
        # if 'gpr' in model_type and not cfg.get('accept_only_scalar_data'):
        #     mlr_model.print_kernel_info()


def run_mmm_model(cfg, group_attribute, grouped_datasets):
    """Run simple MMM model(s) on input data."""
    for (descr, datasets) in grouped_datasets.items():
        if descr is not None:
            attr = '' if group_attribute is None else f'{group_attribute} '
            logger.info("Creating MMM model for %s'%s'", attr, descr)
        create_mmm_model(cfg, input_data=datasets, description=descr)


def main(cfg):
    """Run the diagnostic."""
    check_cfg(cfg)
    if 'mlr_model_type' not in cfg:
        default = 'gbr_sklearn'
        logger.warning(
            "'mlr_model_type' not given in recipe, defaulting to '%s'",
            default)
        model_type = default
    else:
        model_type = cfg['mlr_model_type']
        logger.info("Found mlr_model_type '%s'", model_type)
    (group_attr, grouped_datasets) = get_input_data(cfg)
    if model_type == 'mmm':
        run_mmm_model(cfg, group_attr, grouped_datasets)
    else:
        run_mlr_model(cfg, model_type, group_attr, grouped_datasets)


# Run main function when this script is called
if __name__ == '__main__':
    for warning_kwargs in WARNINGS_TO_IGNORE:
        warning_kwargs.setdefault('action', 'ignore')
        warnings.filterwarnings(**warning_kwargs)
    with run_diagnostic() as config:
        main(config)
