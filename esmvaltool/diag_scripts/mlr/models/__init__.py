"""Base class for MLR models."""

import importlib
import logging
import os
import re
from copy import deepcopy
from functools import partial
from inspect import getfullargspec
from pprint import pformat

import iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathos.multiprocessing as mp
import seaborn as sns
from cf_units import Unit
from skater.core.explanations import Interpretation
from skater.core.local_interpretation.lime.lime_tabular import \
    LimeTabularExplainer
from skater.model import InMemoryModel
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.inspection import plot_partial_dependence
from sklearn.model_selection import (GridSearchCV, LeaveOneOut,
                                     cross_val_score, train_test_split)
from sklearn.preprocessing import StandardScaler

from esmvaltool.diag_scripts import mlr
from esmvaltool.diag_scripts.shared import group_metadata, io, select_metadata

logger = logging.getLogger(os.path.basename(__file__))


class MLRModel():
    """Base class for MLR models.

    Note
    ----
    All datasets must have the attribute `var_type` which specifies this
    dataset. Possible values are `feature` (independent variables used for
    training/testing), `label` (dependent variables, y-axis) or
    `prediction_input` (independent variables used for prediction of dependent
    variables, usually observational data).

    Training data
    -------------
    All groups (specified in `group_datasets_by_attributes`, if desired) given
    for `label` must also be given for the `feature` datasets. Within these
    groups, all `feature` and `label` datasets must have the same shape, except
    the attribute `broadcast_from` is set to a list of suitable coordinate
    indices (must be done for each feature/label).

    Prediction data
    ---------------
    All `tags` specified for `prediction_input` datasets must also be given for
    the `feature` datasets (except `allow_missing_features` is set to `True`).
    Multiple predictions can be specified by `prediction_name`. Within these
    predictions, all `prediction_input` datasets must have the same shape,
    except the attribute `broadcast_from` is given.

    Adding new MLR models
    ---------------------
    MLR models are subclasses of this base class. To add a new one, create a
    new file in :mod:`esmvaltool.diag_scripts.mlr.models` with a child class
    of this class decorated by the method `register_mlr_model`.

    Configuration options in recipe
    -------------------------------
    accept_only_scalar_data : bool, optional (default: False)
        Only accept scalar diagnostic data, if set to True
        'group_datasets_by_attributes should be given.
    allow_missing_features : bool, optional (default: False)
        Allow missing features in the training data.
    cache_intermediate_results : bool, optional (default: True)
        Cache the intermediate results of the pipeline's transformers.
    coords_as_features : list, optional
        If given, specify a list of coordinates which should be used as
        features.
    dtype : str, optional (default: 'float64')
        Internal data type which is used for all calculations, see
        <https://docs.scipy.org/doc/numpy/user/basics.types.html> for a list
        of allowed values.
    estimate_prediction_error : dict, optional
        Estimate (constant) squared prediction error using RMSE. This can be
        calculated by a (holdout) test data set (`type: test`) or by cross-
        validation from the training data (`type: cv'). The latter uses
        :mod:`sklearn.model_selection.cross_val_score` (see
        <https://scikit-learn.org/stable/modules/cross_validation.html>),
        additional keyword arguments can be passed via the `kwargs` key.
    fit_kwargs : dict, optional
        Optional keyword arguments for the pipeline's `fit()` function. These
        arguments have to be given for each step of the pipeline seperated by
        two underscores, i.e. `s__p` is the parameter `p` for step `s`.
    grid_search_cv_kwargs : dict, optional
        Keyword arguments for the grid search cross-validation, see
        <https://scikit-learn.org/stable/modules/generated/
        sklearn.model_selection.GridSearchCV.html>.
    grid_search_cv_param_grid : dict or list of dict, optional
        Parameters (keys) and ranges (values) for exhaustive parameter search
        using cross-validation. Have to be given for each step of the pipeline
        seperated by two underscores, i.e. `s__p` is the parameter `p` for step
        `s`.
    group_datasets_by_attributes : list of str, optional
        List of dataset attributes which are used to group input data for
        `features` and `labels`, e.g. specify `dataset` to use the different
        `dataset`s as observations for the MLR model.
    imputation_strategy : str, optional (default: 'remove')
        Strategy for the imputation of missing values in the features. Must be
        one of `remove`, `mean`, `median`, `most_frequent` or `constant`.
    n_jobs : int, optional (default: 1)
        Maximum number of jobs spawned by this class.
    parameters : dict, optional
        Parameters used for the whole pipeline. Have to be given for each step
        of the pipeline seperated by two underscores, i.e. `s__p` is the
        parameter `p` for step `s`.
    parameters_final_regressor : dict, optional
        Parameters used for the **final** regressor. If these parameters are
        updated using the function `self.update_parameters()`, the new names
        have to be given for each step of the pipeline seperated by two
        underscores, i.e. `s__p` is the parameter `p` for step `s`.
    pca : bool, optional (default: True)
        Preprocess input features using PCA.
    predict_kwargs : dict, optional
        Optional keyword arguments for the regressor's `predict()` function.
    prediction_pp : dict, optional
        Postprocess prediction output (e.g. combine best estimate and standard
        deviation in one cube). Accepts keywords `mean` and `sum` (followed by
        list of coordinates) and `area_weights` (bool, default: True).
    return_lime_importance : bool, optional (default: False)
        Return cube with feature importance given by LIME (Local Interpretable
        Model-agnostic Explanations) during prediction.
    seaborn_settings : dict, optional
        Options for seaborn's `set()` method (affects all plots), see
        <https://seaborn.pydata.org/generated/seaborn.set.html>.
    standardize_data : bool, optional (default: True)
        Linearly standardize input data by removing mean and scaling to unit
        variance.
    test_size : float, optional (default: 0.25)
        If given, exclude the desired fraction of input data from training and
        use it as test data.

    """

    _CLF_TYPE = None
    _MODELS = {}

    @staticmethod
    def _load_mlr_models():
        """Load MLR models from :mod:`esmvaltool.diag_scripts.mlr.models`."""
        current_path = os.path.dirname(os.path.realpath(__file__))
        models_path = os.path.join(current_path)
        for (root, _, model_files) in os.walk(models_path):
            for model_file in model_files:
                rel_path = ('' if root == models_path else os.path.relpath(
                    root, models_path))
                module = os.path.join(rel_path,
                                      os.path.splitext(model_file)[0])
                try:
                    importlib.import_module(
                        'esmvaltool.diag_scripts.mlr.models.{}'.format(
                            module.replace(os.sep, '.')))
                except ImportError:
                    pass

    @classmethod
    def register_mlr_model(cls, model):
        """Add model (subclass of this class) to `_MODEL` dict (decorator)."""
        logger.debug("Found available MLR model '%s'", model)

        def decorator(subclass):
            """Decorate subclass."""
            cls._MODELS[model] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, model, *args, **kwargs):
        """Create desired MLR model subclass (factory method)."""
        cls._load_mlr_models()
        if not cls._MODELS:
            logger.warning("No MLR models found, please add subclasses to "
                           "'esmvaltool.diag_scripts.mlr.models' decorated by "
                           "'MLRModel.register_mlr_model'")
            return cls(*args, **kwargs)
        default_model = list(cls._MODELS.keys())[0]
        if model not in cls._MODELS:
            logger.warning(
                "MLR model '%s' not found in 'esmvaltool."
                "diag_scripts.mlr.models', using default model "
                "'%s'", model, default_model)
            model = default_model
        logger.info("Created MLR model '%s' with final regressor %s", model,
                    cls._MODELS[model]._CLF_TYPE)
        return cls._MODELS[model](*args, **kwargs)

    def __init__(self, cfg, root_dir=None, **metadata):
        """Initialize base class members.

        Parameters
        ----------
        cfg : dict
            Diagnostic script configuration.
        root_dir : str, optional
            Root directory for output (subdirectory in `work_dir` and
            `plot_dir`).
        metadata : keyword arguments
            Metadata for selecting only specific datasets as `features` and
            `labels` (e.g. `dataset='CanESM2'`).

        """
        self._cfg = deepcopy(cfg)
        self._clf = None
        self._data = {}
        self._data['pred'] = {}
        self._datasets = {}
        self._skater = {}
        self._classes = {}
        self._parameters = {}

        # Default settings
        self._cfg.setdefault('cache_intermediate_results', True)
        self._cfg.setdefault('dtype', 'float64')
        self._cfg.setdefault('imputation_strategy', 'remove')
        self._cfg.setdefault('n_jobs', 1)
        self._cfg.setdefault('parameters', {})
        self._cfg.setdefault('pca', True)
        self._cfg.setdefault('prediction_pp', {})
        self._cfg.setdefault('return_lime_importance', False)
        self._cfg.setdefault('standardize_data', True)
        self._cfg.setdefault('test_size', 0.25)
        logger.info("Using imputation strategy '%s'",
                    self._cfg['imputation_strategy'])

        # Seaborn
        sns.set(**self._cfg.get('seaborn_settings', {}))

        # Adapt output directories
        if root_dir is None:
            root_dir = ''
        self._cfg['root_dir'] = root_dir
        self._cfg['mlr_work_dir'] = os.path.join(self._cfg['work_dir'],
                                                 root_dir)
        self._cfg['mlr_plot_dir'] = os.path.join(self._cfg['plot_dir'],
                                                 root_dir)
        if not os.path.exists(self._cfg['mlr_work_dir']):
            os.makedirs(self._cfg['mlr_work_dir'])
            logger.info("Created %s", self._cfg['mlr_work_dir'])
        if not os.path.exists(self._cfg['mlr_plot_dir']):
            os.makedirs(self._cfg['mlr_plot_dir'])
            logger.info("Created %s", self._cfg['mlr_plot_dir'])

        # Load datasets, classes and training data
        self._load_input_datasets(**metadata)
        self._load_classes()
        self._load_data()

        # Create pipeline (with all preprocessor steps and final regressor)
        self._create_pipeline()
        if self._cfg['parameters']:
            logger.debug("Found parameter(s) in recipe: %s",
                         self._cfg['parameters'])
        self.update_parameters(**self._cfg['parameters'])

        # Log successful initialerrorization
        logger.info("Initialized MLR model (using at most %i processes)",
                    self._cfg['n_jobs'])
        logger.debug("With parameters")
        logger.debug(pformat(self.parameters))

    @property
    def data(self):
        """Input data of the model (read-only)."""
        return self._data

    @property
    def features(self):
        """Features of the model (read-only)."""
        return self._classes['features'].index.values

    @property
    def features_types(self):
        """Types of the features of the model (read-only)."""
        return self._classes['features'].types

    @property
    def features_units(self):
        """Units of the features of the model (read-only)."""
        return self._classes['features'].units

    @property
    def group_attributes(self):
        """Group attributes of the model (read-only)."""
        return self._classes['group_attributes']

    @property
    def label(self):
        """Label of the model (read-only)."""
        return self._classes['label'].index.values[0]

    @property
    def label_units(self):
        """Units of the label of the model (read-only)."""
        return self._classes['label'].units.values[0]

    @property
    def parameters(self):
        """Parameters of the final regressor (read-only)."""
        return self._parameters

    def export_prediction_data(self, filename=None):
        """Export all prediction data contained in `self._data`.

        Parameters
        ----------
        filename : str, optional (default: '{data_type}_{pred_name}.csv')
            Name of the exported files.

        """
        for pred_name in self.data['pred']:
            self._save_csv_file('pred', filename, pred_name=pred_name)

    def export_training_data(self, filename=None):
        """Export all training data contained in `self._data`.

        Parameters
        ----------
        filename : str, optional (default: '{data_type}.csv')
            Name of the exported files.

        """
        for data_type in ('all', 'train', 'test'):
            self._save_csv_file(data_type, filename)

    def fit(self, **kwargs):
        """Fit MLR model.

        Parameters
        ----------
        **kwargs : keyword arguments, optional
            Additional options for the `self._clf.fit()` function. Have to be
            given for each step of the pipeline seperated by two underscores,
            i.e. `s__p` is the parameter `p` for step `s`.
            Overwrites default and recipe settings.

        """
        if not self._clf_is_valid(text='Fitting MLR model'):
            return
        logger.info(
            "Fitting MLR model with final regressor %s on %i training "
            "point(s)", self._CLF_TYPE, len(self.data['train'].index))
        fit_kwargs = dict(self._cfg.get('fit_kwargs', {}))
        fit_kwargs.update(kwargs)
        if fit_kwargs:
            logger.info("Using keyword argument(s) %s for fit() function",
                        fit_kwargs)
        fit_kwargs = self._update_fit_kwargs(fit_kwargs)

        # Create MLR model with desired parameters and fit it
        x_train = self.data['train'].x.values
        y_train = self.data['train'].y.squeeze().values
        self._clf.fit(x_train, y_train, **fit_kwargs)
        self._parameters = self._get_clf_parameters()
        logger.info("Successfully fitted MLR model on %i training point(s)",
                    len(self.data['train'].index))
        logger.debug("Pipeline steps:")
        logger.debug(pformat(list(self._clf.named_steps.keys())))
        logger.debug("Parameters:")
        logger.debug(pformat(self.parameters))

        # Interpretation
        self._load_skater_interpreters()

    def grid_search_cv(self, param_grid=None, **kwargs):
        """Perform exhaustive parameter search using cross-validation.

        Parameters
        ----------
        param_grid : dict or list of dict, optional
            Parameter names (keys) and ranges (values) for the search. Have to
            be given for each step of the pipeline seperated by two
            underscores, i.e. `s__p` is the parameter `p` for step `s`.
            Overwrites default and recipe settings.
        **kwargs : keyword arguments, optional
            Additional options for the `GridSearchCV` class. See
            <https://scikit-learn.org/stable/modules/generated/
            sklearn.model_selection.GridSearchCV.html>. Overwrites default and
            recipe settings.

        """
        if not self._clf_is_valid(text='GridSearchCV'):
            return
        parameter_grid = dict(self._cfg.get('grid_search_cv_param_grid', {}))
        if param_grid is not None:
            parameter_grid = param_grid
        if not parameter_grid:
            logger.warning(
                "Cannot perform exhaustive grid search, no parameter grid "
                "given (neither in recipe nor in grid_search_cv() function)")
            return
        logger.info(
            "Performing exhaustive grid search cross-validation with final "
            "regressor %s and parameter grid %s on %i training points",
            self._CLF_TYPE, parameter_grid, len(self.data['train'].index))

        # Get keyword arguments
        verbosity = self._get_verbosity_parameters(GridSearchCV)
        cv_kwargs = {
            'n_jobs': self._cfg['n_jobs'],
            **verbosity,
        }
        cv_kwargs.update(self._cfg.get('grid_search_cv_kwargs', {}))
        cv_kwargs.update(kwargs)
        logger.info("Using keyword argument(s) %s for GridSearchCV class",
                    cv_kwargs)
        if isinstance(cv_kwargs.get('cv'), str):
            if cv_kwargs['cv'].lower() == 'loo':
                cv_kwargs['cv'] = LeaveOneOut()
        fit_kwargs = dict(self._cfg.get('fit_kwargs', {}))
        if fit_kwargs:
            logger.info("Using keyword argument(s) %s for fit() function",
                        fit_kwargs)
        fit_kwargs = self._update_fit_kwargs(fit_kwargs)

        # Create and fit GridSearchCV instance
        clf = GridSearchCV(self._clf, parameter_grid, **cv_kwargs)
        x_train = self.data['train'].x.values
        y_train = self.data['train'].y.squeeze().values
        clf.fit(x_train, y_train, **fit_kwargs)

        # Try to find best estimator
        if hasattr(clf, 'best_estimator_'):
            self._clf = clf.best_estimator_
        elif hasattr(clf, 'best_params_'):
            self.update_parameters(**clf.best_params_)
            self._clf.fit(x_train, y_train, **fit_kwargs)
        else:
            raise ValueError(
                "GridSearchCV not successful, cannot determine best estimator "
                "(neither using 'best_estimator_' nor 'best_params_'), "
                "adapt 'grid_search_cv_kwargs' accordingly (see "
                "<https://scikit-learn.org/stable/modules/generated/"
                "sklearn.model_selection.GridSearchCV.html> for help)")
        self._parameters = self._get_clf_parameters()
        logger.info(
            "Exhaustive grid search successful, found best parameter(s) %s",
            clf.best_params_)
        logger.debug("CV results:")
        logger.debug(pformat(clf.cv_results_))
        logger.info("Successfully fitted MLR model on %i training point(s)",
                    len(self.data['train'].index))
        logger.debug("Pipeline steps:")
        logger.debug(pformat(list(self._clf.named_steps.keys())))
        logger.debug("Parameters:")
        logger.debug(pformat(self.parameters))

        # Interpretation
        self._load_skater_interpreters()

    def plot_feature_importance(self, filename=None):
        """Plot feature importance.

        Parameters
        ----------
        filename : str, optional (default: 'feature_importance_{method}')
            Name of the plot file.

        """
        if not self._is_ready_for_plotting():
            return
        logger.info("Plotting feature importance")
        if filename is None:
            filename = 'feature_importance_{method}'
        progressbar = True if self._cfg['log_level'] == 'debug' else False

        # Plot
        for method in ('model-scoring', 'prediction-variance'):
            logger.debug("Plotting feature importance for method '%s'", method)
            (_, axes) = (self._skater['global_interpreter'].feature_importance.
                         plot_feature_importance(self._skater['model'],
                                                 method=method,
                                                 n_jobs=self._cfg['n_jobs'],
                                                 progressbar=progressbar))
            axes.set_title('Variable Importance ({} Model)'.format(
                self._CLF_TYPE))
            axes.set_xlabel('Relative Importance')
            new_filename = (filename.format(method=method) + '.' +
                            self._cfg['output_file_type'])
            new_path = os.path.join(self._cfg['mlr_plot_dir'], new_filename)
            plt.savefig(new_path, orientation='landscape', bbox_inches='tight')
            logger.info("Wrote %s", new_path)
            plt.close()

    def plot_lime(self, index=0, data_type='test', filename=None):
        """Plot LIME explanations for specific input.

        Note
        ----
        LIME = Local Interpretable Model-agnostic Explanations.

        Parameters
        ----------
        filename : str, optional (default: 'lime')
            Name of the plot file.

        """
        if not self._is_ready_for_plotting():
            return
        logger.info("Plotting LIME")
        if data_type not in self.data:
            logger.warning("Cannot plot LIME, got invalid data type '%s'",
                           data_type)
            return
        if index >= len(self.data[data_type].index):
            logger.warning(
                "Cannot plot LIME, index %i is out of range for '%s' data",
                index, data_type)
            return
        if filename is None:
            filename = 'lime'
        new_filename_plot = filename + '.' + self._cfg['output_file_type']
        new_filename_html = filename + '.html'
        plot_path = os.path.join(self._cfg['mlr_plot_dir'], new_filename_plot)
        html_path = os.path.join(self._cfg['mlr_plot_dir'], new_filename_html)

        # LIME
        explainer = self._skater['local_interpreter'].explain_instance(
            self.data[data_type].x.loc[index].values, self._clf.predict)
        logger.debug("Local feature importance at index %i of '%s' data",
                     index, data_type)
        logger.debug(pformat(explainer.as_list()))

        # Html
        pred_dtype = self._get_prediction_dtype()
        if pred_dtype == 'float64':
            explainer.save_to_file(html_path)
            logger.info("Wrote %s", html_path)
        else:
            logger.warning(
                "Saving LIME output in HTML format is only supported for "
                "regressors which save predictions as dtype 'float64', "
                "%s writes '%s'", self._CLF_TYPE, pred_dtype)

        # Plot
        explainer.as_pyplot_figure()
        plt.savefig(plot_path, orientation='landscape', bbox_inches='tight')
        logger.info("Wrote %s", plot_path)
        plt.close()

    def plot_pairplots(self, filename=None):
        """Plot pairplots for features and labels.

        Parameters
        ----------
        filename : str, optional (default: 'pairplot_{data_type}')
            Name of the plot file.

        """
        if not self._is_ready_for_plotting():
            return
        logger.info("Plotting pairplots")
        if filename is None:
            filename = 'pairplot_{data_type}'

        # Plot pairplots for all data types
        for data_type in ('all', 'train', 'test'):
            if data_type not in self.data:
                continue
            data_frame = self._impute_nans(self.data[data_type])
            sns.pairplot(data_frame)
            new_filename = (filename.format(data_type=data_type) + '.' +
                            self._cfg['output_file_type'])
            new_path = os.path.join(self._cfg['mlr_plot_dir'], new_filename)
            plt.savefig(new_path, orientation='landscape', bbox_inches='tight')
            logger.info("Wrote %s", new_path)
            plt.close()

    def plot_partial_dependences(self, filename=None):
        """Plot partial dependences for every feature.

        Parameters
        ----------
        filename : str, optional (default: 'partial_dependece_{feature}')
            Name of the plot file.

        """
        if not self._is_ready_for_plotting():
            return
        logger.info("Plotting partial dependences")
        if filename is None:
            filename = 'partial_dependece_{feature}'

        # Plot for every feature
        x_train = self._impute_nans(self.data['train']).x.values
        verbosity = self._get_verbosity_parameters(plot_partial_dependence)
        for feature_name in self.features:
            logger.debug("Plotting partial dependence of '%s'", feature_name)
            plot_partial_dependence(
                self._clf,
                x_train,
                features=[feature_name],
                feature_names=self.features,
                n_jobs=self._cfg['n_jobs'],
                **verbosity,
            )
            plt.title('Partial dependence ({} Model)'.format(self._CLF_TYPE))
            plt.xlabel(f'{feature_name} / {self.features_units[feature_name]}')
            plt.ylabel(f'Partial dependence on {self.label}')
            new_filename = (filename.format(feature=feature_name) + '.' +
                            self._cfg['output_file_type'])
            new_path = os.path.join(self._cfg['mlr_plot_dir'], new_filename)
            plt.savefig(new_path, orientation='landscape', bbox_inches='tight')
            logger.info("Wrote %s", new_path)
            plt.close()

    def plot_scatterplots(self, filename=None):
        """Plot scatterplots label vs. feature for every feature.

        Parameters
        ----------
        filename : str, optional (default: 'scatterplot_{feature}')
            Name of the plot file.

        """
        if not self._is_ready_for_plotting():
            return
        logger.info("Plotting scatterplots")
        if filename is None:
            filename = 'scatterplot_{feature}'

        # Plot scatterplot for every feature
        for feature in self.features:
            logger.debug("Plotting scatterplot of '%s'", feature)
            (_, axes) = plt.subplots()
            if self._cfg.get('accept_only_scalar_data'):
                for (g_idx, group_attr) in enumerate(self.group_attributes):
                    axes.scatter(self.data['all'].x.loc[g_idx, feature],
                                 self.data['all'].y.iloc[g_idx, 0],
                                 label=group_attr)
                for (pred_name, pred) in self.data['pred'].items():
                    axes.axvline(pred.x.loc[0, feature],
                                 linestyle='--',
                                 color='black',
                                 label=('Observation'
                                        if pred_name is None else pred_name))
                legend = axes.legend(loc='center left',
                                     ncol=2,
                                     bbox_to_anchor=[1.05, 0.5],
                                     borderaxespad=0.0)
            else:
                axes.plot(self.data['all'].x.loc[:, feature],
                          self.data['all'].y.squeeze().values, '.')
                legend = None
            axes.set_title(feature)
            axes.set_xlabel('{} / {}'.format(feature,
                                             self.features_units[feature]))
            axes.set_ylabel('{} / {}'.format(self.label, self.label_units))
            new_path = os.path.join(
                self._cfg['mlr_plot_dir'],
                filename.format(feature=feature) + '.' +
                self._cfg['output_file_type'])
            plt.savefig(new_path,
                        orientation='landscape',
                        bbox_inches='tight',
                        additional_artists=[legend])
            logger.info("Wrote %s", new_path)
            plt.close()

    def predict(self, **kwargs):
        """Perform prediction using the MLR model(s) and write netcdf.

        Parameters
        ----------
        **kwargs : keyword arguments, optional
            Additional options for the `self._clf.predict()` function.
            Overwrites default and recipe settings.

        """
        if not self._is_fitted():
            logger.warning(
                "Prediction not possible, MLR model is not fitted yet")
            return
        logger.info("Started prediction")
        predict_kwargs = dict(self._cfg.get('predict_kwargs', {}))
        predict_kwargs.update(kwargs)
        if predict_kwargs:
            logger.info(
                "Using additional keyword argument(s) %s for predict() "
                "function", predict_kwargs)

        # Iterate over predictions
        if not self._datasets['prediction']:
            logger.warning("Prediction not possible, no 'prediction_input' "
                           "datasets given")
            return
        for pred_name in self._datasets['prediction']:
            if pred_name is not None:
                logger.info("Predicting '%s'", pred_name)

            # Prediction
            (x_pred, x_cube) = self._extract_prediction_input(pred_name)
            pred_dict = self._get_prediction_dict(x_pred, **predict_kwargs)
            pred_dict = self._estimate_prediction_error(pred_dict)

            # Save data
            y_pred = pd.DataFrame(pred_dict[None],
                                  columns=[self.label],
                                  dtype=self._cfg['dtype'])
            self._data['pred'][pred_name] = pd.concat([x_pred, y_pred],
                                                      axis=1,
                                                      keys=['x', 'y'])

            # Get (and save) prediction cubes
            (predictions,
             pred_types) = self._get_prediction_cubes(pred_dict, pred_name,
                                                      x_cube)

            # Postprocess prediction cubes (if desired)
            self._postprocess_predictions(predictions, pred_types,
                                          **predict_kwargs)

    def print_correlation_matrices(self):
        """Print correlation matrices for all datasets."""
        if not self._is_fitted():
            logger.warning(
                "Printing correlation matrices not possible, MLR model is not "
                "fitted yet")
            return
        for data_type in ('all', 'train', 'test'):
            if data_type not in self.data:
                continue
            logger.info("Correlation matrix for %s data:", data_type)
            logger.info(self.data[data_type].corr())

    def print_regression_metrics(self):
        """Print all available regression metrics for the test data."""
        if not self._is_fitted():
            logger.warning(
                "Printing regression metrics not possible, MLR model is not "
                "fitted yet")
            return
        regression_metrics = [
            'explained_variance_score',
            'mean_absolute_error',
            'mean_squared_error',
            'median_absolute_error',
            'r2_score',
        ]
        for data_type in ('all', 'train', 'test'):
            if data_type not in self.data:
                continue
            logger.info("Evaluating regression metrics for %s data", data_type)
            x_data = self.data[data_type].x.values
            y_true = self.data[data_type].y.squeeze().values
            y_pred = self._clf.predict(x_data)
            y_norm = np.std(y_true)
            for metric in regression_metrics:
                metric_function = getattr(metrics, metric)
                value = metric_function(y_true, y_pred)
                if 'squared' in metric:
                    value = np.sqrt(value)
                    metric = f'root_{metric}'
                if metric.endswith('_error'):
                    value /= y_norm
                    metric = f'{metric} (normalized by std)'
                logger.info("%s: %s", metric, value)

    def update_parameters(self, **params):
        """Update parameters of the whole pipeline.

        Parameters
        ----------
        **params : keyword arguments, optional
            Paramaters for the pipeline which should be updated.

        Note
        ----
        Parameter names have to be given for each step of the pipeline
        seperated by two underscores, i.e. `s__p` is the parameter `p` for
        step `s`.

        """
        if not self._clf_is_valid(text='Updating parameters of MLR model'):
            return
        allowed_params = self._get_clf_parameters()
        new_params = {}
        for (key, val) in params.items():
            if key in allowed_params:
                new_params[key] = val
            else:
                logger.warning(
                    "'%s' is not a valid parameter for the pipeline", key)
        self._clf.set_params(**new_params)
        self._parameters = self._get_clf_parameters()
        if new_params:
            logger.info("Updated pipeline with parameters %s", new_params)

    def _check_cube_coords(self, cube, expected_coords, text=None):
        """Check shape and coordinates of a given cube."""
        msg = '' if text is None else ' for {}'.format(text)
        if self._cfg.get('accept_only_scalar_data'):
            allowed_shapes = [(), (1, )]
            if cube.shape not in allowed_shapes:
                raise ValueError(
                    "Expected only cubes with shapes {} when option 'accept_"
                    "only_scalar_data' is set to 'True', got {}{}".format(
                        allowed_shapes, cube.shape, msg))
        else:
            if expected_coords is not None:
                cube_coords = cube.coords(dim_coords=True)
                cube_coords_str = [
                    '{}, shape {}'.format(coord.name(), coord.shape)
                    for coord in cube_coords
                ]
                expected_coords_str = [
                    '{}, shape {}'.format(coord.name(), coord.shape)
                    for coord in expected_coords
                ]
                if cube_coords_str != expected_coords_str:
                    raise ValueError(
                        "Expected field with coordinates {}{}, got {}. "
                        "Consider regridding, pre-selecting data at class "
                        "initialization using '**metadata' or the options "
                        "'broadcast_from' or 'group_datasets_by_attributes'".
                        format(expected_coords_str, msg, cube_coords_str))
                for (idx, cube_coord) in enumerate(cube_coords):
                    expected_coord = expected_coords[idx]
                    if not np.allclose(cube_coord.points,
                                       expected_coord.points):
                        logger.warning(
                            "'%s' coordinate for different cubes does not "
                            "match, got %s%s, expected %s (values differ by "
                            "more than allowed tolerance, check input cubes)",
                            cube_coord.name(), cube_coord.points, msg,
                            expected_coord.points)

    def _check_dataset(self, datasets, var_type, tag, text=None):
        """Check if datasets exist and are valid."""
        datasets = select_metadata(datasets, tag=tag)
        msg = '' if text is None else text
        if not datasets:
            if var_type == 'label':
                raise ValueError("Label '{}'{} not found".format(tag, msg))
            if not self._cfg.get('allow_missing_features'):
                raise ValueError(
                    "{} '{}'{} not found, use 'allow_missing_features' to "
                    "ignore this".format(var_type, tag, msg))
            logger.info(
                "Ignored missing %s '%s'%s since 'allow_missing_features' is "
                "set to 'True'", var_type, tag, msg)
            return None
        if len(datasets) > 1:
            raise ValueError(
                "{} '{}'{} not unique, consider the use if '**metadata' in "
                "class initialization to pre-select datasets of specify "
                "suitable attributes to group datasets with the option "
                "'group_datasets_by_attributes'".format(var_type, tag, msg))
        if var_type == 'label':
            units = self.label_units
        else:
            units = self.features_units[tag]
        if units != Unit(datasets[0]['units']):
            raise ValueError(
                "Expected units '{}' for {} '{}'{}, got '{}'".format(
                    units, var_type, tag, msg, datasets[0]['units']))
        return datasets[0]

    def _clf_is_valid(self, text=None):
        """Check if valid regressor type is given."""
        msg = '' if text is None else '{} not possible: '.format(text)
        if self._CLF_TYPE is None:
            logger.warning(
                "%sNo MLR model specified, please use factory function "
                "'MLRModel.create()' to initialize this class or populate the "
                "module 'esmvaltool.diag_scripts.mlr.models' if necessary",
                msg)
            return False
        return True

    def _convert_units_in_cube(self, cube, new_units, power=None, text=None):
        """Convert units of cube if possible."""
        msg = '' if text is None else f' of {text}'
        if isinstance(new_units, str):
            new_units = Unit(new_units)
        if power:
            logger.debug("Raising target units of cube '%s' by power of %i",
                         cube.summary(shorten=True), power)
            new_units = self._units_power(new_units, power)
        new_units_name = (new_units.symbol
                          if new_units.origin is None else new_units.origin)
        logger.debug("Converting units%s from '%s' to '%s'", msg,
                     cube.units.symbol, new_units_name)
        try:
            cube.convert_units(new_units)
        except ValueError:
            logger.warning("Units conversion%s from '%s' to '%s' failed", msg,
                           cube.units.symbol, new_units_name)

    def _create_pipeline(self):
        """Create pipeline with correct settings."""
        if not self._clf_is_valid(text='Creating pipeline'):
            return
        steps = []

        # Imputer
        if self._cfg['imputation_strategy'] != 'remove':
            verbosity = self._get_verbosity_parameters(SimpleImputer)
            imputer = SimpleImputer(
                strategy=self._cfg['imputation_strategy'],
                **verbosity,
            )
            steps.append(('imputer', imputer))

        # Scaler
        scale_data = self._cfg['standardize_data']
        x_scaler = StandardScaler(with_mean=scale_data, with_std=scale_data)
        y_scaler = StandardScaler(with_mean=scale_data, with_std=scale_data)
        steps.append(('x_scaler', x_scaler))

        # PCA
        if self._cfg['pca']:
            pca = PCA()
            steps.append(('pca', pca))

        # Regressor
        final_parameters = self._load_final_parameters()
        regressor = self._CLF_TYPE(**final_parameters)
        transformed_regressor = mlr.AdvancedTransformedTargetRegressor(
            transformer=y_scaler, regressor=regressor)
        steps.append(('transformed_target_regressor', transformed_regressor))

        # Final pipeline
        if self._cfg['cache_intermediate_results']:
            if self._cfg['n_jobs'] is None or self._cfg['n_jobs'] == 1:
                memory = self._cfg['mlr_work_dir']
            else:
                logger.debug(
                    "Caching intermediate results of Pipeline is not "
                    "supported for multiple processes (using at most %i "
                    "processes)", self._cfg['n_jobs'])
                memory = None
        else:
            memory = None
        self._clf = mlr.AdvancedPipeline(steps, memory=memory)
        logger.debug("Created pipeline with steps %s",
                     list(self._clf.named_steps.keys()))

    def _estimate_prediction_error(self, pred_dict):
        """Estimate squared prediction error."""
        if not self._cfg.get('estimate_prediction_error'):
            return pred_dict
        cfg = deepcopy(self._cfg['estimate_prediction_error'])
        cfg.setdefault('type', 'cv')
        cfg.setdefault('kwargs', {})
        cfg['kwargs'].setdefault('cv', 5)
        cfg['kwargs'].setdefault('scoring', 'neg_mean_squared_error')
        logger.debug("Estimating squared prediction error using %s", cfg)
        error = None

        # Test data set
        if cfg['type'] == 'test':
            if 'test' in self.data:
                y_pred = self._clf.predict(self.data['test'].x.values)
                error = metrics.mean_squared_error(self.data['test'].y.values,
                                                   y_pred)
            else:
                logger.warning(
                    "Cannot estimate squared prediction error using 'type: "
                    "test', no test data set given (use 'test_size' option), "
                    "using cross-validation instead")
                cfg['type'] = 'cv'

        # CV
        if cfg['type'] == 'cv':
            error = cross_val_score(self._clf, self.data['train'].x.values,
                                    self.data['train'].y.values,
                                    **cfg['kwargs'])
            error = np.mean(error)
            if cfg['kwargs']['scoring'].startswith('neg_'):
                error = -error
            if 'squared' not in cfg['kwargs']['scoring']:
                error *= error

        # Get correct shape and mask
        if error is None:
            logger.warning(
                "Got invalid type for prediction error estimation, got '%s', "
                "expected 'test' or 'cv'", cfg['type'])
            return pred_dict
        units = self._units_power(self.label_units, 2)
        pred_error = np.where(np.isnan(pred_dict[None]), np.nan, error)
        pred_dict[f"squared_error_estim_{cfg['type']}"] = pred_error
        logger.info(
            "Estimated squared prediction error by %s %s using %s data", error,
            units.symbol if units.origin is None else units.origin,
            cfg['type'])
        return pred_dict

    def _extract_features_and_labels(self):
        """Extract feature and label data points from training data."""
        datasets = self._datasets['training']
        (x_data, _) = self._extract_x_data(datasets, 'feature')
        y_data = self._extract_y_data(datasets)

        # Check number of input points
        if len(x_data.index) != len(y_data.index):
            raise ValueError(
                "Sizes of features and labels do not match, got {:d} point(s) "
                "for the features and {:d} point(s) for the label".format(
                    len(x_data.index), len(y_data.index)))
        logger.info("Found %i raw input data point(s) with data type '%s'",
                    len(y_data.index), self._cfg['dtype'])

        # Remove missing values in labels
        (x_data, y_data) = self._remove_missing_labels(x_data, y_data)

        # Remove missing values in features (if desired)
        (x_data, y_data) = self._remove_missing_features(x_data, y_data)

        return (x_data, y_data)

    def _extract_prediction_input(self, prediction_name):
        """Extract prediction input data points for `prediction_name`."""
        datasets = self._datasets['prediction'][prediction_name]
        (x_data,
         prediction_input_cube) = self._extract_x_data(datasets,
                                                       'prediction_input')
        logger.info(
            "Found %i raw prediction input data point(s) with data type '%s'",
            len(x_data.index), self._cfg['dtype'])
        return (x_data, prediction_input_cube)

    def _extract_x_data(self, datasets, var_type):
        """Extract required x data of type `var_type` from `datasets`."""
        allowed_types = ('feature', 'prediction_input')
        if var_type not in allowed_types:
            raise ValueError("Excepted one of '{}' for 'var_type', got "
                             "'{}'".format(allowed_types, var_type))

        # Collect data from datasets and return it
        datasets = select_metadata(datasets, var_type=var_type)
        x_data = pd.DataFrame(columns=self.features, dtype=self._cfg['dtype'])
        cube = None

        # Iterate over datasets
        if var_type == 'feature':
            groups = self.group_attributes
        else:
            groups = [None]
        for group_attr in groups:
            group_datasets = select_metadata(datasets,
                                             group_attribute=group_attr)
            if group_attr is not None:
                logger.info("Loading '%s' data of '%s'", var_type, group_attr)
            msg = '' if group_attr is None else " for '{}'".format(group_attr)
            if not group_datasets:
                raise ValueError("No '{}' data{} found".format(var_type, msg))
            (group_data,
             cube) = self._get_x_data_for_group(group_datasets, var_type,
                                                group_attr)
            x_data = x_data.append(group_data, ignore_index=True)

        return (x_data, cube)

    def _extract_y_data(self, datasets):
        """Extract y data (labels) from `datasets`."""
        datasets = select_metadata(datasets, var_type='label')
        y_data = pd.DataFrame(columns=[self.label], dtype=self._cfg['dtype'])
        for group_attr in self.group_attributes:
            if group_attr is not None:
                logger.info("Loading 'label' data of '%s'", group_attr)
            msg = '' if group_attr is None else " for '{}'".format(group_attr)
            datasets_ = select_metadata(datasets, group_attribute=group_attr)
            dataset = self._check_dataset(datasets_, 'label', self.label, msg)
            cube = self._load_cube(dataset)
            text = "label '{}'{}".format(self.label, msg)
            self._check_cube_coords(cube, None, text)
            cube_data = pd.DataFrame(self._get_cube_data(cube),
                                     columns=[self.label],
                                     dtype=self._cfg['dtype'])
            y_data = y_data.append(cube_data, ignore_index=True)
        return y_data

    def _get_ancestor_datasets(self):
        """Get ancestor datasets."""
        datasets = io.netcdf_to_metadata(self._cfg)
        if not datasets:
            logger.debug("Skipping loading ancestor datasets, no files found")
            return []
        logger.debug("Found ancestor file(s):")
        logger.debug(pformat([d['filename'] for d in datasets]))

        # Check MLR attributes
        valid_datasets = []
        for dataset in datasets:
            if mlr.datasets_have_mlr_attributes([dataset]):
                valid_datasets.append(dataset)
            else:
                logger.debug("Skipping %s", dataset['filename'])
        return valid_datasets

    def _get_area_weights(self, cube, pred_type=None):
        """Get area weights for a cube."""
        logger.debug("Calculating area weights")
        area_weights = None
        for coord in cube.coords(dim_coords=True):
            if not coord.has_bounds():
                coord.guess_bounds()
        try:
            area_weights = iris.analysis.cartography.area_weights(cube)
        except ValueError as exc:
            logger.warning(
                "Calculation of area weights for prediction cube '%s' failed",
                cube.summary(shorten=True))
            logger.warning(str(exc))
        power = self._pred_type_to_power(pred_type)
        area_weights = area_weights**power
        return area_weights

    def _get_broadcasted_cube(self, dataset, ref_cube, text=None):
        """Get broadcasted cube."""
        msg = '' if text is None else text
        target_shape = ref_cube.shape
        cube_to_broadcast = self._load_cube(dataset)
        data_to_broadcast = np.ma.filled(cube_to_broadcast.data, np.nan)
        try:
            new_axis_pos = np.delete(np.arange(len(target_shape)),
                                     dataset['broadcast_from'])
        except IndexError:
            raise IndexError(
                "Broadcasting to shape {} failed{}, index out of bounds".
                format(target_shape, msg))
        logger.info("Broadcasting %s from %s to %s", msg,
                    data_to_broadcast.shape, target_shape)
        for idx in new_axis_pos:
            data_to_broadcast = np.expand_dims(data_to_broadcast, idx)
        data_to_broadcast = np.broadcast_to(data_to_broadcast, target_shape)
        new_cube = ref_cube.copy(np.ma.masked_invalid(data_to_broadcast))
        for idx in dataset['broadcast_from']:
            new_coord = new_cube.coord(dimensions=idx)
            new_coord.points = cube_to_broadcast.coord(new_coord).points
        logger.debug("Added broadcasted %s", msg)
        return new_cube

    def _get_clf_parameters(self, deep=True):
        """Get parameters of pipeline."""
        return self._clf.get_params(deep=deep)

    def _get_features(self):
        """Extract all features from the `prediction_input` datasets."""
        logger.debug("Extracting features from 'prediction_input' datasets")
        pred_name = list(self._datasets['prediction'].keys())[0]
        datasets = self._datasets['prediction'][pred_name]
        msg = ('' if pred_name is None else
               " for prediction '{}'".format(pred_name))
        (units,
         types) = self._get_features_of_datasets(datasets, 'prediction_input',
                                                 msg)

        # Check if features were found
        if not units:
            raise ValueError(
                "No features for 'prediction_input' data{} found".format(msg))

        # Check for wrong options
        if self._cfg.get('accept_only_scalar_data'):
            if 'broadcasted' in types.values():
                raise TypeError(
                    "The use of 'broadcast_from' is not possible if "
                    "'accept_only_scalar_data' is given")
            if 'coordinate' in types.values():
                raise TypeError(
                    "The use of 'coords_as_features' is not possible if "
                    "'accept_only_scalar_data' is given")

        # Convert to DataFrame and sort it
        units = pd.DataFrame.from_dict(units,
                                       orient='index',
                                       columns=['units'])
        types = pd.DataFrame.from_dict(types,
                                       orient='index',
                                       columns=['types'])
        features = pd.concat([units, types], axis=1).sort_index()

        # Return features
        logger.info(
            "Found %i feature(s) (defined in 'prediction_input' data%s)",
            len(features.index), msg)
        for feature in features.index:
            logger.debug("'%s' with units '%s' and type '%s'", feature,
                         features.units.loc[feature],
                         features.types.loc[feature])
        return features

    def _get_features_of_datasets(self, datasets, var_type, msg):
        """Extract all features (with units and types) of given datasets."""
        units = {}
        types = {}
        cube = None
        ref_cube = None
        for (tag, datasets_) in group_metadata(datasets, 'tag').items():
            dataset = datasets_[0]
            cube = self._load_cube(dataset)
            if 'broadcast_from' not in dataset:

                ref_cube = cube
            units[tag] = Unit(dataset['units'])
            if 'broadcast_from' in dataset:
                types[tag] = 'broadcasted'
            else:
                types[tag] = 'regular'

        # Check if reference cube was given
        if ref_cube is None:
            if cube is None:
                raise ValueError("Expected at least one '{}' dataset{}".format(
                    var_type, msg))
            else:
                raise ValueError(
                    "Expected at least one '{}' dataset{} without the option "
                    "'broadcast_from'".format(var_type, msg))

        # Coordinate features
        for coord_name in self._cfg.get('coords_as_features', []):
            try:
                coord = ref_cube.coord(coord_name)
            except iris.exceptions.CoordinateNotFoundError:
                raise iris.exceptions.CoordinateNotFoundError(
                    "Coordinate '{}' given in 'coords_as_features' not found "
                    "in '{}' data{}".format(coord_name, var_type, msg))
            units[coord_name] = coord.units
            types[coord_name] = 'coordinate'

        return (units, types)

    def _get_group_attributes(self):
        """Get all group attributes from `label` datasets."""
        logger.debug("Extracting group attributes from 'label' datasets")
        datasets = select_metadata(self._datasets['training'],
                                   var_type='label')
        grouped_datasets = group_metadata(datasets,
                                          'group_attribute',
                                          sort=True)
        group_attributes = list(grouped_datasets.keys())
        if group_attributes == [None]:
            logger.debug("No group attributes given")
        else:
            logger.info(
                "Found %i group attribute(s) (defined in 'label' data)",
                len(group_attributes))
            logger.debug(pformat(group_attributes))
        return np.array(group_attributes)

    def _get_label(self):
        """Extract label from training data."""
        logger.debug("Extracting label from training datasets")
        datasets = select_metadata(self._datasets['training'],
                                   var_type='label')
        if not datasets:
            raise ValueError("No 'label' datasets given")
        grouped_datasets = group_metadata(datasets, 'tag')
        labels = list(grouped_datasets.keys())
        if len(labels) > 1:
            raise ValueError(
                "Expected unique label tag, got {}".format(labels))
        units = Unit(datasets[0]['units'])
        logger.info(
            "Found label '%s' with units '%s' (defined in 'label' "
            "data)", labels[0], units)
        label = pd.DataFrame.from_dict({labels[0]: units},
                                       orient='index',
                                       columns=['units'])
        return label

    def _get_lime_feature_importance(self, x_data):
        """Get most important feature given by LIME."""
        logger.info(
            "Calculating global feature importance using LIME (this may take "
            "a while...)")

        # Most important feature for single input
        def _most_important_feature(x_input, self):
            """Get most important feature for single input."""
            explainer = self._skater['local_interpreter'].explain_instance(
                x_input, self._clf.predict, num_features=1, num_samples=200)
            return explainer.as_map()[1][0][0]

        _most_important_feature = partial(_most_important_feature, self=self)

        # Apply on whole input (using multiple processes)
        pool = mp.ProcessPool(processes=self._cfg['n_jobs'])
        return np.array(pool.map(_most_important_feature, x_data))

    def _get_prediction_dict(self, x_data, **kwargs):
        """Get prediction output in a dictionary."""
        mask = x_data.isnull().any(axis=1).values
        if self._cfg['imputation_strategy'] == 'remove':
            x_data = x_data[~mask].reset_index(drop=True)
            diff = mask.shape[0] - len(x_data.index)
            if diff:
                logger.info(
                    "Removed %i prediction input point(s) where "
                    "features were missing'", diff)

        # Get prediction dictionary
        logger.info("Predicting %i point(s)", len(x_data.index))
        y_preds = self._clf.predict(x_data.values, **kwargs)
        pred_dict = self._prediction_to_dict(y_preds, **kwargs)

        # LIME feature importance
        if self._cfg['return_lime_importance']:
            pred_dict['lime'] = self._get_lime_feature_importance(
                x_data.values)

        # Transform arrays correctly
        for (pred_type, y_pred) in pred_dict.items():
            if y_pred.ndim == 1 and y_pred.shape[0] != mask.shape[0]:
                new_y_pred = np.empty(mask.shape[0], dtype=self._cfg['dtype'])
                new_y_pred[mask] = np.nan
                new_y_pred[~mask] = y_pred
            else:
                new_y_pred = y_pred
            pred_dict[pred_type] = new_y_pred
            if pred_type is not None:
                logger.debug("Found additional prediction type '%s'",
                             pred_type)
        logger.info("Successfully created prediction array with %i point(s)",
                    pred_dict[None].size)
        return pred_dict

    def _get_prediction_cubes(self, pred_dict, pred_name, x_cube):
        """Get (multi-dimensional) prediction output."""
        logger.debug("Creating output cubes")
        prediction_cubes = {}
        prediction_types = {}
        for (pred_type, y_pred) in pred_dict.items():
            if y_pred.size == np.prod(x_cube.shape):
                y_pred = y_pred.reshape(x_cube.shape)
                if (self._cfg['imputation_strategy'] == 'remove'
                        and np.ma.is_masked(x_cube.data)):
                    y_pred[x_cube.data.mask] = np.nan
                pred_cube = x_cube.copy(np.ma.masked_invalid(y_pred))
            else:
                dim_coords = []
                for (dim_idx, dim_size) in enumerate(y_pred.shape):
                    dim_coords.append((iris.coords.DimCoord(
                        np.arange(dim_size, dtype=np.float64),
                        long_name=f'MLR prediction index {dim_idx}',
                        var_name=f'idx_{dim_idx}'), dim_idx))
                pred_cube = iris.cube.Cube(np.ma.masked_invalid(y_pred),
                                           dim_coords_and_dims=dim_coords)
            new_path = self._set_prediction_cube_attributes(
                pred_cube, pred_type, pred_name=pred_name)
            prediction_cubes[new_path] = pred_cube
            prediction_types[new_path] = pred_type
            io.save_iris_cube(pred_cube, new_path)
        return (prediction_cubes, prediction_types)

    def _get_prediction_dtype(self):
        """Get `dtype` of the output of `predict()` of the final regressor."""
        x_data = self.data['all'].x.loc[0].values.reshape(1, -1)
        y_pred = self._clf.predict(x_data)
        return y_pred.dtype

    def _get_prediction_properties(self):
        """Get important properties of prediction input."""
        datasets = select_metadata(self._datasets['training'],
                                   var_type='label')
        properties = {}
        for attr in ('dataset', 'exp', 'project', 'start_year', 'end_year'):
            attrs = list(group_metadata(datasets, attr).keys())
            properties[attr] = attrs[0]
            if len(attrs) > 1:
                if attr == 'start_year':
                    properties[attr] = min(attrs)
                elif attr == 'end_year':
                    properties[attr] = max(attrs)
                else:
                    properties[attr] = '|'.join(attrs)
                logger.debug(
                    "Attribute '%s' of label data is not unique, got values "
                    "%s, using '%s' for prediction cubes", attr, attrs,
                    properties[attr])
        return properties

    def _get_reference_cube(self, datasets, var_type, text=None):
        """Get reference cube for `datasets`."""
        msg = '' if text is None else text
        tags = self.features_types[self.features_types ==
                                   'regular'].index.values

        for tag in tags:
            dataset = self._check_dataset(datasets, var_type, tag, msg)
            if dataset is not None:
                ref_cube = self._load_cube(dataset)
                logger.debug(
                    "For var_type '%s'%s, use reference cube with tag '%s'",
                    var_type, msg, tag)
                logger.debug(ref_cube.summary(shorten=True))
                return ref_cube
        raise ValueError(
            "No {} data{} without the option 'broadcast_from' found".format(
                var_type, msg))

    def _get_verbosity_parameters(self, function, boolean=False):
        """Get verbosity parameters for class initialization."""
        verbosity_params = {
            'silent': {
                'debug': False,
                'info': False,
                'default': True,
            },
            'verbose': {
                'debug': 1,
                'info': 0,
                'default': 0,
            },
            'verbosity': {
                'debug': 2,
                'info': 1,
                'default': 0,
            },
        }
        parameters = {}
        for (param, log_levels) in verbosity_params.items():
            if param in getfullargspec(function).args:
                parameters[param] = log_levels.get(self._cfg['log_level'],
                                                   log_levels['default'])
                if boolean:
                    parameters[param] = bool(parameters[param])
                logger.debug("Set verbosity parameter '%s' of %s to '%s'",
                             param, str(function), parameters[param])
        return parameters

    def _get_x_data_for_group(self, datasets, var_type, group_attr=None):
        """Get x data for a group of datasets."""
        msg = '' if group_attr is None else " for '{}'".format(group_attr)
        ref_cube = self._get_reference_cube(datasets, var_type, msg)
        group_data = pd.DataFrame(columns=self.features,
                                  dtype=self._cfg['dtype'])

        # Iterate over all features
        for tag in self.features:
            if self.features_types[tag] != 'coordinate':
                dataset = self._check_dataset(datasets, var_type, tag, msg)
                if dataset is None:
                    new_data = np.nan
                else:
                    text = "{} '{}'{}".format(var_type, tag, msg)
                    if 'broadcast_from' in dataset:
                        cube = self._get_broadcasted_cube(
                            dataset, ref_cube, text)
                    else:
                        cube = self._load_cube(dataset)
                    self._check_cube_coords(cube,
                                            ref_cube.coords(dim_coords=True),
                                            text)
                    new_data = self._get_cube_data(cube)
            else:
                new_data = self._get_coordinate_data(ref_cube, var_type, tag,
                                                     msg)
            group_data[tag] = new_data

        # Return data and reference cube
        return (group_data, ref_cube)

    def _group_by_attributes(self, datasets):
        """Group datasets by specified attributes."""
        attributes = self._cfg.get('group_datasets_by_attributes', [])
        if not attributes:
            if self._cfg.get('accept_only_scalar_data'):
                attributes = ['dataset']
                logger.warning("Automatically set 'group_datasets_by_'"
                               "attributes' to ['dataset'] because 'accept_"
                               "only_scalar_data' is given")
            else:
                for dataset in datasets:
                    dataset['group_attribute'] = None
                return datasets
        for dataset in datasets:
            group_attribute = ''
            for attribute in attributes:
                if attribute in dataset:
                    group_attribute += dataset[attribute] + '-'
            if not group_attribute:
                group_attribute = dataset['dataset']
            else:
                group_attribute = group_attribute[:-1]
            dataset['group_attribute'] = group_attribute
        logger.info("Grouped feature and label datasets by %s", attributes)
        return datasets

    def _is_fitted(self):
        """Check if the MLR models are fitted."""
        if self._clf is None:
            return False
        x_dummy = np.ones((1, self.features.size), dtype=self._cfg['dtype'])
        try:
            self._clf.predict(x_dummy)
        except NotFittedError:
            return False
        return True

    def _is_ready_for_plotting(self):
        """Check if the class is ready for plotting."""
        if not self._is_fitted():
            logger.warning(
                "Plotting not possible, MLR model is not fitted yet")
            return False
        if not self._cfg['write_plots']:
            logger.debug("Plotting not possible, 'write_plots' is set to "
                         "'False' in user configuration file")
            return False
        return True

    def _load_classes(self):
        """Populate self._classes and check for errors."""
        self._classes['group_attributes'] = self._get_group_attributes()
        self._classes['features'] = self._get_features()
        self._classes['label'] = self._get_label()

    def _load_cube(self, dataset):
        """Load iris cube, check data type and convert units if desired."""
        logger.debug("Loading %s", dataset['filename'])
        cube = iris.load_cube(dataset['filename'])

        # Check dtype
        if not np.issubdtype(cube.dtype, np.number):
            raise TypeError(
                "Data type of cube loaded from '{}' is '{}', at "
                "the moment only numerical data is supported".format(
                    dataset['filename'], cube.dtype))

        # Convert dtypes
        cube.data = cube.core_data().astype(self._cfg['dtype'],
                                            casting='same_kind')
        for coord in cube.coords():
            try:
                coord.points = coord.points.astype(self._cfg['dtype'],
                                                   casting='same_kind')
            except TypeError:
                logger.debug(
                    "Cannot convert dtype of coordinate array '%s' from '%s' "
                    "to '%s'", coord.name(), coord.points.dtype,
                    self._cfg['dtype'])

        # Convert and check units
        if dataset.get('convert_units_to'):
            self._convert_units_in_cube(cube, dataset['convert_units_to'])
        if not cube.units == Unit(dataset['units']):
            raise ValueError(
                "Units of cube '{}' for {} '{}' differ from units given in "
                "dataset list (retrieved from ancestors or metadata.yml), got "
                "'{}' in cube and '{}' in dataset list".format(
                    dataset['filename'], dataset['var_type'], dataset['tag'],
                    cube.units.symbol, dataset['units']))
        return cube

    def _load_data(self):
        """Load train/test data (features/labels)."""
        (x_all, y_all) = self._extract_features_and_labels()
        self._data['all'] = pd.concat([x_all, y_all], axis=1, keys=['x', 'y'])
        logger.info("Loaded %i input data point(s)", len(y_all.index))

        # Split train/test data if desired
        test_size = self._cfg['test_size']
        if test_size:
            (self._data['train'], self._data['test']) = self._train_test_split(
                x_all, y_all, test_size)
            logger.info(
                "Using %i%% of the input data as test data (%i point(s))",
                int(test_size * 100), len(self.data['test'].index))
            logger.info("%i point(s) remain(s) for training",
                        len(self.data['train'].index))
        else:
            self._data['train'] = self.data['all'].copy()
            logger.info("Using all %i input data point(s) for training",
                        len(y_all.index))

    def _load_final_parameters(self):
        """Load parameters for final regressor from recipe."""
        parameters = self._cfg.get('parameters_final_regressor', {})
        logger.debug("Found parameter(s) for final regressor in recipe: %s",
                     parameters)
        verbosity_params = self._get_verbosity_parameters(self._CLF_TYPE)
        for (param, verbosity) in verbosity_params.items():
            parameters.setdefault(param, verbosity)
        return parameters

    def _load_input_datasets(self, **metadata):
        """Load input datasets (including ancestors)."""
        input_datasets = deepcopy(list(self._cfg['input_data'].values()))
        input_datasets.extend(self._get_ancestor_datasets())
        mlr.datasets_have_mlr_attributes(input_datasets,
                                         log_level='warning',
                                         mode='only_var_type')

        # Extract features and labels
        feature_datasets = select_metadata(input_datasets, var_type='feature')
        label_datasets = select_metadata(input_datasets, var_type='label')
        feature_datasets = select_metadata(feature_datasets, **metadata)
        label_datasets = select_metadata(label_datasets, **metadata)
        if metadata:
            logger.info("Only considered features and labels matching %s",
                        metadata)

        # Prediction datasets
        prediction_datasets = select_metadata(input_datasets,
                                              var_type='prediction_input')
        training_datasets = feature_datasets + label_datasets

        # Check datasets
        msg = ("At least one '{}' dataset does not have necessary MLR "
               "attributes")
        if not mlr.datasets_have_mlr_attributes(training_datasets,
                                                log_level='error'):
            raise ValueError(msg.format('training'))
        if not mlr.datasets_have_mlr_attributes(prediction_datasets,
                                                log_level='error'):
            raise ValueError(msg.format('prediction'))

        # Check if data was found
        if not training_datasets:
            msg = ' for metadata {}'.format(metadata) if metadata else ''
            raise ValueError(
                "No training data (features/labels){} found".format(msg))

        # Convert units
        self._convert_units_in_metadata(training_datasets)
        self._convert_units_in_metadata(prediction_datasets)

        # Set datasets
        logger.info(
            "Found %i training dataset(s) and %i prediction dataset(s)",
            len(training_datasets), len(prediction_datasets))
        logger.debug("Training datasets:")
        logger.debug(pformat([d['filename'] for d in training_datasets]))
        logger.debug("Prediction datasets:")
        logger.debug(pformat([d['filename'] for d in prediction_datasets]))
        self._datasets['training'] = self._group_by_attributes(
            training_datasets)
        self._datasets['prediction'] = self._group_prediction_datasets(
            prediction_datasets)

    def _load_skater_interpreters(self):
        """Load :mod:`skater` interpretation modules."""
        data_frame = self._impute_nans(self.data['train'])
        x_train = data_frame.x.values
        y_train = data_frame.y.squeeze().values

        # Global interpreter
        self._skater['global_interpreter'] = Interpretation(
            x_train, training_labels=y_train, feature_names=self.features)
        logger.debug("Loaded global skater interpreter with new training data")

        # Local interpreter (LIME)
        verbosity = self._get_verbosity_parameters(LimeTabularExplainer,
                                                   boolean=True)
        if self._cfg['return_lime_importance']:
            for param in verbosity:
                verbosity[param] = False
        self._skater['local_interpreter'] = LimeTabularExplainer(
            x_train,
            mode='regression',
            training_labels=y_train,
            feature_names=self.features,
            class_names=[self.label],
            **verbosity,
        )
        logger.debug(
            "Loaded local skater interpreter (LIME) with new training data")

        # Model
        example_size = min(y_train.size, 20)
        self._skater['model'] = InMemoryModel(
            self._clf.predict,
            feature_names=self.features,
            examples=x_train[:example_size],
            model_type='regressor',
        )
        logger.debug("Loaded skater model with new regressor")

    def _postprocess_covariance(self, cube, cov_weights):
        """Postprocess covariance prediction cube."""
        logger.debug("Postprocessing covariance matrix")
        cube = cube.collapsed(cube.coords(dim_coords=True),
                              iris.analysis.SUM,
                              weights=cov_weights)
        cube.units *= Unit('m4')
        new_units = self._cfg['prediction_pp'].get('units')
        if new_units:
            self._convert_units_in_cube(cube,
                                        new_units,
                                        power=2,
                                        text='postprocessed covariance')
        return cube

    def _postprocess_cube(self, cube, pred_type, **kwargs):
        """Postprocess regular prediction cube."""
        logger.debug("Postprocessing regular prediction cube%s",
                     '' if pred_type is None else f" of type '{pred_type}'")
        cfg = dict(self._cfg['prediction_pp'])
        cov_weights = None
        if all([
                cfg['area_weights'],
                pred_type is None,
                'return_cov' in kwargs,
        ]):
            cov_weights = self._get_area_weights(cube).ravel()
            cov_weights = cov_weights[~np.ma.getmaskarray(cube.data).ravel()]
        ops = {'mean': iris.analysis.MEAN, 'sum': iris.analysis.SUM}

        # Perform desired postprocessing operations
        n_points_mean = None
        old_size = np.prod(cube.shape, dtype=np.int)
        power = self._pred_type_to_power(pred_type)
        for (op_type, iris_op) in ops.items():
            if not cfg.get(op_type):
                continue
            logger.debug("Calculating %s for coordinates %s", op_type,
                         cfg[op_type])
            weights = None
            if all([
                    cfg['area_weights'],
                    'latitude' in cfg[op_type],
                    'longitude' in cfg[op_type],
            ]):
                weights = self._get_area_weights(cube, pred_type)
            cube = cube.collapsed(cfg[op_type], iris_op, weights=weights)
            if op_type == 'mean':
                new_size = np.prod(cube.shape, dtype=np.int)
                n_points_mean = int(old_size / new_size)
            elif op_type == 'sum' and weights is not None:
                cube.units *= self._units_power(Unit('m2'), power)

        # Units conversion
        if cfg.get('units'):
            self._convert_units_in_cube(
                cube,
                cfg['units'],
                power=power,
                text='postprocessed prediction output{}'.format(
                    '' if pred_type is None else f" of type '{pred_type}'"))

        # Weights for covariance matrix
        if cov_weights is not None:
            logger.debug("Calculating covariance weights (memory-intensive)")
            cov_weights = np.outer(cov_weights, cov_weights)
            if n_points_mean is not None:
                cov_weights /= n_points_mean**2
        return (cube, cov_weights)

    def _postprocess_predictions(self, predictions, pred_types, **kwargs):
        """Postprocess prediction cubes if desired."""
        if not self._cfg['prediction_pp']:
            return
        self._cfg['prediction_pp'].setdefault('area_weights', True)
        ref_path = list(pred_types.keys())[list(
            pred_types.values()).index(None)]
        logger.info("Postprocessing prediction output using %s",
                    self._cfg['prediction_pp'])
        logger.debug("Using reference cube at '%s'", ref_path)

        # Process and save reference cube
        ref_cube = predictions[ref_path]
        ref_shape = ref_cube.shape
        (ref_cube,
         cov_weights) = self._postprocess_cube(ref_cube, None, **kwargs)
        new_path = self._get_postprocessed_filename(ref_path, ref_path)
        ref_cube.attributes['source'] = ref_path
        ref_cube.attributes['filename'] = new_path  # TODO remove after merge
        io.save_iris_cube(ref_cube, new_path)

        # Process other cubes
        new_path = ref_path.replace('.nc', '_pp.nc')
        ref_cube.attributes['source'] = ref_path
        ref_cube.attributes['filename'] = new_path  # TODO remove after merge
        for (path, cube) in predictions.items():
            if path == ref_path or cube.attributes.get('skip_for_pp'):
                continue

            # Regualar cubes
            if cube.shape == ref_shape:
                (cube, _) = self._postprocess_cube(cube, pred_types[path])

            # Covariance
            else:
                if cov_weights is not None:
                    if cube.shape != cov_weights.shape:
                        logger.warning(
                            "Cannot postprocess all prediction cubes, "
                            "expected shapes %s or %s (for covariance), got "
                            "%s", ref_shape, cov_weights.shape, cube.shape)
                        continue
                cube = self._postprocess_covariance(cube, cov_weights)

            # Fix attributes and append
            new_path = self._get_postprocessed_filename(path, ref_path)
            cube.attributes['source'] = path
            cube.attributes['filename'] = new_path  # TODO remove after merge
            io.save_iris_cube(cube, new_path)

    def _prediction_to_dict(self, pred_out, **kwargs):
        """Convert output of `clf.predict()` to `dict`."""
        if not isinstance(pred_out, (list, tuple)):
            pred_out = [pred_out]
        idx_to_name = {0: None}
        if 'return_var' in kwargs:
            idx_to_name[1] = 'var'
        elif 'return_cov' in kwargs:
            idx_to_name[1] = 'cov'
        pred_dict = {}
        for (idx, pred) in enumerate(pred_out):
            pred = pred.astype(self._cfg['dtype'], casting='same_kind')
            if pred.ndim == 2 and pred.shape[1] == 1:
                logger.warning(
                    "Prediction output is 2D and length of second axis is 1, "
                    "squeezing second axis")
                pred = np.squeeze(pred, axis=1)
            pred_dict[idx_to_name.get(idx, idx)] = pred
        return pred_dict

    def _remove_missing_features(self, x_data, y_data):
        """Remove missing values in the features data (if desired)."""
        if self._cfg['imputation_strategy'] != 'remove':
            return (x_data, y_data)
        mask = x_data.isnull().any(axis=1).values
        new_x_data = x_data[~mask].reset_index(drop=True)
        new_y_data = y_data[~mask].reset_index(drop=True)
        diff = len(y_data.index) - len(new_y_data.index)
        if diff:
            msg = ('Removed %i training point(s) where features were '
                   'missing')
            if self._cfg.get('accept_only_scalar_data'):
                removed_groups = self.group_attributes[mask]
                msg += ' ({})'.format(removed_groups)
                self._classes['group_attributes'] = (
                    self.group_attributes[~mask])
            logger.info(msg, diff)
        return (new_x_data, new_y_data)

    def _impute_nans(self, data_frame, copy=True):
        """Impute all nans of a `data_frame`."""
        if copy:
            data_frame = data_frame.copy()
        if 'imputer' in self._clf.named_steps:
            new_x = self._clf.named_steps['imputer'].transform(
                data_frame.x.values)
            data_frame.x.values[:] = new_x
        return data_frame

    def _save_csv_file(self, data_type, filename, pred_name=None):
        """Save CSV file."""
        if data_type not in self.data:
            return
        if data_type == 'pred':
            csv_data = self.data[data_type][pred_name]
        else:
            csv_data = self.data[data_type]

        # Filename and path
        if filename is None:
            if pred_name is None:
                filename = '{}.csv'.format(data_type)
            else:
                filename = '{}_{}.csv'.format(data_type, pred_name)
        path = os.path.join(self._cfg['mlr_work_dir'], filename)

        # Save file
        csv_data.to_csv(path, na_rep='nan')
        logger.info("Wrote %s", path)

    def _set_prediction_cube_attributes(self, cube, pred_type, pred_name=None):
        """Set the attributes of the prediction cube."""
        cube.attributes = {
            'regressor': str(self._CLF_TYPE),
            'description': 'MLR model prediction',
            'tag': self.label,
            'var_type': 'prediction_output',
        }
        if pred_name is not None:
            cube.attributes['prediction_name'] = pred_name
        cube.attributes.update(self._get_prediction_properties())
        for (key, val) in self.parameters.items():
            cube.attributes[key] = str(val)
        label = select_metadata(self._datasets['training'],
                                var_type='label')[0]
        label_cube = self._load_cube(label)
        for attr in ('standard_name', 'var_name', 'long_name', 'units'):
            setattr(cube, attr, getattr(label_cube, attr))

        # Modify variable name depending on prediction type
        suffix = f'_{pred_type}'
        if pred_type is None:
            suffix = ''
        elif isinstance(pred_type, int):
            cube.var_name += '_{:d}'.format(pred_type)
            cube.long_name += ' {:d}'.format(pred_type)
            logger.warning("Got unknown prediction type with index %i",
                           pred_type)
        elif pred_type in ('var', 'cov'):
            cube.var_name += suffix
            cube.long_name += (' (variance)'
                               if pred_type == 'var' else ' (covariance)')
            cube.units = self._units_power(cube.units, 2)
        elif 'squared_error_estim' in pred_type:
            cube.var_name += suffix
            cube.long_name += (' (squared error estimation using {})'.format(
                'cross-validation' if 'cv' in
                pred_type else 'holdout test data set'))
            cube.units = self._units_power(cube.units, 2)
        elif pred_type == 'lime':
            cube.var_name = 'lime_feature_importance'
            cube.long_name = (f'Most important feature for predicting '
                              f'{self.label} given by LIME')
            cube.units = Unit('no_unit')
            cube.attributes.update({
                'features':
                pformat(dict(enumerate(self.features))),
                'skip_for_pp':
                1,
            })
        else:
            logger.warning(
                "Got unknown prediction type '%s', setting correct attributes "
                "not possible", pred_type)

        # Get new path
        pred_str = '' if pred_name is None else f'_{pred_name}'
        root_str = ('' if self._cfg['root_dir'] == '' else
                    f"{self._cfg['root_dir']}_")
        filename = f'{root_str}prediction{pred_str}{suffix}.nc'
        new_path = os.path.join(self._cfg['mlr_work_dir'], filename)
        cube.attributes['filename'] = new_path
        return new_path

    def _train_test_split(self, x_data, y_data, test_size):
        """Split data into training and test data."""
        (x_train, x_test, y_train,
         y_test) = train_test_split(x_data.values,
                                    y_data.values,
                                    test_size=test_size)
        x_train = pd.DataFrame(x_train, columns=self.features)
        x_test = pd.DataFrame(x_test, columns=self.features)
        y_train = pd.DataFrame(y_train, columns=[self.label])
        y_test = pd.DataFrame(y_test, columns=[self.label])
        train = pd.concat([x_train, y_train], axis=1, keys=['x', 'y'])
        test = pd.concat([x_test, y_test], axis=1, keys=['x', 'y'])
        return (train, test)

    def _update_fit_kwargs(self, fit_kwargs):
        """Update fit kwargs (only used for some models)."""
        return fit_kwargs

    @staticmethod
    def _convert_units_in_metadata(datasets):
        """Convert units of datasets if desired."""
        for dataset in datasets:
            if not dataset.get('convert_units_to'):
                continue
            units_from = Unit(dataset['units'])
            units_to = Unit(dataset['convert_units_to'])
            try:
                units_from.convert(0.0, units_to)
            except ValueError:
                logger.warning(
                    "Cannot convert units of %s '%s' from '%s' to '%s'",
                    dataset['var_type'], dataset['tag'], units_from.origin,
                    units_to.origin)
                dataset.pop('convert_units_to')
            else:
                dataset['units'] = dataset['convert_units_to']

    @staticmethod
    def _get_coordinate_data(ref_cube, var_type, tag, text=None):
        """Get coordinate variable `ref_cube` which can be used as x data."""
        msg = '' if text is None else text
        try:
            coord = ref_cube.coord(tag)
        except iris.exceptions.CoordinateNotFoundError:
            raise iris.exceptions.CoordinateNotFoundError(
                "Coordinate '{}' given in 'coords_as_features' not found in "
                "reference cube for '{}'{}".format(tag, var_type, msg))
        coord_array = np.ma.filled(coord.points, np.nan)
        coord_dims = ref_cube.coord_dims(coord)
        if coord_dims == ():
            logger.warning(
                "Coordinate '%s' is scalar, including it as feature does not "
                "add any information to the model (array is constant)", tag)
        else:
            new_axis_pos = np.delete(np.arange(ref_cube.ndim), coord_dims)
            for idx in new_axis_pos:
                coord_array = np.expand_dims(coord_array, idx)
        coord_array = np.broadcast_to(coord_array, ref_cube.shape)
        logger.debug("Added coordinate %s '%s'%s", var_type, tag, msg)
        return coord_array.ravel()

    @staticmethod
    def _get_cube_data(cube):
        """Get data from cube."""
        cube_data = np.ma.filled(cube.data, np.nan)
        return cube_data.ravel()

    @staticmethod
    def _get_postprocessed_filename(path, ref_path):
        """Get name of postprocessed prediction file."""
        path = path.replace('.nc', '')
        ref_path = ref_path.replace('.nc', '')
        suffix = path.replace(ref_path, '')
        return f'{ref_path}_pp{suffix}.nc'

    @staticmethod
    def _group_prediction_datasets(datasets):
        """Group prediction datasets (use `prediction_name` key)."""
        for dataset in datasets:
            dataset['group_attribute'] = None
        return group_metadata(datasets, 'prediction_name')

    @staticmethod
    def _pred_type_to_power(pred_type):
        """Get power for prediction type (e.g. 2 for variance)."""
        if pred_type is None:
            return 1
        if pred_type in ('cov', 'var'):
            return 2
        if 'squared_error_estim' in pred_type:
            return 2
        default = 1
        logger.debug(
            "No specific power for prediction type '%s' defined, defaulting "
            "to %i", pred_type, default)
        return default

    @staticmethod
    def _remove_missing_labels(x_data, y_data):
        """Remove missing values in the label data."""
        mask = y_data.isnull().values
        new_x_data = x_data[~mask].reset_index(drop=True)
        new_y_data = y_data[~mask].reset_index(drop=True)
        diff = len(y_data.index) - len(new_y_data.index)
        if diff:
            logger.info(
                "Removed %i training point(s) where labels were missing", diff)
        return (new_x_data, new_y_data)

    @staticmethod
    def _units_power(units, power):
        """Raise a :mod:`cf_units.Unit` to a given power preserving symbols."""
        if round(power) != power:
            raise TypeError(f"Expected integer power for units "
                            f"exponentiation, got {power}")
        if any([units.is_no_unit(), units.is_unknown()]):
            logger.warning("Cannot raise units '%s' to power %i", units.name,
                           power)
            return units
        if units.origin is None:
            logger.warning(
                "Symbol-preserving exponentiation of units '%s' is not "
                "supported, origin is not given", units.symbol)
            return units**power
        if units.origin.split()[0][0].isdigit():
            logger.warning(
                "Symbol-preserving exponentiation of units '%s' is not "
                "supported yet because of leading numbers", units.symbol)
            return units**power
        new_units_list = []
        for split in units.origin.split():
            for elem in split.split('.'):
                if elem[-1].isdigit():
                    exp = [int(d) for d in re.findall(r'-?\d+', elem)][0]
                    val = ''.join(
                        [abc for abc in re.findall(r'[A-Za-z]', elem)])
                    new_units_list.append(f'{val}{exp * power}')
                else:
                    new_units_list.append(f'{elem}{power}')
        new_units = ' '.join(new_units_list)
        return Unit(new_units)
