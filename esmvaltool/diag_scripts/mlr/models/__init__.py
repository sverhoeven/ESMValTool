"""Base class for MLR models."""

import copy
import importlib
import logging
import os
import re
from inspect import getfullargspec
from pprint import pformat

import iris
import matplotlib.pyplot as plt
import numpy as np
import pathos.multiprocessing as mp
from cf_units import Unit
from skater.core.explanations import Interpretation
from skater.core.local_interpretation.lime.lime_tabular import \
    LimeTabularExplainer
from skater.model import InMemoryModel
from sklearn import metrics
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (GridSearchCV, LeaveOneOut,
                                     cross_val_score, train_test_split)
from sklearn.preprocessing import StandardScaler

from esmvaltool.diag_scripts import mlr
from esmvaltool.diag_scripts.shared import (group_metadata, io, plot,
                                            select_metadata)

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
    the `feature` datasets (except `allow_missing_features`) is set to `True`.
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
    dtype : str, optional (default: 'float64')
        Internal data type which is used for all calculations, see
        <https://docs.scipy.org/doc/numpy/user/basics.types.html> for a list
        of allowed values.
    estimate_prediction_error : dict, optional
        Estimate (constant) prediction error using RMSE. This can be calculated
        by a (holdout) test data set (`type: test`) or by cross-validation
        from the training data (`type: cv'). The latter uses
        :mod:`sklearn.model_selection.cross_val_score` (see
        <https://scikit-learn.org/stable/modules/cross_validation.html>),
        additional keyword arguments can be passed via the `kwargs` key.
    fit_kwargs : dict, optional
        Optional keyword arguments for the classifier's `fit()` function. Have
        to be given for each step of the pipeline seperated by two underscores,
        i.e. `s__p` is the parameter `p` for step `s`.
    grid_search_cv_param_grid : dict or list of dict, optional
        Parameters (keys) and ranges (values) for exhaustive parameter search
        using cross-validation. Have to be given for each step of the pipeline
        seperated by two underscores, i.e. `s__p` is the parameter `p` for step
        `s`.
    grid_search_cv_kwargs : dict, optional
        Keyword arguments for the grid search cross-validation, see
        <https://scikit-learn.org/stable/modules/generated/
        sklearn.model_selection.GridSearchCV.html>.
    group_datasets_by_attributes : list of str, optional
        List of dataset attributes which are used to group input data for
        `features` and `labels`, e.g. specify `dataset` to use the different
        `dataset`s as observations for the MLR model.
    imputation_strategy : str, optional (default: 'remove')
        Strategy for the imputation of missing values in the features. Must be
        one of `remove`, `mean`, `median`, `most_frequent` or `constant`.
    imputation_constant : str or numerical value, optional (default: None)
        When imputation strategy is `constant`, replace missing values with
        this value. If `None`, replace numerical values by 0 and others by
        `missing_value`.
    matplotlib_style_file : str, optional
        Matplotlib style file (should be located in
        `esmvaltool.diag_scripts.shared.plot.styles_python.matplotlib`).
    n_jobs : int, optional (default: 1)
        Maximum number of jobs spawned by this class.
    parameters : dict, optional
        Parameters used in the **final** regressor. If these parameters are
        updated using the function `self.update_parameters()`, the new names
        have to be given for each step of the pipeline seperated by two
        underscores, i.e. `s__p` is the parameter `p` for step `s`.
    predict_kwargs : dict, optional
        Optional keyword arguments for the classifier's `predict()` function.
    prediction_pp : dict, optional
        Postprocess prediction output (e.g. combine best estimate and standard
        deviation in one cube). Accepts keywords `mean` and `sum` (followed by
        list of coordinates) and `area_weighted` (bool).
    return_lime_importance : bool, optional (default: True)
        Return cube with feature importance given by LIME (Local Interpretable
        Model-agnostic Explanations).
    standardize_data : bool, optional (default: True)
        Linearly standardize input data by removing mean and scaling to unit
        variance.
    test_size : float, optional (default: 0.25)
        Fraction of feature/label data which is used as test data and not for
        training (if desired).
    use_coords_as_feature : list, optional
        Use coordinates (e.g. 'air_pressure' or 'latitude') as features.
    use_only_coords_as_features : bool, optional (default: False)
        Use only the specified coordinates as features.

    """

    _CLF_TYPE = None
    _MODELS = {}
    _PIPELINE_FINAL_STEP = 'transformed_target_regressor'

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

        def decorator(subclass):
            """Decorate subclass."""
            cls._MODELS[model] = subclass
            return subclass

        logger.debug("Found available MLR model '%s'", model)
        return decorator

    @classmethod
    def create(cls, model, *args, **kwargs):
        """Create desired MLR model subclass (factory method)."""
        cls._load_mlr_models()
        if not cls._MODELS:
            logger.error("No MLR models found, please add subclasses to "
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
        logger.info("Created MLR model '%s' with classifier %s", model,
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
        self._cfg = copy.deepcopy(cfg)
        self._clf = None
        self._data = {}
        self._data['x_pred'] = {}
        self._data['y_pred'] = {}
        self._datasets = {}
        self._skater = {}
        self._classes = {}
        self._parameters = self._load_cfg_parameters()

        # Default parameters
        self._cfg.setdefault('dtype', 'float64')
        self._cfg.setdefault('imputation_strategy', 'remove')
        self._cfg.setdefault('n_jobs', 1)
        self._cfg.setdefault('prediction_pp', {})
        self._cfg.setdefault('return_lime_importance', True)
        plt.style.use(
            plot.get_path_to_mpl_style(self._cfg.get('matplotlib_style_file')))

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
        self._load_training_data()

        # Create classifier (with all preprocessor steps)
        self._create_classifier()

        # Log successful initialization
        logger.info("Initialized MLR model (using at most %i processes)",
                    self._cfg['n_jobs'])
        logger.debug("With parameters")
        logger.debug(pformat(self.parameters))

    @property
    def classes(self):
        """Classes of the model (read-only), e.g. features, label, etc."""
        return self._classes

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
        for data_type in ('x_pred', 'y_pred'):
            for pred_name in self._data[data_type]:
                self._save_csv_file(data_type,
                                    filename,
                                    is_prediction=True,
                                    pred_name=pred_name)

    def export_training_data(self, filename=None):
        """Export all training data contained in `self._data`.

        Parameters
        ----------
        filename : str, optional (default: '{data_type}.csv')
            Name of the exported files.

        """
        for data_type in ('x_data', 'x_train', 'x_test', 'y_data', 'y_train',
                          'y_test'):
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
            "Fitting MLR model with final classifier %s on %i training "
            "point(s)", self._CLF_TYPE, self._data['y_train'].size)
        fit_kwargs = dict(self._cfg.get('fit_kwargs', {}))
        fit_kwargs.update(kwargs)
        if fit_kwargs:
            logger.info(
                "Using additional keyword argument(s) %s for fit() function",
                fit_kwargs)

        # Create MLR model with desired parameters and fit it
        self._clf.fit(self._data['x_train'], self._data['y_train'],
                      **fit_kwargs)
        self._parameters = self._get_clf_parameters()
        logger.info("Successfully fitted MLR model on %i training point(s)",
                    self._data['y_train'].size)
        logger.debug("Pipeline steps:")
        logger.debug(pformat(self._clf.named_steps))
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
            logger.error(
                "No parameter grid given (neither in recipe nor in grid_"
                "search_cv() function)")
            return
        logger.info(
            "Performing exhaustive grid search cross-validation with final "
            "classifier %s and parameter grid %s on %i training points",
            self._CLF_TYPE, parameter_grid, self._data['y_train'].size)

        # Get keyword arguments
        log_level = {'debug': 2, 'info': 1}
        gridsearch_kwargs = {
            'n_jobs': self._cfg['n_jobs'],
            'verbose': log_level.get(self._cfg['log_level'], 0),
        }
        gridsearch_kwargs.update(self._cfg.get('grid_search_cv_kwargs', {}))
        gridsearch_kwargs.update(kwargs)
        logger.info(
            "Using keyword argument(s) %s for grid_search_cv() "
            "function", gridsearch_kwargs)
        if isinstance(gridsearch_kwargs.get('cv'), str):
            if gridsearch_kwargs['cv'].lower() == 'loo':
                gridsearch_kwargs['cv'] = LeaveOneOut()

        # Create GridSearchCV instance
        clf = GridSearchCV(self._clf, parameter_grid, **gridsearch_kwargs)
        clf.fit(self._data['x_train'], self._data['y_train'])
        self._parameters.update(clf.best_params_)
        if hasattr(clf, 'best_estimator_'):
            self._clf = clf.best_estimator_
        else:
            self._clf = self._CLF_TYPE(**self.parameters)
            self._clf.fit(self._data['x_train'], self._data['y_train'])
        self._parameters = self._get_clf_parameters()
        logger.info(
            "Exhaustive grid search successful, found best parameter(s) %s",
            clf.best_params_)
        logger.debug("CV results:")
        logger.debug(pformat(clf.cv_results_))
        logger.info("Successfully fitted MLR model on %i training point(s)",
                    self._data['y_train'].size)
        logger.debug("Pipeline steps:")
        logger.debug(pformat(self._clf.named_steps))
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
                self._CLF_TYPE.__name__))
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
        x_type = f'x_{data_type}'
        if x_type not in self._data:
            logger.error("Cannot plot LIME, got invalid data type '%s'",
                         data_type)
            return
        if index >= self._data[x_type].shape[0]:
            logger.error(
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
            self._data[x_type][index], self._clf.predict)
        logger.info(pformat(explainer.as_list()))
        logger.info(pformat(explainer.as_map()))
        explainer.save_to_file(html_path)
        logger.info("Wrote %s", html_path)

        # Plot
        explainer.as_pyplot_figure()
        plt.savefig(plot_path, orientation='landscape', bbox_inches='tight')
        logger.info("Wrote %s", plot_path)
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
        progressbar = True if self._cfg['log_level'] == 'debug' else False

        # Plot for every feature
        for feature_name in self.classes['features']:
            logger.debug("Plotting partial dependence of '%s'", feature_name)
            ((_, axes), ) = (self._skater['global_interpreter'].
                             partial_dependence.plot_partial_dependence(
                                 [feature_name],
                                 self._skater['model'],
                                 n_jobs=self._cfg['n_jobs'],
                                 progressbar=progressbar,
                                 with_variance=True))
            axes.set_title('Partial dependence ({} Model)'.format(
                self._CLF_TYPE.__name__))
            axes.set_xlabel(feature_name)
            axes.set_ylabel(self.classes['label'])
            axes.get_legend().remove()
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
        for (f_idx, feature) in enumerate(self.classes['features']):
            logger.debug("Plotting scatterplot of '%s'", feature)
            (_, axes) = plt.subplots()
            if self._cfg.get('accept_only_scalar_data'):
                for (g_idx, group_attr) in enumerate(
                        self.classes['group_attributes']):
                    axes.scatter(self._data['x_data'][g_idx, f_idx],
                                 self._data['y_data'][g_idx],
                                 label=group_attr)
                for (pred_name, x_pred) in self._data['x_pred'].items():
                    axes.axvline(x_pred[0, f_idx],
                                 linestyle='--',
                                 color='black',
                                 label=('Observation'
                                        if pred_name is None else pred_name))
                legend = axes.legend(loc='center left',
                                     ncol=2,
                                     bbox_to_anchor=[1.05, 0.5],
                                     borderaxespad=0.0)
            else:
                axes.plot(self._data['x_data'][:, f_idx], self._data['y_data'],
                          '.')
                legend = None
            axes.set_title(feature)
            axes.set_xlabel('{} / {}'.format(
                feature, self.classes['features_units'][f_idx]))
            axes.set_ylabel('{} / {}'.format(self.classes['label'],
                                             self.classes['label_units']))
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
            logger.error(
                "Prediction not possible, MLR model is not fitted yet")
            return
        predict_kwargs = dict(self._cfg.get('predict_kwargs', {}))
        predict_kwargs.update(kwargs)
        if predict_kwargs:
            logger.info(
                "Using additional keyword argument(s) %s for predict() "
                "function", predict_kwargs)

        # Iterate over predictions
        if not self._datasets['prediction']:
            logger.error("Prediction not possible, no 'prediction_input' "
                         "datasets given")
        for pred_name in self._datasets['prediction']:
            if pred_name is None:
                logger.info("Started prediction")
            else:
                logger.info("Started prediction for prediction %s", pred_name)

            # Prediction
            (x_pred, x_mask,
             x_cube) = self._extract_prediction_input(pred_name)
            pred_dict = self._get_prediction_dict(x_pred, x_mask,
                                                  **predict_kwargs)
            pred_dict = self._estimate_prediction_error(pred_dict)

            # Save data
            x_pred = np.ma.array(x_pred, mask=x_mask, copy=True)
            self._data['x_pred'][pred_name] = x_pred.filled(np.nan)
            self._data['y_pred'][pred_name] = np.ma.copy(
                pred_dict['pred']).filled(np.nan)

            # Get (and save) prediction cubes
            (predictions,
             ref_path) = self._get_prediction_cubes(pred_dict, pred_name,
                                                    x_cube)

            # Postprocess prediction cubes (if desired)
            self._postprocess_predictions(predictions, ref_path,
                                          **predict_kwargs)

    def print_regression_metrics(self):
        """Print all available regression metrics for the test data."""
        if not self._is_fitted():
            logger.error(
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
        for data_type in ('train', 'test'):
            logger.info("Evaluating regression metrics for %s data", data_type)
            x_data = self._data[f'x_{data_type}']
            y_true = self._data[f'y_{data_type}']
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

    def reset_training_data(self):
        """Reset training and test data."""
        self._data['x_train'] = np.copy(self._data['x_data'])
        self._data['y_train'] = np.copy(self._data['y_data'])
        for data_type in ('x_test', 'y_test'):
            self._data.pop(data_type, None)
        logger.debug("Resetted input data")

    def simple_train_test_split(self, test_size=None):
        """Split input data into training and test data.

        Parameters
        ----------
        test_size : float, optional (default: 0.25)
            Fraction of feature/label data which is used as test data and not
            for training. Overwrites default and recipe settings.

        """
        if test_size is None:
            test_size = self._cfg.get('test_size', 0.25)
        (self._data['x_train'], self._data['x_test'], self._data['y_train'],
         self._data['y_test']) = train_test_split(self._data['x_data'],
                                                  self._data['y_data'],
                                                  test_size=test_size)
        logger.info("Used %i%% of the input data as test data (%i point(s))",
                    int(test_size * 100), self._data['y_test'].size)

    def update_parameters(self, **params):
        """Update parameters of the classifier.

        Parameters
        ----------
        **params : keyword arguments, optional
            Paramaters of the classifier which should be updated.

        Note
        ----
        Parameter names have to be given for each step of the pipeline
        seperated by two underscores, i.e. `s__p` is the parameter `p` for
        step `s`.

        """
        new_params = {}
        for (key, val) in params.items():
            if key in self.parameters:
                new_params[key] = val
            else:
                logger.warning(
                    "'%s' is not a valid parameter for the classifier")
        self._clf.set_params(**new_params)
        self._parameters = self._get_clf_parameters()
        logger.info("Updated classifier with parameters %s", new_params)

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
        """Check if datasets are exist and are valid."""
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
            units = self.classes['label_units']
        else:
            units = self.classes['features_units'][np.where(
                self.classes['features'] == tag)][0]
        if units != Unit(datasets[0]['units']):
            raise ValueError(
                "Expected units '{}' for {} '{}'{}, got '{}'".format(
                    units, var_type, tag, msg, datasets[0]['units']))
        return datasets[0]

    def _check_label(self, label, units):
        """Check if `label` matches with already saved data."""
        if self.classes.get('label') is None:
            self.classes['label'] = label
            self.classes['label_units'] = units
        else:
            if label != self.classes['label']:
                raise ValueError(
                    "Expected unique entries for var_type label', got '{}' "
                    "and '{}'".format(label, self.classes['label']))
            if units != self.classes['label_units']:
                raise ValueError(
                    "Expected unique units for the label '{}', got '{}' and "
                    "'{}'".format(self.classes['label'], units,
                                  self.classes['label_units']))

    def _clf_is_valid(self, text=None):
        """Check if valid classifier type is given."""
        msg = '' if text is None else '{} not possible: '.format(text)
        if self._CLF_TYPE is None:
            logger.error(
                "%sNo MLR model specified, please use factory function "
                "'MLRModel.create()' to initialize this class or populate the "
                "module 'esmvaltool.diag_scripts.mlr.models' if necessary",
                msg)
            return False
        return True

    def _create_classifier(self):
        """Create classifier with correct settings."""
        if not self._clf_is_valid(text='Creating classifier'):
            return
        steps = []

        # Imputer
        if self._cfg['imputation_strategy'] != 'remove':
            imputer = SimpleImputer(
                strategy=self._cfg['imputation_strategy'],
                fill_value=self._cfg.get('imputation_constant'))
            steps.append(('imputer', imputer))

        # Scaler
        scale_data = self._cfg.get('standardize_data', True)
        x_scaler = StandardScaler(with_mean=scale_data, with_std=scale_data)
        y_scaler = StandardScaler(with_mean=scale_data, with_std=scale_data)
        steps.append(('x_scaler', x_scaler))

        # Regressor
        regressor = self._CLF_TYPE(**self.parameters)
        transformed_regressor = mlr.AdvancedTransformedTargetRegressor(
            transformer=y_scaler, regressor=regressor)
        steps.append((self._PIPELINE_FINAL_STEP, transformed_regressor))

        # Final classifier
        if self._cfg.get('cache_intermediate_results', True):
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
        self._parameters = self._get_clf_parameters()
        logger.debug("Created classifier")

    def _estimate_prediction_error(self, pred_dict):
        """Estimate prediction error."""
        if not self._cfg.get('estimate_prediction_error'):
            return pred_dict
        cfg = copy.deepcopy(self._cfg['estimate_prediction_error'])
        cfg.setdefault('type', 'cv')
        cfg.setdefault('kwargs', {})
        cfg['kwargs'].setdefault('cv', 5)
        cfg['kwargs'].setdefault('scoring', 'neg_mean_squared_error')
        logger.debug("Estimating prediction error using %s", cfg)
        error = None

        # Test data set
        if cfg['type'] == 'test':
            if 'x_test' in self._data and 'y_test' in self._data:
                y_pred = self._clf.predict(self._data['x_test'])
                error = np.sqrt(
                    metrics.mean_squared_error(self._data['y_test'], y_pred))
            else:
                logger.warning(
                    "Cannot estimate prediction error using 'type: test', "
                    "no test data set given (use 'simple_train_test_split'), "
                    "using cross-validation instead")
                cfg['type'] = 'cv'

        # CV
        if cfg['type'] == 'cv':
            error = cross_val_score(self._clf, self._data['x_train'],
                                    self._data['y_train'], **cfg['kwargs'])
            error = np.mean(error)
            if cfg['kwargs']['scoring'].startswith('neg_'):
                error = -error
            if 'squared' in cfg['kwargs']['scoring']:
                error = np.sqrt(error)

        # Get correct shape and mask
        if error is not None:
            pred_error = np.ma.array(np.full_like(pred_dict['pred'], error),
                                     mask=pred_dict['pred'].mask)
            pred_dict[f"error_estim_{cfg['type']}"] = pred_error
            logger.info("Estimated prediction error by %s using %s data",
                        error, cfg['type'])
            return pred_dict
        logger.error(
            "Got invalid type for prediction error estimation, got '%s', "
            "expected 'test' or 'cv'", cfg['type'])
        return pred_dict

    def _extract_features_and_labels(self):
        """Extract feature and label data points from training data."""
        datasets = self._datasets['training']
        (x_data, _) = self._extract_x_data(datasets, 'feature')
        y_data = self._extract_y_data(datasets)
        if x_data.shape[0] != y_data.size:
            raise ValueError(
                "Sizes of features and labels do not match, got {:d} points "
                "for the features and {:d} points for the label".format(
                    x_data.shape[0], y_data.size))
        logger.info("Found %i raw input data point(s) with data type '%s'",
                    y_data.size, y_data.dtype)

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
            x_data.shape[0], x_data.dtype)

        # If desired missing values get removed in the output cube via a mask
        x_mask = np.ma.getmaskarray(x_data)
        x_data = x_data.filled(np.nan)
        return (x_data, x_mask, prediction_input_cube)

    def _extract_x_data(self, datasets, var_type):
        """Extract required x data of type `var_type` from `datasets`."""
        allowed_types = ('feature', 'prediction_input')
        if var_type not in allowed_types:
            raise ValueError("Excepted one of '{}' for 'var_type', got "
                             "'{}'".format(allowed_types, var_type))

        # Collect data from datasets and return it
        datasets = select_metadata(datasets, var_type=var_type)
        x_data = None
        cube = None

        # Iterate over datasets
        if var_type == 'feature':
            groups = self.classes['group_attributes']
        else:
            groups = [None]
        for group_attr in groups:
            attr_datasets = select_metadata(datasets,
                                            group_attribute=group_attr)
            if group_attr is not None:
                logger.info("Loading '%s' data of '%s'", var_type, group_attr)
            msg = '' if group_attr is None else " for '{}'".format(group_attr)
            if not attr_datasets:
                raise ValueError("No '{}' data{} found".format(var_type, msg))
            (attr_data,
             cube) = self._get_x_data_for_group(attr_datasets, var_type,
                                                group_attr)

            # Append data
            if x_data is None:
                x_data = attr_data
            else:
                x_data = np.ma.vstack((x_data, attr_data))

        return (x_data, cube)

    def _extract_y_data(self, datasets):
        """Extract y data (labels) from `datasets`."""
        datasets = select_metadata(datasets, var_type='label')
        y_data = np.ma.array([], dtype=self._cfg['dtype'])
        for group_attr in self.classes['group_attributes']:
            if group_attr is not None:
                logger.info("Loading 'label' data of '%s'", group_attr)
            msg = '' if group_attr is None else " for '{}'".format(group_attr)
            datasets_ = select_metadata(datasets, group_attribute=group_attr)
            dataset = self._check_dataset(datasets_, 'label',
                                          self.classes['label'], msg)
            cube = self._load_cube(dataset)
            text = "label '{}'{}".format(self.classes['label'], msg)
            self._check_cube_coords(cube, None, text)
            y_data = np.ma.hstack((y_data, self._get_cube_data(cube)))
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

    def _get_broadcasted_cube(self, dataset, ref_cube, text=None):
        """Get broadcasted cube."""
        msg = 'data' if text is None else text
        target_shape = ref_cube.shape
        cube_to_broadcast = self._load_cube(dataset)
        data_to_broadcast = np.ma.array(cube_to_broadcast.data)
        try:
            new_axis_pos = np.delete(np.arange(len(target_shape)),
                                     dataset['broadcast_from'])
        except IndexError:
            raise ValueError(
                "Broadcasting to shape {} failed{}, index out of bounds".
                format(target_shape, msg))
        logger.info("Broadcasting %s from %s to %s", msg,
                    data_to_broadcast.shape, target_shape)
        for idx in new_axis_pos:
            data_to_broadcast = np.ma.expand_dims(data_to_broadcast, idx)
        mask = data_to_broadcast.mask
        data_to_broadcast = np.broadcast_to(data_to_broadcast,
                                            target_shape,
                                            subok=True)
        data_to_broadcast.mask = np.broadcast_to(mask, target_shape)
        new_cube = ref_cube.copy(data_to_broadcast)
        for idx in dataset['broadcast_from']:
            new_coord = new_cube.coord(dimensions=idx)
            new_coord.points = cube_to_broadcast.coord(new_coord).points
        logger.debug("Added broadcasted %s", msg)
        return new_cube

    def _get_clf_parameters(self, deep=True):
        """Get parameters of classifier."""
        return self._clf.get_params(deep=deep)

    def _get_features(self):
        """Extract all features from the `prediction_input` datasets."""
        pred_name = list(self._datasets['prediction'].keys())[0]
        datasets = self._datasets['prediction'][pred_name]
        msg = ('' if pred_name is None else
               " for prediction '{}'".format(pred_name))
        (features, units,
         types) = self._get_features_of_datasets(datasets, 'prediction_input',
                                                 msg)

        # Check if features were found
        if not features:
            raise ValueError(
                "No features for 'prediction_input' data{} found, 'use_only_"
                "coords_as_features' can only be used when at least one "
                "coordinate for 'use_coords_as_feature' is given".format(msg))

        # Check for wrong options
        if self._cfg.get('accept_only_scalar_data'):
            if 'broadcasted' in types:
                raise TypeError(
                    "The use of 'broadcast_from' is not possible if "
                    "'accept_only_scalar_data' is given")
            if 'coordinate' in types:
                raise TypeError(
                    "The use of 'use_coords_as_feature' is not possible if "
                    "'accept_only_scalar_data' is given")

        # Sort
        sort_idx = np.argsort(features)
        features = np.array(features)[sort_idx]
        units = np.array(units)[sort_idx]
        types = np.array(types)[sort_idx]

        # Return features
        logger.info(
            "Found %i feature(s) (defined in 'prediction_input' data%s)",
            len(features), msg)
        for (idx, feature) in enumerate(features):
            logger.debug("'%s' with units '%s' and type '%s'", feature,
                         units[idx], types[idx])
        return (features, units, types)

    def _get_features_of_datasets(self, datasets, var_type, msg):
        """Extract all features of given datasets."""
        features = []
        units = []
        types = []
        cube = None
        ref_cube = None
        for (tag, datasets_) in group_metadata(datasets, 'tag').items():
            dataset = datasets_[0]
            cube = self._load_cube(dataset)
            if 'broadcast_from' not in dataset:
                ref_cube = cube
            if not self._cfg.get('use_only_coords_as_features'):
                features.append(tag)
                units.append(Unit(dataset['units']))
                if 'broadcast_from' in dataset:
                    types.append('broadcasted')
                else:
                    types.append('regular')

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
        for coord_name in self._cfg.get('use_coords_as_feature', []):
            try:
                coord = ref_cube.coord(coord_name)
            except iris.exceptions.CoordinateNotFoundError:
                raise iris.exceptions.CoordinateNotFoundError(
                    "Coordinate '{}' given in 'use_coords_as_feature' not "
                    "found in '{}' data{}".format(coord_name, msg, var_type))
            features.append(coord_name)
            units.append(coord.units)
            types.append('coordinate')

        return (features, units, types)

    def _get_group_attributes(self):
        """Get all group attributes from `label` datasets."""
        datasets = select_metadata(self._datasets['training'],
                                   var_type='label')
        grouped_datasets = group_metadata(datasets,
                                          'group_attribute',
                                          sort=True)
        group_attributes = list(grouped_datasets.keys())
        if group_attributes != [None]:
            logger.info(
                "Found %i group attribute(s) (defined in 'label' data)",
                len(group_attributes))
            logger.debug(pformat(group_attributes))
        return np.array(group_attributes)

    def _get_label(self):
        """Extract label from training data."""
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
        return (labels[0], units)

    def _get_lime_feature_importance(self, x_input):
        """Get most important feature given by LIME."""
        explainer = self._skater['local_interpreter'].explain_instance(
            x_input, self._clf.predict, num_features=1, num_samples=200)
        return explainer.as_map()[1][0][0]

    def _get_prediction_dict(self, x_data, x_mask, **kwargs):
        """Get prediction output in a dictionary."""
        mask_1d = np.any(x_mask, axis=1)
        if self._cfg['imputation_strategy'] == 'remove':
            x_data = x_data[~mask_1d]
            n_removed = x_mask.shape[0] - x_data.shape[0]
            if n_removed:
                logger.info(
                    "Removed %i prediction input point(s) where "
                    "features were missing'", n_removed)

        # Get prediction
        logger.info("Predicting %i point(s)", x_data.shape[0])
        y_preds = self._clf.predict(x_data, **kwargs)
        if not isinstance(y_preds, (list, tuple)):
            y_preds = [y_preds]

        # Create dictionary
        idx_to_name = {0: 'pred'}
        if 'return_var' in kwargs:
            idx_to_name[1] = 'var'
        elif 'return_cov' in kwargs:
            idx_to_name[1] = 'cov'
        pred_dict = {}
        for (idx, y_pred) in enumerate(y_preds):
            pred_dict[idx_to_name.get(idx, idx)] = y_pred

        # LIME feature importance
        if self._cfg['return_lime_importance']:
            logger.info("Calculating global feature importance using LIME")
            pool = mp.ProcessPool(processes=self._cfg['n_jobs'])
            pred_dict['lime'] = np.array(
                pool.map(self._get_lime_feature_importance, x_data))

        # Transform arrays correctly
        for (pred_type, y_pred) in pred_dict.items():
            if y_pred.ndim == 1 and y_pred.shape[0] != x_mask.shape[0]:
                new_y_pred = np.ma.empty(x_mask.shape[0],
                                         dtype=self._cfg['dtype'])
                new_y_pred[mask_1d] = np.ma.masked
                new_y_pred[~mask_1d] = y_pred
                pred_dict[pred_type] = new_y_pred
            else:
                pred_dict[pred_type] = np.ma.array(y_pred)
            logger.debug("Found prediction type '%s'", pred_type)
        logger.info("Successfully created prediction array with %i point(s)",
                    pred_dict['pred'].size)
        return pred_dict

    def _get_prediction_cubes(self, pred_dict, pred_name, x_cube):
        """Get (multi-dimensional) prediction output."""
        logger.debug("Creating output cubes")
        prediction_cubes = {}
        ref_path = None
        for (pred_type, y_pred) in pred_dict.items():
            if y_pred.size == np.prod(x_cube.shape):
                y_pred = y_pred.reshape(x_cube.shape)
                if (self._cfg['imputation_strategy'] == 'remove'
                        and np.ma.is_masked(x_cube.data)):
                    y_pred = np.ma.array(y_pred,
                                         mask=y_pred.mask | x_cube.data.mask)
                pred_cube = x_cube.copy(data=y_pred)
            else:
                dim_coords = []
                for (dim_idx, dim_size) in enumerate(y_pred.shape):
                    dim_coords.append((iris.coords.DimCoord(
                        np.arange(dim_size, dtype=np.float64),
                        long_name=f'MLR prediction index {dim_idx}',
                        var_name=f'idx_{dim_idx}'), dim_idx))
                pred_cube = iris.cube.Cube(y_pred,
                                           dim_coords_and_dims=dim_coords)
            new_path = self._set_prediction_cube_attributes(
                pred_cube, pred_type, pred_name=pred_name)
            prediction_cubes[new_path] = pred_cube
            io.save_iris_cube(pred_cube, new_path)
            if pred_type == 'pred':
                ref_path = new_path
        return (prediction_cubes, ref_path)

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
                logger.info(
                    "Attribute '%s' of label data is not unique, got values "
                    "%s, using %s for prediction cubes", attr, attrs,
                    properties[attr])
        return properties

    def _get_reference_cube(self, datasets, var_type, text=None):
        """Get reference cube for `datasets`."""
        msg = '' if text is None else text
        tags = self.classes['features'][np.where(
            self.classes['features_types'] == 'regular')]
        for tag in tags:
            dataset = self._check_dataset(datasets, var_type, tag, msg)
            if dataset is not None:
                ref_cube = self._load_cube(dataset)
                logger.debug(
                    "For var_type '%s'%s, use reference cube with tag '%s'",
                    var_type, msg, tag)
                logger.debug(ref_cube)
                return ref_cube
        raise ValueError(
            "No {} data{} without the option 'broadcast_from' found".format(
                var_type, msg))

    def _get_x_data_for_group(self, datasets, var_type, group_attr=None):
        """Get x data for a group of datasets."""
        msg = '' if group_attr is None else " for '{}'".format(group_attr)
        ref_cube = self._get_reference_cube(datasets, var_type, msg)
        shape = (np.prod(ref_cube.shape,
                         dtype=np.int), len(self.classes['features']))
        attr_data = np.ma.empty(shape, dtype=self._cfg['dtype'])

        # Iterate over all features
        for (idx, tag) in enumerate(self.classes['features']):
            if self.classes['features_types'][idx] != 'coordinate':
                dataset = self._check_dataset(datasets, var_type, tag, msg)
                if dataset is None:
                    new_data = np.ma.masked
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
            attr_data[:, idx] = new_data

        # Return data and reference cube
        return (attr_data, ref_cube)

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
        x_dummy = np.ones((1, self.classes['features'].size),
                          dtype=self._cfg['dtype'])
        try:
            self._clf.predict(x_dummy)
        except NotFittedError:
            return False
        return True

    def _is_ready_for_plotting(self):
        """Check if the class is ready for plotting."""
        if not self._is_fitted():
            logger.error("Plotting not possible, MLR model is not fitted yet")
            return False
        if not self._cfg['write_plots']:
            logger.debug("Plotting not possible, 'write_plots' is set to "
                         "'False' in user configuration file")
            return False
        return True

    def _load_cfg_parameters(self):
        """Load parameters for classifier from recipe."""
        parameters = self._cfg.get('parameters', {})
        logger.debug("Found parameter(s) %s in recipe", parameters)
        if 'verbose' in getfullargspec(self._CLF_TYPE).args:
            log_level = {'debug': 1, 'info': 1}
            parameters.setdefault('verbose',
                                  log_level.get(self._cfg['log_level'], 0))
            logger.debug("Set verbosity of classifier to %i",
                         parameters['verbose'])
        return parameters

    def _load_classes(self):
        """Populate self.classes and check for errors."""
        self._classes['group_attributes'] = self._get_group_attributes()
        (self._classes['features'], self._classes['features_units'],
         self._classes['features_types']) = self._get_features()
        (self._classes['label'],
         self._classes['label_units']) = self._get_label()

    def _load_cube(self, dataset):
        """Load iris cube, check data type and convert units if desired."""
        cube = iris.load_cube(dataset['filename'])

        # Check dtype
        if not np.issubdtype(cube.dtype, np.number):
            raise TypeError(
                "Data type of cube loaded from '{}' is '{}', at "
                "the moment only numerical data is supported".format(
                    dataset['filename'], cube.dtype))

        # Convert dtypes
        cube_data = cube.core_data()
        cube = cube.copy(
            data=cube_data.astype(self._cfg['dtype'], casting='same_kind'))
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

    def _load_input_datasets(self, **metadata):
        """Load input datasets (including ancestors)."""
        input_datasets = copy.deepcopy(list(self._cfg['input_data'].values()))
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
        self._datasets['training'] = self._group_by_attributes(
            training_datasets)
        self._datasets['prediction'] = self._group_prediction_datasets(
            prediction_datasets)
        logger.debug("Found training data:")
        logger.debug(pformat(self._datasets['training']))
        logger.debug("Found prediction data:")
        logger.debug(pformat(self._datasets['prediction']))

    def _load_skater_interpreters(self):
        """Load :mod:`skater` interpretation modules."""
        x_train = np.copy(self._data['x_train'])
        y_train = np.copy(self._data['y_train'])
        if self._cfg['imputation_strategy'] != 'remove':
            x_train = self._clf.named_steps['imputer'].transform(x_train)

        # Interpreters
        verbosity = (True if self._cfg['log_level'] == 'debug'
                     and not self._cfg['return_lime_importance'] else False)
        self._skater['global_interpreter'] = Interpretation(
            x_train,
            training_labels=y_train,
            feature_names=self.classes['features'])
        logger.debug("Loaded global skater interpreter with new training data")
        self._skater['local_interpreter'] = LimeTabularExplainer(
            x_train,
            mode='regression',
            training_labels=y_train,
            feature_names=self.classes['features'],
            verbose=verbosity,
            class_names=[self.classes['label']])
        logger.debug(
            "Loaded local skater interpreter (LIME) with new training data")

        # Model
        example_size = min(y_train.size, 20)
        self._skater['model'] = InMemoryModel(
            self._clf.predict,
            feature_names=self.classes['features'],
            examples=x_train[:example_size],
            model_type='regressor',
        )
        logger.debug("Loaded skater model with new classifier")

    def _load_training_data(self):
        """Load training data (features/labels)."""
        (self._data['x_data'],
         self._data['y_data']) = self._extract_features_and_labels()
        logger.info("Loaded %i input data point(s)", self._data['y_data'].size)
        self.reset_training_data()

    def _postprocess_cube(self, cube, return_cov_weights=False, **kwargs):
        """Postprocess single prediction cube."""
        cfg = self._cfg['prediction_pp']
        calc_weights = cfg.get('area_weights', True)
        cov_weights = None
        if all([calc_weights, return_cov_weights, 'return_cov' in kwargs]):
            cov_weights = self._get_area_weights(cube).ravel()
            cov_weights = cov_weights[~np.ma.getmaskarray(cube.data).ravel()]
        ops = {'mean': iris.analysis.MEAN, 'sum': iris.analysis.SUM}

        # Perform desired postprocessing operations
        n_points_mean = None
        old_size = np.prod(cube.shape, dtype=np.int)
        for (op_type, iris_op) in ops.items():
            if not cfg.get(op_type):
                continue
            logger.debug("Calculating %s for coordinates %s", op_type,
                         cfg[op_type])
            weights = None
            if all([
                    calc_weights, 'latitude' in cfg[op_type],
                    'longitude' in cfg[op_type]
            ]):
                weights = self._get_area_weights(cube)
            cube = cube.collapsed(cfg[op_type], iris_op, weights=weights)
            if op_type == 'mean':
                new_size = np.prod(cube.shape, dtype=np.int)
                n_points_mean = int(old_size / new_size)
            elif op_type == 'sum':
                cube.units *= Unit('m2')

        # Units conversion
        if cfg.get('units'):
            self._convert_units_in_cube(cube,
                                        cfg['units'],
                                        text='postprocessed prediction output')

        # Weights for covariance matrix
        if cov_weights is not None:
            logger.debug("Calculating covariance weights (memory-intensive)")
            cov_weights = np.outer(cov_weights, cov_weights)
            if n_points_mean is not None:
                cov_weights /= n_points_mean**2
        return (cube, cov_weights)

    def _postprocess_predictions(self, predictions, ref_path=None, **kwargs):
        """Postprocess prediction cubes if desired."""
        if not self._cfg['prediction_pp']:
            return
        if ref_path is None:
            return
        logger.info("Postprocessing prediction output using %s",
                    self._cfg['prediction_pp'])
        logger.debug("Using reference cube at '%s'", ref_path)

        # Process reference cube
        ref_cube = predictions[ref_path]
        ref_shape = ref_cube.shape
        (ref_cube,
         cov_weights) = self._postprocess_cube(ref_cube,
                                               return_cov_weights=True,
                                               **kwargs)

        # Process other cubes
        pp_cubes = [ref_cube]
        for (path, cube) in predictions.items():
            if path == ref_path or cube.attributes.get('skip_for_pp'):
                continue
            if cube.shape == ref_shape:
                (cube, _) = self._postprocess_cube(cube)
            else:
                if cov_weights is not None:
                    if cube.shape != cov_weights.shape:
                        logger.error(
                            "Cannot postprocess all prediction cubes, "
                            "expected shapes %s or %s (for covariance), got "
                            "%s", ref_shape, cov_weights.shape, cube.shape)
                        continue
                logger.debug("Collapsing covariance matrix")
                cube = cube.collapsed(cube.coords(dim_coords=True),
                                      iris.analysis.SUM,
                                      weights=cov_weights)
                cube.units *= Unit('m4')

                # Units conversion
                new_units = self._cfg['prediction_pp'].get('units')
                if new_units:
                    new_units = self._units_power(Unit(new_units), 2)
                    self._convert_units_in_cube(
                        cube, new_units, text='postprocessed covariance')
            pp_cubes.append(cube)

        # Save cubes
        new_path = ref_path.replace('.nc', '_pp.nc')
        io.save_iris_cube(pp_cubes, new_path)

    def _remove_missing_features(self, x_data, y_data=None):
        """Remove missing values in the features data (if desired)."""
        if self._cfg['imputation_strategy'] != 'remove':
            new_x_data = x_data.filled(np.nan)
            new_y_data = None if y_data is None else y_data.filled(np.nan)
        else:
            mask = np.any(np.ma.getmaskarray(x_data), axis=1)
            new_x_data = x_data.filled()[~mask]
            new_y_data = None if y_data is None else y_data.filled()[~mask]
            n_removed = x_data.shape[0] - new_x_data.shape[0]
            if n_removed:
                msg = ('Removed %i training point(s) where features were '
                       'missing')
                if self._cfg.get('accept_only_scalar_data'):
                    removed_groups = self.classes['group_attributes'][mask]
                    msg += ' ({})'.format(removed_groups)
                    self.classes['group_attributes'] = (
                        self.classes['group_attributes'][~mask])
                logger.info(msg, n_removed)
        return (new_x_data, new_y_data)

    def _save_csv_file(self,
                       data_type,
                       filename,
                       is_prediction=False,
                       pred_name=None):
        """Save CSV file."""
        if data_type not in self._data:
            return
        if is_prediction:
            if pred_name not in self._data[data_type]:
                return
            csv_data = self._data[data_type][pred_name]
        else:
            csv_data = self._data[data_type]

        # Filename and path
        if filename is None:
            if pred_name is None:
                filename = '{}.csv'.format(data_type)
            else:
                filename = '{}_{}.csv'.format(data_type, pred_name)
        path = os.path.join(self._cfg['mlr_work_dir'], filename)

        # File Header
        if 'x_' in data_type:
            sub_txt = 'features: {}'.format(self.classes['features'])
        else:
            sub_txt = 'label: {}'.format(self.classes['label'])
        header = ('{} with shape {}\n{:d}: number of observations)\n{}\nNote:'
                  'nan indicates missing values').format(
                      data_type, csv_data.shape, csv_data.shape[0], sub_txt)

        # Save file
        np.savetxt(path, csv_data, delimiter=',', header=header)
        logger.info("Wrote %s", path)

    def _set_prediction_cube_attributes(self, cube, pred_type, pred_name=None):
        """Set the attributes of the prediction cube."""
        cube.attributes = {
            'classifier': str(self._CLF_TYPE),
            'description': 'MLR model prediction',
            'tag': self.classes['label'],
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
        suffix = pred_type
        if isinstance(suffix, int):
            cube.var_name += '_{:d}'.format(suffix)
            cube.long_name += ' {:d}'.format(suffix)
        elif suffix == 'pred':
            pass
        elif suffix in ('var', 'cov'):
            cube.var_name += f'_{suffix}'
            cube.long_name += (' (variance)'
                               if suffix == 'var' else ' (covariance)')
            cube.units = self._units_power(cube.units, 2)
        elif 'error_estim' in suffix:
            cube.var_name += f'_{suffix}'
            cube.long_name += (' (error estimation using {})'.format(
                'cross-validation' if 'cv' in
                suffix else 'holdout test data set'))
        elif suffix == 'lime':
            cube.var_name = 'lime_feature_importance'
            cube.long_name = (f"Most important feature for predicting "
                              f"'{self.classes['label']}' given by LIME")
            cube.units = Unit('no_unit')
            cube.attributes.update({
                'features':
                pformat(dict(enumerate(self.classes['features']))),
                'skip_for_pp':
                1,
            })
        else:
            logger.warning("Got unknown prediction type '%s'", pred_type)

        # Get new path
        pred_str = '' if pred_name is None else f'_{pred_name}'
        root_str = ('' if self._cfg['root_dir'] == '' else
                    f"{self._cfg['root_dir']}_")
        filename = f'{root_str}prediction{pred_str}_{suffix}.nc'
        new_path = os.path.join(self._cfg['mlr_work_dir'], filename)
        cube.attributes['filename'] = new_path
        return new_path

    @staticmethod
    def _convert_units_in_cube(cube, new_units, text=None):
        """Convert units of cube if possible."""
        msg = '' if text is None else f' of {text}'
        if isinstance(new_units, str):
            new_units = Unit(new_units)
        new_units_name = (new_units.symbol
                          if new_units.origin is None else new_units.origin)
        logger.debug("Converting units%s from '%s' to '%s'", msg,
                     cube.units.symbol, new_units_name)
        try:
            cube.convert_units(new_units)
        except ValueError:
            logger.warning("Units conversion%s from '%s' to '%s' failed", msg,
                           cube.units.symbol, new_units_name)

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
    def _get_area_weights(cube):
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
                "Calculation of area weights for prediction cubes failed")
            logger.warning(str(exc))
        return area_weights

    @staticmethod
    def _get_coordinate_data(ref_cube, var_type, tag, text=None):
        """Get coordinate variable `ref_cube` which can be used as x data."""
        msg = '' if text is None else text
        try:
            coord = ref_cube.coord(tag)
        except iris.exceptions.CoordinateNotFoundError:
            raise iris.exceptions.CoordinateNotFoundError(
                "Coordinate '{}' given in 'use_coords_as_feature' not "
                "found in reference cube for {}{}".format(tag, var_type, msg))
        coord_array = np.ma.array(coord.points)
        coord_dims = ref_cube.coord_dims(coord)
        if coord_dims == ():
            logger.warning(
                "Coordinate '%s' is scalar, including it as feature does not "
                "add any information to the model (array is constant)", tag)
        else:
            new_axis_pos = np.delete(np.arange(ref_cube.ndim), coord_dims)
            for idx in new_axis_pos:
                coord_array = np.ma.expand_dims(coord_array, idx)
        mask = coord_array.mask
        coord_array = np.broadcast_to(coord_array, ref_cube.shape, subok=True)
        coord_array.mask = np.broadcast_to(mask, ref_cube.shape)
        logger.debug("Added coordinate %s '%s'%s", var_type, tag, msg)
        return coord_array.ravel()

    @staticmethod
    def _get_cube_data(cube):
        """Get data from cube."""
        if cube.shape == ():
            return cube.data
        return cube.data.ravel()

    @staticmethod
    def _group_prediction_datasets(datasets):
        """Group prediction datasets (use `prediction_name` key)."""
        for dataset in datasets:
            dataset['group_attribute'] = None
        return group_metadata(datasets, 'prediction_name')

    @staticmethod
    def _remove_missing_labels(x_data, y_data):
        """Remove missing values in the label data."""
        mask = np.ma.getmaskarray(y_data)
        new_x_data = x_data[~mask]
        new_y_data = y_data[~mask]
        diff = y_data.size - new_y_data.size
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
