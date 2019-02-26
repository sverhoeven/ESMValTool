"""Base class for MLR models."""

import copy
import importlib
import logging
import os
from pprint import pformat

import cf_units
import iris
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, LeaveOneOut, train_test_split
from sklearn.preprocessing import StandardScaler

from esmvaltool.diag_scripts import mlr
from esmvaltool.diag_scripts.shared import (
    io,
    group_metadata,
    plot,
    select_metadata,
)

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
    grid_search_cv_param_grid : dict or list of dict, optional
        Parameters (keys) and ranges (values) for exhaustive parameter search
        using cross-validation.
    grid_search_cv_kwargs : dict, optional
        Keyword arguments for the grid search cross-validation, see
        https://scikit-learn.org/stable/modules/generated/
        sklearn.model_selection.GridSearchCV.html.
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
    mlr_model : str, optional (default: 'gbr')
        Regression model which is used. Allowed models are child classes of
        this base class given in :mod:`esmvaltool.diag_scripts.mlr.models`.

    normalize_data : dict, optional
        Specify tags (keys) and constants (`float` or `'mean'`, value) for
        normalization of data. This is done by dividing the data by the
        given constant and might be necessary when raw data is very small or
        large which leads to large errors due to finite machine precision.
    parameters : dict, optional
        Parameters used in the classifier.
    predict_kwargs : dict, optional
        Optional keyword arguments for the `clf.predict()` function.
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
        plt.style.use(plot.get_path_to_mpl_style())

        # Private members
        self._cfg = cfg
        self._clf = None
        self._data = {}
        self._data['x_pred'] = {}
        self._data['y_pred'] = {}
        self._datasets = {}
        self._transformer = {}
        self._transformer['imputer'] = mlr.Imputer(
            strategy=self._cfg.get('imputation_strategy', 'remove'),
            fill_value=self._cfg.get('imputation_constant'))
        for scaler in ('x_scaler', 'y_scaler'):
            self._transformer[scaler] = StandardScaler(
                with_mean=self._cfg.get('standardize_data', True),
                with_std=self._cfg.get('standardize_data', True))

        # Public members
        self.classes = {}
        self.parameters = self._load_parameters()

        # Adapt output directories
        if root_dir is None:
            root_dir = ''
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

        # Log successful initialization
        msg = ('' if not self.parameters else ' with parameters {} found in '
               'recipe'.format(self.parameters))
        logger.info("Initialized MRT model%s", msg)

    def export_prediction_data(self, filename=None):
        """Export all prediction data contained in `self._data`.

        Parameters
        ----------
        filename : str, optional (default: '{data_type}_{pred_name}.csv')
            Name of the exported files.

        """
        for data_type in ('x_pred', 'y_pred'):
            for pred_name in self._data[data_type]:
                self._save_csv_file(
                    data_type,
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
        for data_type in ('x_raw', 'x_data', 'x_train', 'x_test', 'y_raw',
                          'y_data', 'y_train', 'y_test'):
            self._save_csv_file(data_type, filename)

    def fit(self, **parameters):
        """Initialize and fit the MLR model(s).

        Parameters
        ----------
        parameters : fit parameters, optional
            Parameters to initialize and fit the MLR model(s). Overwrites
            default and recipe settings.

        """
        if not self._clf_is_valid(text='Fitting MLR model'):
            return
        logger.info("Fitting MLR model with classifier %s", self._CLF_TYPE)
        if parameters:
            logger.info(
                "Using additional parameter(s) %s given in fit() function",
                parameters)

        # Create MLR model with desired parameters and fit it
        params = dict(self.parameters)
        params.update(parameters)
        self._clf = self._CLF_TYPE(**params)  # noqa
        self._clf.fit(self._data['x_train'], self._data['y_train'])
        self.parameters = self._clf.get_params()
        logger.info(
            "Successfully fitted '%s' model on %i training point(s) "
            "with parameter(s) %s", self._CLF_TYPE.__name__,
            self._data['y_train'].size, self.parameters)

    def grid_search_cv(self, param_grid=None, **kwargs):
        """Perform exhaustive parameter search using cross-validation.

        Parameters
        ----------
        param_grid : dict or list of dict, optional
            Parameter names (keys) and ranges (values) for the search.
            Overwrites default and recipe settings.
        **kwargs : keyword arguments, optional
            Additional options for the `GridSearchCV` class. See
            https://scikit-learn.org/stable/modules/generated/
            sklearn.model_selection.GridSearchCV.html. Overwrites default and
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
            "Performing exhaustive grid search cross-validation with "
            "classifier %s and parameter grid %s", self._CLF_TYPE,
            parameter_grid)
        additional_args = dict(self._cfg.get('grid_search_cv_kwargs', {}))
        additional_args.update(kwargs)
        if additional_args:
            logger.info(
                "Using additional keyword argument(s) %s given in "
                "recipe and grid_search_cv() function", additional_args)
            if additional_args.get('cv', '').lower() == 'loo':
                additional_args['cv'] = LeaveOneOut()

        # Create MLR model with desired parameters and fit it
        clf = GridSearchCV(
            self._CLF_TYPE(**self.parameters), parameter_grid,
            **additional_args)
        clf.fit(self._data['x_train'], self._data['y_train'])
        self.parameters.update(clf.best_params_)
        if hasattr(clf, 'best_estimator_'):
            self._clf = clf.best_estimator_
        else:
            self._clf = self._CLF_TYPE(**self.parameters)
            self._clf.fit(self._data['x_train'], self._data['y_train'])
        self.parameters = self._clf.get_params()
        logger.info(
            "Exhaustive grid search successful, found best parameter(s) %s",
            clf.best_params_)
        logger.debug("CV results:")
        logger.debug(pformat(clf.cv_results_))
        logger.info(
            "Successfully fitted '%s' model on %i training point(s) "
            "with parameters %s", self._CLF_TYPE.__name__,
            self._data['y_train'].size, self.parameters)

    def plot_scatterplots(self, filename=None):
        """Plot scatterplots label vs. feature for every feature.

        Parameters
        ----------
        filename : str, optional (default: 'scatterplot_{feature}')
            Name of the plot file.

        """
        if not self._is_ready_for_plotting():
            return
        if filename is None:
            filename = 'scatterplot_{feature}'
        (_, axes) = plt.subplots()

        # Plot scatterplot for every feature
        x_data = self._transformer['x_scaler'].inverse_transform(
            self._data['x_data'])
        y_data = self._transformer['y_scaler'].inverse_transform(
            np.expand_dims(self._data['y_data'], axis=-1))
        y_data = np.squeeze(y_data)
        for (f_idx, feature) in enumerate(self.classes['features']):
            if self._cfg.get('accept_only_scalar_data'):
                for (g_idx, group_attr) in enumerate(
                        self.classes['group_attributes']):
                    axes.scatter(
                        x_data[g_idx, f_idx], y_data[g_idx], label=group_attr)
                for (pred_name, x_pred) in self._data['x_pred'].items():
                    x_pred = self._transformer['x_scaler'].inverse_transform(
                        x_pred)
                    axes.axvline(
                        x_pred[0, f_idx],
                        linestyle='--',
                        color='black',
                        label=('Observation'
                               if pred_name is None else pred_name))
                legend = axes.legend(
                    loc='center left',
                    ncol=2,
                    bbox_to_anchor=[1.05, 0.5],
                    borderaxespad=0.0)
            else:
                axes.plot(x_data[:, f_idx], y_data, '.')
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
            plt.savefig(
                new_path,
                orientation='landscape',
                bbox_inches='tight',
                additional_artists=[legend])
            logger.info("Wrote %s", new_path)
            axes.clear()
        plt.close()

    def predict(self, **kwargs):
        """Perform prediction using the MLR model(s) and write netcdf.

        Parameters
        ----------
        **kwargs : keyword arguments, optional
            Additional options for the `self._clf.predict()` function.
            Overwrites default and recipe settings.

        Returns
        -------
        dict
            Prediction names (keys) and data (values).

        """
        if not self._is_fitted():
            logger.error("Prediction not possible because the model is not "
                         "fitted yet, call fit() first")
            return None
        predict_kwargs = dict(self._cfg.get('predict_kwargs', {}))
        predict_kwargs.update(kwargs)
        if predict_kwargs:
            logger.info(
                "Using additional keyword argument(s) %s for predict() "
                "function", predict_kwargs)

        # Iterate over predictions
        predictions = {}
        if not self._datasets['prediction']:
            logger.error("Prediction not possible, no 'prediction_input' "
                         "datasets given")
        for (pred_name, datasets) in self._datasets['prediction'].items():
            if pred_name is None:
                logger.info("Started prediction")
                filename = 'prediction.nc'
            else:
                logger.info("Started prediction for prediction %s", pred_name)
                filename = 'prediction_{}.nc'.format(pred_name)
            (x_pred, cube) = self._extract_prediction_input(datasets)
            y_pred = self._clf.predict(x_pred, **kwargs)

            # Save data in arrays
            self._data['x_pred'][pred_name] = np.copy(x_pred)
            self._data['y_pred'][pred_name] = np.copy(y_pred)
            y_pred = self._transformer['y_scaler'].inverse_transform(
                np.expand_dims(y_pred, axis=-1))
            y_pred = np.squeeze(y_pred)
            predictions[pred_name] = np.copy(y_pred)

            # Save data into cubes
            pred_cube = cube.copy(data=y_pred.reshape(cube.shape))
            if self._transformer['imputer'].strategy == 'remove':
                if np.ma.is_masked(cube.data):
                    pred_cube.data = np.ma.array(
                        pred_cube.data, mask=cube.data.mask)
            new_path = os.path.join(self._cfg['mlr_work_dir'], filename)
            self._set_prediction_cube_attributes(
                pred_cube, new_path, prediction_name=pred_name)
            io.save_iris_cube(pred_cube, new_path)
            logger.info("Successfully predicted %i point(s)", y_pred.size)

        return predictions

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
        self._reset_input_data()
        (self._data['x_train'], self._data['x_test'], self._data['y_train'],
         self._data['y_test']) = train_test_split(
             self._data['x_raw'], self._data['y_raw'], test_size=test_size)
        self._preprocess_data()
        logger.info("Used %i%% of the input data as test data (%i point(s))",
                    int(test_size * 100), self._data['y_test'].size)

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
                if cube.coords(dim_coords=True) != expected_coords:
                    cube_coords = [
                        '{}, shape {}'.format(coord.name(), coord.shape)
                        for coord in cube.coords(dim_coords=True)
                    ]
                    coords = [
                        '{}, shape {}'.format(coord.name(), coord.shape)
                        for coord in expected_coords
                    ]
                    raise ValueError(
                        "Expected field with coordinates {}{}, got {}. "
                        "Consider regridding, pre-selecting data at class "
                        "initialization using '**metadata' or the options "
                        "'broadcast_from' or 'group_datasets_by_attributes'".
                        format(coords, msg, cube_coords))

    def _check_dataset(self, datasets, var_type, tag, text=None):
        """Check if `tag` of datasets exists."""
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
        if units != cf_units.Unit(datasets[0]['units']):
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

    def _extract_features_and_labels(self):
        """Extract feature and label data points from training data."""
        datasets = self._datasets['training']
        (x_data, _) = self._extract_x_data(datasets, 'feature')
        y_data = self._extract_y_data(datasets)

        # Handle missing values in labels
        logger.debug("Found %i input data point(s)", x_data.shape[0])
        (x_data, y_data) = self._remove_missing_labels(x_data, y_data)

        # Check sizes
        if x_data.shape[0] != y_data.size:
            raise ValueError(
                "Sizes of features and labels do not match, got {:d} points "
                "for the features and {:d} points for the label".format(
                    x_data.shape[0], y_data.size))

        return (x_data, y_data)

    def _extract_prediction_input(self, datasets):
        """Extract prediction input data points `datasets`."""
        (x_data, prediction_input_cube) = self._extract_x_data(
            datasets, 'prediction_input')

        # Impute data (point removing is done later in cube mask)
        if self._transformer['imputer'].strategy == 'remove':
            x_data = x_data.filled(np.ma.mean(x_data))
        else:
            (x_data, _) = self._impute_data(x_data, text='prediction input')

        # Scale data
        (x_data, _) = self._scale_data(x_data, text='prediction_input')

        return (x_data, prediction_input_cube)

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
            attr_datasets = select_metadata(
                datasets, group_attribute=group_attr)
            if group_attr is not None:
                logger.info("Loading '%s' data of '%s'", var_type, group_attr)
            msg = '' if group_attr is None else " for '{}'".format(group_attr)
            if not attr_datasets:
                raise ValueError("No '{}' data{} found".format(var_type, msg))
            (attr_data, cube) = self._get_x_data_for_group(
                attr_datasets, var_type, group_attr)

            # Append data
            if x_data is None:
                x_data = attr_data
            else:
                x_data = np.ma.vstack((x_data, attr_data))

        return (x_data, cube)

    def _extract_y_data(self, datasets):
        """Extract y data (labels) from `datasets`."""
        datasets = select_metadata(datasets, var_type='label')
        y_data = np.ma.array([])
        for group_attr in self.classes['group_attributes']:
            if group_attr is not None:
                logger.info("Loading 'label' data of '%s'", group_attr)
            msg = '' if group_attr is None else " for '{}'".format(group_attr)
            datasets_ = select_metadata(datasets, group_attribute=group_attr)
            dataset = self._check_dataset(datasets_, 'label',
                                          self.classes['label'], msg)
            cube = self._load_cube(dataset['filename'])
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
        cube_to_broadcast = self._load_cube(dataset['filename'])
        data_to_broadcast = np.ma.array(cube_to_broadcast.data)
        try:
            new_axis_pos = np.delete(
                np.arange(len(target_shape)), dataset['broadcast_from'])
        except IndexError:
            raise ValueError(
                "Broadcasting to shape {} failed{}, index out of bounds".
                format(target_shape, msg))
        logger.info("Broadcasting %s from %s to %s", msg,
                    data_to_broadcast.shape, target_shape)
        for idx in new_axis_pos:
            data_to_broadcast = np.ma.expand_dims(data_to_broadcast, idx)
        mask = data_to_broadcast.mask
        data_to_broadcast = np.broadcast_to(
            data_to_broadcast, target_shape, subok=True)
        data_to_broadcast.mask = np.broadcast_to(mask, target_shape)
        new_cube = ref_cube.copy(data_to_broadcast)
        for idx in dataset['broadcast_from']:
            new_coord = new_cube.coord(dimensions=idx)
            new_coord.points = cube_to_broadcast.coord(new_coord).points
        logger.debug("Added broadcasted %s", msg)
        return new_cube

    def _get_coordinate_data(self, ref_cube, var_type, tag, text=None):
        """Get coordinate variable `ref_cube` which can be used as x data."""
        msg = '' if text is None else text
        try:
            coord = ref_cube.coord(tag)
        except iris.exceptions.CoordinateNotFoundError:
            raise iris.exceptions.CoordinateNotFoundError(
                "Coordinate '{}' given in 'use_coords_as_feature' not "
                "found in reference cube for {}{}".format(tag, var_type, msg))
        coord_array = np.ma.array(coord.points)
        new_axis_pos = np.delete(
            np.arange(ref_cube.ndim), ref_cube.coord_dims(coord))
        for idx in new_axis_pos:
            coord_array = np.ma.expand_dims(coord_array, idx)
        mask = coord_array.mask
        coord_array = np.broadcast_to(coord_array, ref_cube.shape, subok=True)
        coord_array.mask = np.broadcast_to(mask, ref_cube.shape)
        logger.debug("Added coordinate %s '%s'%s", var_type, tag, msg)
        return coord_array.ravel()

    def _get_cube_data(self, cube):
        """Get data from cube."""
        if cube.shape == ():
            return cube.data
        return cube.data.ravel()

    def _get_features(self):
        """Extract all features from the `prediction_input` datasets."""
        pred_name = list(self._datasets['prediction'].keys())[0]
        datasets = self._datasets['prediction'][pred_name]
        msg = ('' if pred_name is None else
               " for prediction '{}'".format(pred_name))
        (features, units, types) = self._get_features_of_datasets(
            datasets, 'prediction_input', msg)

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
        grouped_datasets = group_metadata(datasets, 'tag')
        for (tag, data) in grouped_datasets.items():
            cube = iris.load_cube(data[0]['filename'])
            if 'broadcast_from' not in data[0]:
                ref_cube = cube
            if not self._cfg.get('use_only_coords_as_features'):
                features.append(tag)
                units.append(cube.units)
                if 'broadcast_from' in data[0]:
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
        datasets = select_metadata(
            self._datasets['training'], var_type='label')
        grouped_datasets = group_metadata(
            datasets, 'group_attribute', sort=True)
        group_attributes = list(grouped_datasets.keys())
        if group_attributes != [None]:
            logger.info(
                "Found %i group attribute(s) (defined in 'label' data)",
                len(group_attributes))
            logger.debug(pformat(group_attributes))
        return group_attributes

    def _get_label(self):
        """Extract label from training data."""
        datasets = select_metadata(
            self._datasets['training'], var_type='label')
        if not datasets:
            raise ValueError("No 'label' datasets given")
        grouped_datasets = group_metadata(datasets, 'tag')
        labels = list(grouped_datasets.keys())
        if len(labels) > 1:
            raise ValueError(
                "Expected unique label tag, got {}".format(labels))
        cube = iris.load_cube(datasets[0]['filename'])
        logger.info(
            "Found label '%s' with units '%s' (defined in 'label' "
            "data)", labels[0], cube.units)
        return (labels[0], cube.units)

    def _get_reference_cube(self, datasets, var_type, text=None):
        """Get reference cube for `datasets`."""
        msg = '' if text is None else text
        tags = self.classes['features'][np.where(
            self.classes['features_types'] == 'regular')]
        for tag in tags:
            dataset = self._check_dataset(datasets, var_type, tag, msg)
            if dataset is not None:
                ref_cube = self._load_cube(dataset['filename'])
                logger.debug("For %s%s, use reference cube", var_type, msg)
                logger.debug(ref_cube)
                return ref_cube
        raise ValueError(
            "No {} data{} without the option 'broadcast_from' found".format(
                var_type, msg))

    def _get_x_data_for_group(self, datasets, var_type, group_attr=None):
        """Get x data for a group of datasets."""
        msg = '' if group_attr is None else " for '{}'".format(group_attr)
        ref_cube = self._get_reference_cube(datasets, var_type, msg)
        shape = (np.prod(ref_cube.shape, dtype=np.int),
                 len(self.classes['features']))
        attr_data = np.ma.empty(shape)

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
                        cube = self._load_cube(dataset['filename'])
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

    def _group_prediction_datasets(self, datasets):
        """Group prediction datasets (use `prediction_name` key)."""
        for dataset in datasets:
            dataset['group_attribute'] = None
        return group_metadata(datasets, 'prediction_name')

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

    def _impute_data(self, x_data, y_data=None, text=None):
        """Impute missing values in the feature data."""
        (new_x_data, new_y_data,
         n_imputes) = self._transformer['imputer'].transform(x_data, y_data)
        strategy = self._transformer['imputer'].strategy
        if strategy == 'remove':
            strategy_str = 'removing point(s)'
            if self._cfg.get('accept_only_scalar_data'):
                mask = self._transformer['imputer'].mask_
                removed_groups = self.classes['group_attributes_raw'][mask]
                if removed_groups:
                    strategy += ' {}'.format(removed_groups)
                self.classes['group_attributes'] = np.copy(
                    self.classes['group_attributes_raw'][~mask])
        else:
            strategy_str = 'setting them to {} ({})'.format(
                strategy, self._transformer['imputer'].statistics_)
        if n_imputes:
            msg = '' if text is None else ' for {} data'.format(text)
            logger.info("Imputed %i missing feature(s)%s", n_imputes, msg)
            logger.debug("by %s", strategy_str)
        return (new_x_data, new_y_data)

    def _is_fitted(self):
        """Check if the MLR models are fitted."""
        return bool(self._clf is not None)

    def _is_ready_for_plotting(self):
        """Check if the class is ready for plotting."""
        if not self._is_fitted():
            logger.error("Plotting not possible, MLR model is not fitted yet, "
                         "call fit() first")
            return False
        if not self._cfg['write_plots']:
            logger.debug("Plotting not possible, 'write_plots' is set to "
                         "'False' in user configuration file")
            return False
        return True

    def _load_classes(self):
        """Populate self.classes and check for errors."""
        self.classes['group_attributes'] = self._get_group_attributes()
        self.classes['group_attributes_raw'] = np.copy(
            self.classes['group_attributes'])
        (self.classes['features'], self.classes['features_units'],
         self.classes['features_types']) = self._get_features()
        (self.classes['label'],
         self.classes['label_units']) = self._get_label()

    def _load_cube(self, path):
        """Load iris cube and check data type."""
        cube = iris.load_cube(path)
        if not np.issubdtype(cube.dtype, np.number):
            raise TypeError(
                "Data type of cube loaded from '{}' is '{}', at "
                "the moment only numerical data is supported".format(
                    path, cube.dtype))
        return cube

    def _load_input_datasets(self, **metadata):
        """Load input datasets (including ancestors)."""
        input_datasets = copy.deepcopy(list(self._cfg['input_data'].values()))
        input_datasets.extend(self._get_ancestor_datasets())
        mlr.datasets_have_mlr_attributes(
            input_datasets, log_level='warning', mode='only_var_type')

        # Extract features and labels
        feature_datasets = select_metadata(input_datasets, var_type='feature')
        label_datasets = select_metadata(input_datasets, var_type='label')
        feature_datasets = select_metadata(feature_datasets, **metadata)
        label_datasets = select_metadata(label_datasets, **metadata)
        if metadata:
            logger.info("Only considered features and labels matching %s",
                        metadata)

        # Prediction datasets
        prediction_datasets = select_metadata(
            input_datasets, var_type='prediction_input')
        training_datasets = feature_datasets + label_datasets

        # Check datasets
        msg = ("At least one '{}' dataset does not have necessary MLR "
               "attributes")
        if not mlr.datasets_have_mlr_attributes(
                training_datasets, log_level='error'):
            raise ValueError(msg.format('training'))
        if not mlr.datasets_have_mlr_attributes(
                prediction_datasets, log_level='error'):
            raise ValueError(msg.format('prediction'))

        # Check if data was found
        if not training_datasets:
            msg = ' for metadata {}'.format(metadata) if metadata else ''
            raise ValueError(
                "No training data (features/labels){} found".format(msg))

        # Set datasets
        self._datasets['training'] = self._group_by_attributes(
            training_datasets)
        self._datasets['prediction'] = self._group_prediction_datasets(
            prediction_datasets)
        logger.debug("Found training data:")
        logger.debug(pformat(self._datasets['training']))
        logger.debug("Found prediction data:")
        logger.debug(pformat(self._datasets['prediction']))

    def _load_parameters(self):
        """Load parameters for classifier from recipe."""
        parameters = self._cfg.get('parameters', {})
        logger.debug("Found parameter(s) %s in recipe", parameters)
        return parameters

    def _load_training_data(self):
        """Load training data (features/labels)."""
        (self._data['x_raw'],
         self._data['y_raw']) = self._extract_features_and_labels()
        self._reset_input_data()
        self._data['x_train'] = np.ma.copy(self._data['x_data'])
        self._data['y_train'] = np.ma.copy(self._data['y_data'])
        logger.debug("Loaded %i raw input data point(s)",
                     self._data['y_raw'].size)
        self._preprocess_data()
        logger.info("Loaded %i input data point(s)", self._data['y_data'].size)

    def _map_to_data(self, function):
        """Map a function to all elements of `self._data` (ignore `raw`)."""
        for data_type in ('data', 'train', 'test'):
            x_type = 'x_' + data_type
            y_type = 'y_' + data_type
            if x_type in self._data:
                (self._data[x_type], self._data[y_type]) = function(
                    self._data[x_type], self._data[y_type], text=data_type)

    def _preprocess_data(self):
        """Preprocess input data."""
        logger.info("Preprocessing training data")

        # Imputing
        self._transformer['imputer'].fit(self._data['x_train'])
        self._map_to_data(self._impute_data)

        # Scaling
        self._transformer['x_scaler'].fit(self._data['x_train'])
        self._transformer['y_scaler'].fit(
            np.expand_dims(self._data['y_train'], axis=-1))
        self._map_to_data(self._scale_data)

    def _remove_missing_labels(self, x_data, y_data):
        """Remove missing values in the label data."""
        new_x_data = x_data[~y_data.mask]
        new_y_data = y_data[~y_data.mask]
        diff = y_data.size - new_y_data.size
        if diff:
            logger.info("Removed %i data point(s) where labels were missing",
                        diff)
        return (new_x_data, new_y_data)

    def _reset_input_data(self):
        """Reset self._data."""
        self._data['x_data'] = np.ma.copy(self._data['x_raw'])
        self._data['y_data'] = np.ma.copy(self._data['y_raw'])
        for data_type in ('x_train', 'x_test', 'y_train', 'y_test'):
            self._data.pop(data_type, None)
        self.classes['group_attributes'] = np.copy(
            self.classes['group_attributes_raw'])

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

        # Save file
        if 'x_' in data_type:
            sub_txt = 'features: {}'.format(self.classes['features'])
            scaler = 'x_scaler'
        else:
            sub_txt = 'label: {}'.format(self.classes['label'])
            scaler = 'y_scaler'
        mean = self._transformer[scaler].mean_
        std = self._transformer[scaler].scale_
        header = '{} with shape {} ({:d}: number of observations)\n{}'.format(
            data_type, csv_data.shape, csv_data.shape[0], sub_txt)
        if 'raw' not in data_type:
            header += ('\nNote: values x have to be transformed by std * x + '
                       'mean to get real data\nmean = {}\nstd = {}'.format(
                           mean, std))
        np.savetxt(path, csv_data, delimiter=',', header=header)
        logger.info("Wrote %s", path)

    def _scale_data(self, x_data, y_data=None, text=None):
        """Scale values in the feature data."""
        msg = '' if text is None else " for data '{}'".format(text)
        new_x_data = self._transformer['x_scaler'].transform(x_data)
        logger.info("Scaled x%s", msg)
        logger.debug("by means %s and standard deviations %s",
                     self._transformer['x_scaler'].mean_,
                     self._transformer['x_scaler'].scale_)
        if y_data is None:
            new_y_data = None
        else:
            new_y_data = self._transformer['y_scaler'].transform(
                np.expand_dims(y_data, axis=-1))
            new_y_data = np.squeeze(new_y_data)
            logger.info("Scaled y%s", msg)
            logger.debug("by means %.3e and standard deviations %.3e",
                         self._transformer['y_scaler'].mean_[0],
                         self._transformer['y_scaler'].scale_[0])
        return (new_x_data, new_y_data)

    def _set_prediction_cube_attributes(self, cube, path,
                                        prediction_name=None):
        """Set the attributes of the prediction cube."""
        cube.attributes = {
            'classifier': str(self._CLF_TYPE),
            'dataset': 'MLR model prediction',
            'filename': path,
            'project': 'MLR',
        }
        if prediction_name is not None:
            cube.attributes['prediction_name'] = prediction_name
        params = {}
        for (key, val) in self.parameters.items():
            params[key] = str(val)
        cube.attributes.update(params)
        label = select_metadata(
            self._datasets['training'], var_type='label')[0]
        label_cube = self._load_cube(label['filename'])
        for attr in ('standard_name', 'var_name', 'long_name', 'units'):
            setattr(cube, attr, getattr(label_cube, attr))
