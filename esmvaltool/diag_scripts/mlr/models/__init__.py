"""Base class for MLR models."""

import logging
import os
from pprint import pformat

import cf_units
import iris
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, LeaveOneOut, train_test_split

# TODO: remove VAR_KEYS
from esmvaltool.diag_scripts.mlr import (
    VAR_KEYS,
    datasets_have_mlr_attributes,
    _load_mlr_models,
)
from esmvaltool.diag_scripts.shared import (
    # TODO
    # group_metadata, plot, select_metadata, save_iris_cube, sorted_metadata)
    group_metadata,
    plot,
    select_metadata,
    sorted_metadata,
)

logger = logging.getLogger(os.path.basename(__file__))


class MLRModel():
    """Base class for MLR models.

    Note
    ----
    All datasets must have the attribute 'var_type' which specifies this
    dataset. Possible values are 'feature' (independent variables used for
    training/testing), 'label' (dependent variables, y-axis) or
    'prediction_input' (independent variables used for prediction of dependent
    variables, usually observational data). All 'feature' and 'label' datasets
    must have the same shape, except the attribute 'broadcast_from' is set to a
    list of suitable coordinate indices (must be done for each feature/label).
    This also applies to the 'prediction_input' data sets. Multiple predictions
    can be specified by the key 'prediction_name'.

    Configuration options in recipe
    -------------------------------
    accept_only_scalar_data : bool, optional (default: False)
        Only accept scalar diagnostic data, if set to True
        'group_datasets_by_attributes should be given.
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
    model : str, optional (default: 'gbr')
        Regression model which is used. Allowed models are child classes of
        this base class given in `esmvaltool.diag_scripts.mlr`.
    normalize_data : dict, optional
        Specify tags (keys) and constants (`float` or `'mean'`, value) for
        normalization of data. This is done by dividing the data by the
        given constant and might be necessary when raw data is very small or
        large which leads to large errors due to finite machine precision.
    parameters : dict, optional
        Parameters used in the classifier.
    predict_kwargs : dict, optional
        Optional keyword arguments for the `clf.predict()` function.
    test_size : float, optional (default: 0.25)
        Fraction of feature/label data which is used as test data and not for
        training (if desired).
    use_coords_as_feature : dict, optional
        Use coordinates (e.g. 'air_pressure' or 'latitude') as features,
        coordinate names are given by the dictionary keys, the associated index
        by the dictionary values.
    use_only_coords_as_features : bool, optional (default: False)
        Use only the specified coordinates as features.

    """

    _CLF_TYPE = None
    _MODELS = {}

    @classmethod
    def register_mlr_model(cls, model):
        """Add model (subclass of this class) to _MODEL dict (decorator)."""
        def decorator(subclass):
            """Decorate subclass."""
            cls._MODELS[model] = subclass
            return subclass

        logger.debug("Found available MLR model '%s'", model)
        return decorator

    @classmethod
    def create(cls, model, *args, **kwargs):
        """Create desired MLR model subclass (factory method)."""
        _load_mlr_models()
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
            return cls._MODELS[default_model](*args, **kwargs)
        logger.info("Created MLR model '%s'", model)
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
        self._clf = None
        self._data = {}
        self._datasets = {}
        self._cfg = cfg
        imputation_strategy = self._cfg.get('imputation_strategy', 'remove')
        if imputation_strategy == 'remove':
            self._imputer = None
        else:
            self._imputer = SimpleImputer(
                strategy=imputation_strategy,
                fill_value=self._cfg.get('imputation_constant'))

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

        # Load input datasets
        (training_datasets,
         prediction_datasets) = self._get_input_datasets(**metadata)
        self._datasets['training'] = self._group_by_attributes(
            training_datasets)
        self._datasets['prediction'] = self._group_prediction_datasets(
            prediction_datasets)
        msg = ('' if not self.parameters else ' with parameters {} found in '
               'recipe'.format(self.parameters))
        logger.info("Initialized MRT model%s", msg)
        logger.debug("Found training data:")
        logger.debug(pformat(self._datasets['training']))
        logger.debug("Found prediction data:")
        logger.debug(pformat(self._datasets['prediction']))

        # Check if data was found
        if not training_datasets:
            msg = ' for metadata {}'.format(metadata) if metadata else ''
            raise ValueError("No training data (features/labels){} "
                             "found".format(msg))

        # Load training data
        self._load_training_data()

    def export_training_data(self, filename=None):
        """Export all data contained in `self._data`.

        Parameters
        ----------
        filename : str, optional (default: '{data_type}.csv')
            Name of the exported files.

        """
        if filename is None:
            filename = '{data_type}.csv'
        for data_type in ('x_raw', 'x_data', 'x_train', 'x_test', 'y_raw',
                          'y_data', 'y_train', 'y_test'):
            if data_type in self._data:
                path = os.path.join(self._cfg['mlr_work_dir'],
                                    filename.format(data_type=data_type))
                if 'x_' in data_type:
                    sub_txt = 'features: {}'.format(self.classes['features'])
                else:
                    sub_txt = 'label: {}'.format(self.classes['label'])
                header = ('{} with shape {} ({:d}: number of '
                          'observations)\n{}'.format(
                              data_type, self._data[data_type].shape,
                              self._data[data_type].shape[0], sub_txt))
                np.savetxt(
                    path, self._data[data_type], delimiter=',', header=header)
                logger.info("Wrote %s", path)

    def fit(self, **parameters):
        """Initialize and fit the MLR model(s).

        Parameters
        ----------
        parameters : fit parameters, optional
            Parameters to initialize and fit the MLR model(s). Overwrites
            default and recipe settings.

        Raises
        ------
        ValueError
            If type of classifier is not specified, e.g. when this method is
            called from base class or no MLR models were found.

        """
        self._check_clf_type(text='Fitting MLR model')
        logger.info("Fitting MLR model with classifier '%s'",
                    self._CLF_TYPE.__name__)
        if parameters:
            logger.info(
                "Using additional parameters %s given in fit() function",
                parameters)

        # Create MLR model with desired parameters and fit it
        params = dict(self.parameters)
        params.update(parameters)
        self._clf = self._CLF_TYPE(**params)  # noqa
        self._clf.fit(self._data['x_train'], self._data['y_train'])
        self.parameters = self._clf.get_params()
        logger.info(
            "Successfully fitted '%s' model on %i training point(s) "
            "with parameters %s", self._CLF_TYPE.__name__,
            len(self._data['y_train']), self.parameters)

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

        Raises
        ------
        ValueError
            * If type of classifier is not specified, e.g. when this method is
              called from base class or no MLR models were found.
            * `param_grid` is not specified, neither in the recipe nor in this
              function.

        """
        self._check_clf_type(text='GridSearchCV')
        parameter_grid = dict(self._cfg.get('grid_search_cv_param_grid', {}))
        if param_grid is not None:
            parameter_grid = param_grid
        if not parameter_grid:
            raise ValueError("No parameter grid given (neither in recipe nor "
                             "in grid_search_cv() function)")
        logger.info(
            "Performing exhaustive grid search cross-validation with "
            "classifier '%s' and parameter grid %s", self._CLF_TYPE.__name__,
            parameter_grid)
        additional_args = dict(self._cfg.get('grid_search_cv_kwargs', {}))
        additional_args.update(kwargs)
        if additional_args:
            logger.info(
                "Using additional keyword arguments %s given in "
                "recipe and grid_search_cv() function", additional_args)
            if additional_args.get('cv', '').lower() == 'loo':
                additional_args['cv'] = LeaveOneOut()

        # Create MLR model with desired parameters and fit it
        clf = GridSearchCV(
            self._CLF_TYPE(**self.parameters),  # noqa
            parameter_grid,
            **additional_args)
        clf.fit(self._data['x_train'], self._data['y_train'])
        self.parameters.update(clf.best_params_)
        if hasattr(clf, 'best_estimator_'):
            self._clf = clf.best_estimator_
        else:
            self._clf = self._CLF_TYPE(**params)  # noqa
            self._clf.fit(self._data['x_train'], self._data['y_train'])
        self.parameters = self._clf.get_params()
        logger.info(
            "Exhaustive grid search successful, found best parameters %s",
            clf.best_params_)
        logger.debug("CV results:")
        logger.debug(pformat(clf.cv_results_))
        logger.info(
            "Successfully fitted '%s' model on %i training point(s) "
            "with parameters %s", self._CLF_TYPE.__name__,
            len(self._data['y_train']), self.parameters)

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
        for (f_idx, feature) in enumerate(self.classes['features']):
            if self._cfg.get('accept_only_scalar_data'):
                for (g_idx, group_attr) in enumerate(
                        self.classes['group_attributes']):
                    x_data = (self._data['x_data'][g_idx, f_idx] *
                              self._data['x_norm'][f_idx])
                    y_data = (
                        self._data['y_data'][g_idx] * self._data['y_norm'])
                    axes.scatter(x_data, y_data, label=group_attr)
                    legend = axes.legend(
                        loc='center left',
                        ncol=2,
                        bbox_to_anchor=[1.05, 0.5],
                        borderaxespad=0.0)
            else:
                x_data = (self._data['x_data'][:, f_idx] *
                          self._data['x_norm'][f_idx])
                y_data = self._data['y_data'] * self._data['y_norm']
                axes.plot(x_data, y_data, '.')
                legend = None
            axes.set_title(feature)
            axes.set_xlabel('{} / {}'.format(
                feature, self.classes['features_units'][f_idx]))
            axes.set_ylabel('{} / {}'.format(self.classes['label'],
                                             self.classes['label_units']))
            new_filename = (filename.format(feature=feature) + '.' +
                            self._cfg['output_file_type'])
            new_path = os.path.join(self._cfg['mlr_plot_dir'], new_filename)
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
                "Using additional keyword arguments %s for predict() function",
                predict_kwargs)

        # Iterate over predictions
        predictions = {}
        for (pred_name, datasets) in self._datasets['prediction'].items():
            if pred_name is None:
                logger.info("Started prediction")
                filename = 'prediction.nc'
            else:
                logger.info("Started prediction for prediction %s", pred_name)
                filename = 'prediction_{}.nc'.format(pred_name)
            (x_pred, cube) = self._extract_prediction_input(datasets)
            x_pred /= self._data['x_norm']
            if self._imputer is not None:
                (x_pred, _) = self._impute_missing_features(
                    x_pred, y_data=None, text='prediction input')
            y_pred = self._clf.predict(x_pred, **kwargs)
            y_pred *= self._data['y_norm']
            self._data[pred_name] = y_pred
            predictions[pred_name] = y_pred

            # Save data into cubes
            pred_cube = cube.copy(data=y_pred.reshape(cube.shape))
            if self._imputer is None:
                if np.ma.is_masked(cube.data):
                    pred_cube.data = np.ma.array(
                        pred_cube.data, mask=cube.data.mask)
            self._set_prediction_cube_attributes(
                pred_cube, prediction_name=pred_name)
            new_path = os.path.join(self._cfg['mlr_work_dir'], filename)
            # TODO
            # save_iris_cube(pred_cube, new_path, self._cfg)
            iris.save(pred_cube, new_path)
            logger.info("Successfully predicted %i point(s)", len(y_pred))

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
        (self._data['x_train'], self._data['x_test'], self._data['y_train'],
         self._data['y_test']) = train_test_split(
             self._data['x_raw'], self._data['y_raw'], test_size=test_size)
        self._data['x_data'] = np.ma.copy(self._data['x_raw'])
        self._data['y_data'] = np.ma.copy(self._data['y_raw'])
        logger.info("Used %i%% of the input data as test data (%i point(s))",
                    int(test_size * 100), len(self._data['y_test']))
        self._impute_all_data()

    def _check_clf_type(self, text=None):
        """Check if valid classifier type is given."""
        msg = '' if text is None else '{} not possible: '.format(text)
        if self._CLF_TYPE is None:
            raise ValueError(
                "{}No MLR model specified, please use factory "
                "function 'MLRModel.create()' to initialize this "
                "class or populate the module 'esmvaltool."
                "diag_scripts.mlr.models' if necessary".format(msg))

    def _check_cube_coords(self, cube, expected_coords, dataset):
        """Check shape and coordinates of a given cube."""
        cube_coords = [
            '{}, shape {}'.format(coord.name(), coord.shape)
            for coord in cube.coords(dim_coords=True)
        ]
        if expected_coords is None:
            logger.debug("Using coordinates %s for '%s' (found in '%s')",
                         cube_coords, dataset['var_type'], dataset['tag'])
        if self._cfg.get('accept_only_scalar_data'):
            allowed_shapes = [(), (1, )]
            if cube.shape not in allowed_shapes:
                raise ValueError("Expected only cubes with shapes {}, got {} "
                                 "from '{}' dataset {}, adapt option "
                                 "'accept_only_scalar_data' in recipe".format(
                                     allowed_shapes, cube.shape,
                                     dataset['tag'], dataset['dataset']))
        else:
            if expected_coords is not None:
                if cube.coords(dim_coords=True) != expected_coords:
                    coords = [
                        '{}, shape {}'.format(coord.name(), coord.shape)
                        for coord in expected_coords
                    ]
                    raise ValueError("Expected fields with identical "
                                     "coordinates ({}) for '{}', but '{}' of "
                                     "dataset '{}' is differing ({}), "
                                     "consider regridding, pre-selecting "
                                     "datasets at class initialization using "
                                     "'**metadata' or the options "
                                     "'broadcast_from' or "
                                     "'group_datasets_by_attributes'".format(
                                         coords, dataset['var_type'],
                                         dataset['tag'], dataset['dataset'],
                                         cube_coords))
        return cube.coords(dim_coords=True)

    def _check_group_attributes(self, group_attributes):
        """Check if `group_attributes` match with already saved data."""
        if self.classes.get('group_attributes') is None:
            self.classes['group_attributes'] = group_attributes
        else:
            if group_attributes != self.classes['group_attributes']:
                raise ValueError(
                    "Expected identical group attributes for "
                    "different var_types, got '{}' and '{}'".format(
                        group_attributes, self.classes['group_attributes']))

    def _check_features(self, features, units, text=None):
        """Check if `features` match with already saved data."""
        msg = '' if text is None else ' for {}'.format(text)

        # Compare new features to already saved ones
        if self.classes.get('features') is None:
            self.classes['features'] = features
            self.classes['features_units'] = units
        else:
            if features != self.classes['features']:
                raise ValueError("Expected tags '{}'{}, got '{}', consider "
                                 "the use of '**metadata' in class "
                                 "initialization to pre-select dataset or "
                                 "specifiy suitable attributes to group "
                                 "datasets with the option 'group_datasets_"
                                 "by_attributes'".format(
                                     self.classes['features'], msg, features))
            if units != self.classes['features_units']:
                raise ValueError("Expected units '{}' for the tags '{}', got "
                                 "'{}'{}".format(
                                     self.classes['features_units'],
                                     self.classes['features'], units, msg))

        # Check for duplicates
        duplicates = {
            f
            for f in self.classes['features']
            if self.classes['features'].count(f) > 1
        }
        if duplicates:
            raise ValueError("Got duplicate features '{}'{}, consider the "
                             "the use of '**metadata' in class initialization "
                             "to pre-select datasets or specify suitable "
                             "attributes to group datasets with the option "
                             "'group_datasets_by_attributes'".format(
                                 duplicates, msg))

    def _check_label(self, label, units):
        """Check if `label` matches with already saved data."""
        if self.classes.get('label') is None:
            self.classes['label'] = label
            self.classes['label_units'] = units
        else:
            if label != self.classes['label']:
                raise ValueError("Expected unique entries for var_type "
                                 "'label', got '{}' and '{}'".format(
                                     label, self.classes['label']))
            if units != self.classes['label_units']:
                raise ValueError("Expected unique units for the label '{}', "
                                 "got '{}' and '{}'".format(
                                     self.classes['label'], units,
                                     self.classes['label_units']))

    def _collect_x_data(self, datasets, var_type):
        """Collect x data from `datasets`."""
        grouped_datasets = group_metadata(
            datasets, 'group_by_attributes', sort=True)
        group_attributes = []
        x_data = None
        cube = None

        # Iterate over datasets
        for (group_attr, attr_datasets) in grouped_datasets.items():
            if group_attr is not None:
                logger.info("Loading x data of %s", group_attr)
            group_attributes.append(group_attr)
            attr_datasets = sorted_metadata(attr_datasets, 'tag')
            (attr_data, feature_names, feature_units,
             cube) = self._get_x_data_for_group(attr_datasets, var_type)

            # Check features
            text = var_type if group_attr is None else "{} ('{}')".format(
                group_attr, var_type)
            self._check_features(feature_names, feature_units, text=text)

            # Append data
            if x_data is None:
                x_data = attr_data
            else:
                x_data = np.ma.vstack((x_data, attr_data))

        # Check group attributes
        if var_type == 'feature':
            self._check_group_attributes(group_attributes)

        return (x_data, cube)

    def _collect_y_data(self, datasets):
        """Collect y data from `datasets`."""
        grouped_datasets = group_metadata(
            datasets, 'group_by_attributes', sort=True)
        group_attributes = []
        y_data = np.ma.array([])
        cube = None

        # Iterate over datasets
        for (group_attr, dataset) in grouped_datasets.items():
            if len(dataset) > 1:
                msg = "" if group_attr is None else " for '{}'".format(
                    group_attr)
                raise ValueError("Expected exactly one 'label' dataset{}, "
                                 "got {}".format(msg, len(dataset)))
            dataset = dataset[0]
            group_attributes.append(group_attr)

            # Check label
            self._check_label(dataset['tag'], cf_units.Unit(dataset['units']))

            # Save data
            cube = self._load_cube(dataset['filename'])
            self._check_cube_coords(cube, None, dataset)
            y_data = np.ma.hstack((y_data, self._get_cube_data(cube)))

        # Check if data was found
        if cube is None:
            raise ValueError("No 'label' datasets found")

        # Check group attributes
        self._check_group_attributes(group_attributes)

        # Return data
        return y_data

    def _extract_features_and_labels(self, datasets):
        """Extract features and labels from `datasets`."""
        (x_data, _) = self._extract_x_data(datasets, 'feature')
        y_data = self._extract_y_data(datasets)

        # Handle missing values in labels
        logger.debug("Found %i input data point(s)", len(x_data))
        (x_data, y_data) = self._remove_missing_labels(x_data, y_data)

        # Check sizes
        if len(x_data) != len(y_data):
            raise ValueError("Size of features and labels do not match, got "
                             "{} observations for the features and {} "
                             "observations for the label".format(
                                 len(x_data), len(y_data)))

        # Normalize data
        (x_norm, y_norm) = self._get_normalization_constants(x_data, y_data)
        x_data /= x_norm
        y_data /= y_norm

        return (x_data, x_norm, y_data, y_norm)

    def _extract_prediction_input(self, datasets):
        """Extract prediction input `datasets`."""
        (x_data, prediction_input_cube) = self._extract_x_data(
            datasets, 'prediction_input')
        return (x_data, prediction_input_cube)

    def _extract_x_data(self, datasets, var_type):
        """Extract required x data of type `var_type` from `datasets`."""
        allowed_types = ('feature', 'prediction_input')
        if var_type not in allowed_types:
            raise ValueError("Excepted one of '{}' for 'var_type', got "
                             "'{}'".format(allowed_types, var_type))

        # Collect data from datasets
        datasets = select_metadata(datasets, var_type=var_type)
        (x_data, cube) = self._collect_x_data(datasets, var_type)

        # Return data
        logger.debug("Found features '%s' with units '%s'",
                     self.classes['features'], self.classes['features_units'])
        return (x_data, cube)

    def _extract_y_data(self, datasets):
        """Extract y data (labels) from `datasets`."""
        datasets = select_metadata(datasets, var_type='label')
        y_data = self._collect_y_data(datasets)
        logger.debug("Found label '%s' with units '%s'", self.classes['label'],
                     self.classes['label_units'])
        return y_data

    def _get_ancestor_datasets(self):
        """Get ancestor datasets."""
        input_dirs = [
            d for d in self._cfg['input_files']
            if not d.endswith('metadata.yml')
        ]
        if not input_dirs:
            logger.debug("Skipping loading ancestor datasets, 'ancestors' key "
                         "not given")
            return []
        logger.debug("Found ancestor directories:")
        logger.debug(pformat(input_dirs))

        # Extract datasets
        datasets = []
        for input_dir in input_dirs:
            for (root, _, files) in os.walk(input_dir):
                for filename in files:
                    if '.nc' not in filename:
                        continue
                    path = os.path.join(root, filename)
                    cube = iris.load_cube(path)
                    dataset_info = dict(cube.attributes)
                    for var_key in VAR_KEYS:
                        dataset_info[var_key] = getattr(cube, var_key)
                    dataset_info['short_name'] = getattr(cube, 'var_name')

                    # Check if necessary keys are available
                    if datasets_have_mlr_attributes([dataset_info]):
                        datasets.append(dataset_info)
                    else:
                        logger.debug("Skipping %s", path)

        return datasets

    def _get_broadcasted_data(self, datasets, target_shape):
        """Get broadcasted data."""
        new_data = []
        names = []
        units = []
        if not datasets:
            return (new_data, names, units)
        if self._cfg.get('accept_only_scalar_data'):
            logger.warning("Broadcasting is not supported for scalar data "
                           "(option 'accept_only_scalar_data')")
            return (new_data, names, units)
        var_type = datasets[0]['var_type']
        for dataset in datasets:
            cube_to_broadcast = self._load_cube(dataset['filename'])
            data_to_broadcast = np.ma.array(cube_to_broadcast.data)
            name = dataset['tag']
            try:
                new_axis_pos = np.delete(
                    np.arange(len(target_shape)), dataset['broadcast_from'])
            except IndexError:
                raise ValueError("Broadcasting failed for '{}', index out of "
                                 "bounds".format(name))
            logger.info("Broadcasting %s '%s' from %s to %s", var_type, name,
                        data_to_broadcast.shape, target_shape)
            for idx in new_axis_pos:
                data_to_broadcast = np.ma.expand_dims(data_to_broadcast, idx)
            mask = data_to_broadcast.mask
            data_to_broadcast = np.broadcast_to(
                data_to_broadcast, target_shape, subok=True)
            data_to_broadcast.mask = np.broadcast_to(mask, target_shape)
            if not self._cfg.get('use_only_coords_as_features'):
                new_data.append(data_to_broadcast.ravel())
                names.append(name)
                units.append(cube_to_broadcast.units)
        return (new_data, names, units)

    def _get_coordinate_data(self, cube):
        """Get coordinate variables of a `cube` which can be used as x data."""
        new_data = []
        names = []
        units = []

        # Iterate over desired coordinates
        for (coord, coord_idx) in self._cfg.get('use_coords_as_feature',
                                                {}).items():
            if self._cfg.get('accept_only_scalar_data'):
                logger.warning("Using coordinate data is not supported for "
                               "scalar data (option 'accept_only_scalar_"
                               "data')")
                return (new_data, names, units)
            coord_array = np.ma.array(cube.coord(coord).points)
            try:
                new_axis_pos = np.delete(np.arange(len(cube.shape)), coord_idx)
            except IndexError:
                raise ValueError("'use_coords_as_feature' failed, index '{}'"
                                 "is out of bounds for coordinate "
                                 "'{}'".format(coord_idx, coord))
            for idx in new_axis_pos:
                coord_array = np.ma.expand_dims(coord_array, idx)
            mask = coord_array.mask
            coord_array = np.broadcast_to(coord_array, cube.shape, subok=True)
            coord_array.mask = np.broadcast_to(mask, cube.shape)
            new_data.append(coord_array.ravel())
            names.append(coord)
            units.append(cube.coord(coord).units)

        # Check if data is empty if necessary
        if self._cfg.get('use_only_coords_as_features') and not new_data:
            raise ValueError("No data found, 'use_only_coords_as_features' "
                             "can only be used when 'use_coords_as_feature' "
                             "is specified")
        return (new_data, names, units)

    def _get_cube_data(self, cube):  # noqa
        """Get data from cube."""
        if cube.shape == ():
            return cube.data
        return cube.data.ravel()

    def _get_input_datasets(self, **metadata):
        """Get input data (including ancestors)."""
        input_datasets = list(self._cfg['input_data'].values())
        input_datasets.extend(self._get_ancestor_datasets())

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
        if not datasets_have_mlr_attributes(
                training_datasets, log_level='error'):
            raise ValueError()
        if not datasets_have_mlr_attributes(
                prediction_datasets, log_level='error'):
            raise ValueError()
        return (training_datasets, prediction_datasets)

    def _get_normalization_constants(self, x_data, y_data):
        """Get normalization constants for features and labels."""
        x_norm = np.ones(x_data.shape[1])
        y_norm = 1.0
        for (tag, constant) in self._cfg.get('normalize_data', {}).items():
            found_tag = False
            if tag in self.classes['features']:
                idx = self.classes['features'].index(tag)
                if isinstance(constant, str):
                    constant = np.ma.mean(x_data[:, idx])
                if constant == 0.0:
                    logger.warning(
                        "Constant for normalization of feature '%s' is 0.0, "
                        "specify another constant in recipe", tag)
                else:
                    x_norm[idx] = constant
                    logger.info("Normalized feature '%s' with %.2E %s", tag,
                                constant, self.classes['features_units'][idx])
                found_tag = True
            if tag == self.classes['label']:
                if isinstance(constant, str):
                    constant = np.ma.mean(y_data)
                if constant == 0.0:
                    logger.warning(
                        "Constant for normalization of label '%s' is 0.0, "
                        "specify another constant in recipe", tag)
                else:
                    y_norm = constant
                    logger.info("Normalized label '%s' with %.2E %s", tag,
                                constant, self.classes['label_units'])
                found_tag = True
            if not found_tag:
                logger.warning("Tag for normalization '%s' not found", tag)
        return (x_norm, y_norm)

    def _get_x_data_for_group(self, datasets, var_type):
        """Get x data for a group of datasets."""
        attr_data = []
        skipped_datasets = []
        feature_names = []
        feature_units = []
        coords = None
        cube = None

        # Iterate over data
        for dataset in datasets:
            if 'broadcast_from' in dataset:
                skipped_datasets.append(dataset)
                continue
            cube = self._load_cube(dataset['filename'])
            name = dataset['tag']
            coords = self._check_cube_coords(cube, coords, dataset)
            if not self._cfg.get('use_only_coords_as_features'):
                attr_data.append(self._get_cube_data(cube))
                feature_names.append(name)
                feature_units.append(cube.units)

        # Check if data was found
        if cube is None:
            if skipped_datasets:
                raise ValueError(
                    "Expected at least one '{}' dataset without "
                    "the option 'broadcast_from'".format(var_type))
            else:
                raise ValueError("No '{}' datasets found".format(var_type))

        # Add skipped data (which needs broadcasting)
        broadcasted_data = self._get_broadcasted_data(skipped_datasets,
                                                      cube.shape)
        attr_data.extend(broadcasted_data[0])
        feature_names.extend(broadcasted_data[1])
        feature_units.extend(broadcasted_data[2])

        # Add coordinate data if desired and possible
        coord_data = self._get_coordinate_data(cube)
        attr_data.extend(coord_data[0])
        feature_names.extend(coord_data[1])
        feature_units.extend(coord_data[2])

        # Convert data to numpy array with correct shape
        attr_data = np.ma.array(attr_data)
        if attr_data.ndim == 1:
            attr_data = np.ma.expand_dims(attr_data, 0)
        elif attr_data.ndim > 1:
            attr_data = np.ma.swapaxes(attr_data, 0, 1)
        return (attr_data, feature_names, feature_units, cube)

    def _group_prediction_datasets(self, datasets):  # noqa
        """Group prediction datasets (use `prediction_name` key)."""
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
                return datasets
        for dataset in datasets:
            group_by_attributes = ''
            for attribute in attributes:
                if attribute in dataset:
                    group_by_attributes += dataset[attribute] + '-'
            if not group_by_attributes:
                group_by_attributes = dataset['dataset']
            else:
                group_by_attributes = group_by_attributes[:-1]
            dataset['group_by_attributes'] = group_by_attributes
        logger.info("Grouped feature and label datasets by %s", attributes)
        all_groups = {data['group_by_attributes'] for data in datasets}
        logger.info("Found groups %s", all_groups)
        return datasets

    def _impute_all_data(self):
        """Impute all data given in `self._data`."""
        if self._imputer is not None:
            self._imputer.fit(self._data['x_train'].filled(np.nan))
        for data_type in ('data', 'train', 'test'):
            x_type = 'x_' + data_type
            y_type = 'y_' + data_type
            if x_type in self._data:
                (self._data[x_type],
                 self._data[y_type]) = self._impute_missing_features(
                     self._data[x_type], self._data[y_type], text=data_type)

    def _impute_missing_features(self, x_data, y_data=None, text=None):
        """Impute missing values in the feature data."""
        if self._imputer is None:
            strategy = 'removing them'
            if x_data.mask.shape == ():
                mask = np.full(x_data.shape[0], False)
            else:
                mask = np.any(x_data.mask, axis=1)
            new_x_data = x_data.filled()[~mask]
            if y_data is not None:
                new_y_data = y_data.filled()[~mask]
            else:
                new_y_data = None
            n_imputes = x_data.shape[0] - new_x_data.shape[0]
        else:
            strategy = 'setting them to {} ({})'.format(
                self._cfg['imputation_strategy'], self._imputer.statistics_)
            new_x_data = self._imputer.transform(x_data.filled(np.nan))
            if y_data is not None:
                new_y_data = y_data.filled()
            else:
                new_y_data = None
            n_imputes = np.count_nonzero(x_data != new_x_data)
        if n_imputes:
            msg = '' if text is None else ' for {}'.format(text)
            logger.info("Imputed %i missing features%s by %s", n_imputes, msg,
                        strategy)
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

    def _load_cube(self, path):  # noqa
        """Load iris cube and check data type."""
        cube = iris.load_cube(path)
        if not np.issubdtype(cube.dtype, np.number):
            raise TypeError(
                "Data type of cube loaded from '{}' is '{}', at "
                "the moment only numerical data is supported".format(
                    path, cube.dtype))
        return cube

    def _load_parameters(self):
        """Load parameters for classifier from recipe."""
        parameters = self._cfg.get('parameters', {})
        logger.debug("Found parameters %s in recipe", parameters)
        return parameters

    def _load_training_data(self):
        """Load training data (features/labels)."""
        (self._data['x_raw'], self._data['x_norm'], self._data['y_raw'],
         self._data['y_norm']) = self._extract_features_and_labels(
             self._datasets['training'])
        self._data['x_data'] = np.ma.copy(self._data['x_raw'])
        self._data['y_data'] = np.ma.copy(self._data['y_raw'])
        self._data['x_train'] = np.ma.copy(self._data['x_raw'])
        self._data['y_train'] = np.ma.copy(self._data['y_raw'])
        logger.debug("Loaded %i raw input data point(s)",
                     len(self._data['y_raw']))
        self._impute_all_data()
        logger.info("Loaded %i input data point(s)", len(self._data['y_data']))

    def _remove_missing_labels(self, x_data, y_data):  # noqa
        """Remove missing values in the label data."""
        new_x_data = x_data[~y_data.mask]
        new_y_data = y_data[~y_data.mask]
        diff = len(y_data) - len(new_y_data)
        if diff:
            logger.info("Removed %i data point(s) where labels were missing",
                        diff)
        return (new_x_data, new_y_data)

    def _set_prediction_cube_attributes(self, cube, prediction_name=None):
        """Set the attributes of the prediction cube."""
        cube.attributes = {}
        cube.attributes['description'] = 'MLR model prediction'
        if self._CLF_TYPE is not None:
            cube.attributes['classifier'] = self._CLF_TYPE.__name__
        if prediction_name is not None:
            cube.attributes['prediction_name'] = prediction_name
        params = {}
        for (key, val) in self.parameters.items():
            params[key] = str(val)
        cube.attributes.update(params)
        label = select_metadata(
            self._datasets['training'], var_type='label')[0]
        label_cube = self._load_cube(label['filename'])
        for attr in ('var_name', 'standard_name', 'long_name', 'units'):
            setattr(cube, attr, getattr(label_cube, attr))
