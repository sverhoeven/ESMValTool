"""Convenience functions for GBRT diagnostics."""

import logging
import os

import iris
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from ._base import group_metadata, select_metadata, save_iris_cube

logger = logging.getLogger(os.path.basename(__file__))

VAR_KEYS = [
    'long_name',
    'standard_name',
    'units',
]
NECESSARY_KEYS = VAR_KEYS + [
    'dataset',
    'filename',
    'label',
    'short_name',
    'var_type',
]


def write_cube(cube, attributes, path, cfg):
    """Write cube with all necessary information for GBRT models.

    Parameters
    ----------
    cube : iris.cube.Cube
        Cube which should be written.
    attributes : dict
        Attributes for the cube (needed for GBRT models).
    path : str
        Path to the new file.
    cfg : dict
        Diagnostic script configuration.

    """
    for key in NECESSARY_KEYS:
        if key not in attributes:
            logger.warning(
                "Cannot save cube to %s, attribute '%s' "
                "not given", path, key)
            return
    for var_key in VAR_KEYS:
        setattr(cube, var_key, attributes.pop(var_key))
    setattr(cube, 'var_name', attributes.pop('short_name'))
    cube.attributes.update(attributes)
    save_iris_cube(cube, path, cfg)


class GBRTBase():
    """Base class for GBRT diagnostics.

    Note
    ----
    Several functions need to be implemented in child classes.

    """

    _DEFAULT_PARAMETERS = {
        'n_estimators': 50,
        'max_depth': 4,
        'min_samples_split': 2,
        'learning_rate': 0.01,
        'loss': 'ls',
    }
    _VAR_KEYS = [
        'long_name',
        'standard_name',
        'units',
    ]
    _NECESSARY_KEYS = _VAR_KEYS + [
        'dataset',
        'filename',
        'label',
        'short_name',
        'var_type',
    ]

    def __init__(self, cfg):
        """Initialize class members.

        Parameters
        ----------
        cfg : dict
            Diagnostic script configuration.

        """
        # Private members
        self._clf = {}
        self._data = {}
        self._cubes = {}
        self._cfg = cfg
        self._datasets = {}

        # Public members
        self.classes = {}
        self.parameters = self._load_parameters()

        # Load input datasets
        (training_datasets, prediction_datasets) = self._get_input_datasets()
        if not self._datasets_have_attributes(training_datasets,
                                              log_level='error'):
            raise ValueError()
        if not self._datasets_have_attributes(prediction_datasets,
                                              log_level='error'):
            raise ValueError()
        self._datasets['training'] = self._group_training_datasets(
            training_datasets)
        self._datasets['prediction'] = self._group_prediction_datasets(
            prediction_datasets)
        logger.debug("Initialized GBRT base class")

    def fit(self):
        """Build the GBRT model(s)."""
        for (model_name, datasets) in self._datasets['training'].items():
            logger.info("Fitting GBRT model%s",
                        self._get_logger_suffix(model_name))

            # Initialize members
            self._data[model_name] = {}
            self._cubes[model_name] = {}
            self_data = self._data[model_name]
            self_cubes = self._cubes[model_name]

            # Extract features and labels
            (self_data['x_data'],
             self_cubes['feature'],
             self_data['x_data'],
             self_cubes['label']) = self._extract_features_and_labels(
                 datasets)

            # Separate training and test data
            (self_data['x_train'],
             self_data['x_test'],
             self_data['y_train'],
             self_data['y_test']) = self._train_test_split()

            # Create GBRT model with desired parameters and fit it
            self._clf[model_name] = GradientBoostingRegressor(
                **self.parameters)
            self._clf[model_name].fit(self_data['x_train'],
                                      self_data['y_train'])
            logger.debug("Successfully fitted GBRT model%s",
                         self._get_logger_suffix(model_name))

    def plot_prediction_error(self, filename):
        """Plot prediction error for training and test data."""
        pass

    def plot_feature_importance(self, filename):
        """Plot feature importance."""
        pass

    def plot_partial_dependence(self, filename):
        """Plot partial dependence."""
        pass

    def predict(self):
        """Perform prediction using the GBRT model(s) and write netcdf."""
        if not self._is_fitted():
            logger.error("Prediction not possible because the model is not "
                         "fitted yet, call fit() first")
            return None
        predictions = {}
        for model_name in self._datasets['training']:
            predictions[model_name] = {}
            logger.info("Predicting%s", self._get_logger_suffix(model_name))
            for (prediction_name, datasets) in (
                    self._datasets['prediction'].items()):
                logger.info("Prediction name: %s", prediction_name)

                # Predict
                (x_pred, cube) = self._extract_prediction_input(datasets)
                y_pred = self._clf[model_name].predict(x_pred)
                self._data[model_name][prediction_name] = y_pred
                self._cubes[model_name][prediction_name] = cube
                predictions[model_name][prediction_name] = y_pred

                # Save data into cubes
                cube = cube.copy(data=y_pred.reshape(cube.shape))
                cube.attributes = {}
                description = 'GBRT model prediction{}'.format(
                    self._get_logger_suffix(model_name))
                cube.attributes.update({
                    'description': description,
                    'model_name': str(model_name),
                    'prediction_name': str(prediction_name)})
                cube.attributes.update(self.parameters)
                filename = 'prediction_{}_for_{}.nc'.format(prediction_name,
                                                            model_name)
                new_path = os.path.join(self._cfg['work_dir'], filename)
                save_iris_cube(cube, new_path, self._cfg)

        return predictions

    def _collect_x_data(self, datasets, var_type):
        """Collect x data, must be implemented in child class."""
        raise NotImplementedError("This method must be implemented in the "
                                  "child class")

    def _collect_y_data(self, datasets):
        """Collect y data, must be implemented in child class."""
        raise NotImplementedError("This method must be implemented in the "
                                  "child class")

    def _datasets_have_attributes(self, datasets, log_level='debug'):
        """Check if necessary dataset attributes are given."""
        for dataset in datasets:
            for key in self._NECESSARY_KEYS:
                if key not in dataset:
                    getattr(logger, log_level)("Dataset '%s' does not have "
                                               "necessary attribute '%s'",
                                               dataset, key)
                    return False
        return True

    def _extract_features_and_labels(self, datasets, check_features=False):
        """Extract features and labels from `datasets`."""
        if check_features:
            required_features = self.classes['features']
        (x_data, feature_cube) = self._extract_x_data(datasets, 'feature')
        (y_data, label_cube) = self._extract_y_data(datasets)

        # Check if all required features are available
        if check_features:
            if set(required_features) != set(self.classes['features']):
                raise ValueError("Expected features '{}', got '{}'".format(
                    required_features, self.classes['features']))

        # Check sizes
        if len(x_data) != len(y_data):
            raise ValueError("Size of features and labels does not match, got "
                             "{} observations for the features and {} "
                             "observations for the label".format(len(x_data),
                                                                 len(y_data)))

        # Check for duplicate features
        duplicates = {f for f in self.classes['features'] if
                      self.classes['features'].count(f) > 1}
        if duplicates:
            raise ValueError("Duplicate features in x_data: "
                             "{}".format(duplicates))

        # Save data and return it
        self._data['x_data'] = x_data
        self._data['y_data'] = y_data
        return (x_data, feature_cube, y_data, label_cube)

    def _extract_prediction_input(self, datasets, check_features=True):
        """Extract prediction input `datasets`."""
        if check_features:
            required_features = self.classes['features']
        (x_data, prediction_input_cube) = self._extract_x_data(
            datasets, 'prediction_input')

        # Check if all required features are available
        if check_features:
            if set(required_features) != set(self.classes['features']):
                raise ValueError("Expected features '{}' for prediction, got "
                                 "'{}'".format(required_features,
                                               self.classes['features']))

        # Check for duplicate features
        duplicates = {f for f in self.classes['features'] if
                      self.classes['features'].count(f) > 1}
        if duplicates:
            raise ValueError("Duplicate prediction input in x_data: "
                             "{}".format(duplicates))

        # Save data and return it
        self._data['x_pred'] = x_data
        return (x_data, prediction_input_cube)

    def _extract_x_data(self, datasets, var_type):
        """Extract required x data of type `var_type` from `datasets`."""
        allowed_types = ('feature', 'prediction_input')
        if var_type not in allowed_types:
            raise ValueError("Excepted one of '{}' for 'var_type', got "
                             "'{}'".format(allowed_types, var_type))

        # Collect data from datasets
        datasets = select_metadata(datasets, var_type=var_type)
        (x_data, names, cube) = self._collect_x_data(datasets, var_type)

        # Return data
        self.classes['features'] = names
        logger.debug("Found features: %s", self.classes['features'])
        return (x_data, cube)

    def _extract_y_data(self, datasets):
        """Extract y data (labels) from `datasets`."""
        datasets = select_metadata(datasets, var_type='label')
        (y_data, name, cube) = self._collect_y_data(datasets)
        self.classes['label'] = name
        logger.debug("Found label: %s", self.classes['label'])
        return (y_data, cube)

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
                    for var_key in self._VAR_KEYS:
                        dataset_info[var_key] = getattr(cube, var_key)
                    dataset_info['short_name'] = getattr(cube, 'var_name')

                    # Check if necessary keys are available
                    if self._datasets_have_attributes([dataset_info]):
                        datasets.append(dataset_info)
                    else:
                        logger.debug("Skipping %s", path)
        return datasets

    def _get_input_datasets(self):
        """Get input data (including ancestors)."""
        input_datasets = list(self._cfg['input_data'].values())
        input_datasets.extend(self._get_ancestor_datasets())
        feature_datasets = select_metadata(input_datasets, var_type='feature')
        label_datasets = select_metadata(input_datasets, var_type='label')
        prediction_datasets = select_metadata(input_datasets,
                                              var_type='prediction_input')
        training_datasets = feature_datasets + label_datasets
        return (training_datasets, prediction_datasets)

    def _get_logger_suffix(self, model_name): # noqa
        """Get message suffix for logger based on `model_name`."""
        msg = '' if model_name is None else ' for model {}'.format(model_name)
        return msg

    def _group_training_datasets(self, datasets):
        """Group training datasets, must be implemented in child class."""
        raise NotImplementedError("This method must be implemented in the "
                                  "child class")

    def _group_prediction_datasets(self, datasets): # noqa
        """Group prediction datasets (use `prediction_name` key)."""
        return group_metadata(datasets, 'prediction_name')

    def _is_fitted(self):
        """Check if the GBRT models are fitted."""
        return bool(self._data)

    def _load_parameters(self):
        """Load parameters for classifier from recipe."""
        parameters = self._DEFAULT_PARAMETERS
        parameters.update(self._cfg.get('parameters', {}))
        logger.debug("Use parameters %s for GBRT", parameters)
        return parameters

    def _train_test_split(self):
        """Split data into training and test data."""
        return train_test_split(
            self._data['x_data'],
            self._data['y_data'],
            test_size=self.parameters.get('test_size', 0.25))
