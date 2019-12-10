"""Gaussian Process Regression model (using :mod:`george`)."""

import logging
import os
from pprint import pformat

import numpy as np
from george import GP
from scipy.optimize import fmin_l_bfgs_b
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_X_y

from esmvaltool.diag_scripts.mlr.models import MLRModel

logger = logging.getLogger(os.path.basename(__file__))


class GeorgeGaussianProcessRegressor(BaseEstimator, RegressorMixin):
    """:mod:`sklearn` API for :mod:`george.GP`.

    Note
    ----
    :mod:`george` offers faster optimization functions useful for large data
    sets. The implementation of this class is based on :mod:`sklearn`
    <https://scikit-learn.org/stable/modules/generated/
    sklearn.gaussian_process.GaussianProcessRegressor.html>.

    """

    _SKLEARN_SEP = '__'
    _GEORGE_SEP = ':'

    def __init__(self,
                 kernel=None,
                 fit_kernel=True,
                 mean=None,
                 fit_mean=None,
                 white_noise=None,
                 fit_white_noise=None,
                 solver=None,
                 optimizer='fmin_l_bfgs_b',
                 n_restarts_optimizer=0,
                 copy_X_train=True,
                 random_state=None,
                 **kwargs):
        """Initialize :mod:`george.GP` object.

        Note
        ----
        See <https://george.readthedocs.io/en/latest/user/gp/>.

        """
        self.kernel = kernel
        self.fit_kernel = fit_kernel
        self.mean = mean
        self.fit_mean = fit_mean
        self.white_noise = white_noise
        self.fit_white_noise = fit_white_noise
        self.solver = solver
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.copy_X_train = copy_X_train
        self.random_state = random_state
        self.kwargs = kwargs
        self._gp = None
        self._init_gp()

        # Training data
        self._x_train = None
        self._y_train = None

        # Random number generator
        self._rng = check_random_state(self.random_state)

    def fit(self, x_train, y_train):
        """Fit regressor using given training data."""
        (x_train, y_train) = check_X_y(x_train,
                                       y_train,
                                       multi_output=True,
                                       y_numeric=True)
        self._x_train = np.copy(x_train) if self.copy_X_train else x_train
        self._y_train = np.copy(y_train) if self.copy_X_train else y_train
        self._gp.compute(self._x_train)

        # Optimize hyperparameters of kernel if desired
        if self.optimizer is None:
            logger.warning(
                "No optimizer for optimizing Gaussian Process (kernel) "
                "hyperparameters specified, using initial values")
            return self
        if not self._gp.vector_size:
            logger.warning(
                "No free hyperparameters for Gaussian Process (kernel) "
                "specified, optimization not possible")
            return self

        # Objective function to minimize (negative log-marginal likelihood)
        def obj_func(theta, eval_gradient=True):
            neg_log_like = self._gp.nll(theta, self._y_train)
            if eval_gradient:
                return (neg_log_like, self._gp.grad_nll(theta, self._y_train))
            return neg_log_like

        # Start optimization from values specfied in kernel
        logger.debug("Optimizing george GP hyperparameters")
        logger.debug(pformat(self.get_george_params()))
        bounds = np.array(self._gp.get_parameter_bounds(), dtype=float)
        bounds[np.isnan(bounds[:, 0]), 0] = -np.inf
        bounds[np.isnan(bounds[:, 1]), 1] = +np.inf
        optima = [
            self._constrained_optimization(obj_func,
                                           self._gp.get_parameter_vector(),
                                           bounds)
        ]
        logger.debug("Found parameters %s", optima[-1][0])

        # Additional runs (chosen from log-uniform intitial theta)
        if self.n_restarts_optimizer > 0:
            if not np.all(np.isfinite(bounds)):
                raise ValueError(
                    f"Multiple optimizer restarts (n_restarts_optimizer > 0) "
                    f"require that all bounds are given and finite. For "
                    f"parameters {self._gp.get_parameter_names()}, got "
                    f"{bounds}")
            for idx in range(self.n_restarts_optimizer):
                logger.debug(
                    "Restarted hyperparameter optimization, "
                    "iteration %3i/%i", idx + 1, self.n_restarts_optimizer)
                theta_initial = self._rng.uniform(bounds[:, 0], bounds[:, 1])
                optima.append(
                    self._constrained_optimization(obj_func, theta_initial,
                                                   bounds))
                logger.debug("Found parameters %s", optima[-1][0])

        # Select best run (with lowest negative log-marginal likelihood)
        log_like_vals = [opt[1] for opt in optima]
        theta_opt = optima[np.argmin(log_like_vals)][0]
        self._gp.set_parameter_vector(theta_opt)
        self._gp.compute(self._x_train)
        logger.debug("Result of hyperparameter optimization:")
        logger.debug(pformat(self.get_george_params()))
        return self

    def get_george_params(self, include_frozen=False, prefix=''):
        """Get :obj:`dict` of parameters of the :class:`george.GP` member."""
        params = self._gp.get_parameter_dict(include_frozen=include_frozen)
        new_params = {}
        for (key, val) in params.items():
            key = self._str_to_sklearn(key)
            new_params[f'{prefix}{key}'] = val
        return new_params

    def predict(self, x_pred, return_std=False, return_cov=False):
        """Predict for unknown data."""
        if not self._gp.computed:
            raise NotFittedError("Prediction not possible, model not fitted")
        x_pred = check_array(x_pred)
        if return_std:
            pred = self._gp.predict(self._y_train, x_pred, return_var=True)
            return (pred[0], np.sqrt(pred[1]))
        return self._gp.predict(self._y_train, x_pred, return_cov=return_cov)

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        valid_gp_params = self._gp.get_parameter_names(include_frozen=True)
        gp_params = {}
        remaining_params = {}
        for (key, val) in params.items():
            new_key = self._str_to_george(key)
            if new_key in valid_gp_params:
                gp_params[new_key] = val
            else:
                remaining_params[key] = val

        # Initialize new GP object and update parameters of this class
        if remaining_params:
            logger.debug("Updating %s with parameters %s", self.__class__,
                         remaining_params)
            super().set_params(**remaining_params)
            self._init_gp()

        # Update parameters of GP member
        valid_gp_params = self._gp.get_parameter_names(include_frozen=True)
        for (key, val) in gp_params.items():
            if key not in valid_gp_params:
                raise ValueError(
                    f"After updating the GP member with new parameters "
                    f"{remaining_params}, '{self._str_to_sklearn(key)}' is "
                    f"not a valid parameter of it anymore")
            self._gp.set_parameter(key, val)
            logger.debug("Set parameter '%s' of george GP member to '%s'",
                         self._str_to_sklearn(key), val)
        return self

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        """Optimize hyperparameters.

        Note
        ----
        See implementation of
        :class:`sklearn.gaussian_process.GaussianProcessRegressor`.

        """
        if self.optimizer == 'fmin_l_bfgs_b':
            (theta_opt, func_min,
             convergence_dict) = fmin_l_bfgs_b(obj_func,
                                               initial_theta,
                                               bounds=bounds)
            if convergence_dict["warnflag"] != 0:
                logger.warning(
                    "fmin_l_bfgs_b terminated abnormally with the state: %s",
                    convergence_dict)
        elif callable(self.optimizer):
            (theta_opt, func_min) = self.optimizer(obj_func,
                                                   initial_theta,
                                                   bounds=bounds)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")
        return (theta_opt, func_min)

    def _init_gp(self):
        """Initialize :mod:`george.GP` instance."""
        self._gp = GP(kernel=self.kernel,
                      fit_kernel=self.fit_kernel,
                      mean=self.mean,
                      fit_mean=self.fit_mean,
                      white_noise=self.white_noise,
                      fit_white_noise=self.fit_white_noise,
                      solver=self.solver,
                      **self.kwargs)
        logger.debug("Initialized george GP member of %s", self.__class__)

    @classmethod
    def _str_to_george(cls, string):
        """Convert seperators of parameter string to :mod:`george`."""
        return string.replace(cls._SKLEARN_SEP, cls._GEORGE_SEP)

    @classmethod
    def _str_to_sklearn(cls, string):
        """Convert seperators of parameter string to :mod:`sklearn`."""
        return string.replace(cls._GEORGE_SEP, cls._SKLEARN_SEP)


@MLRModel.register_mlr_model('gpr_george')
class GeorgeGPRModel(MLRModel):
    """Gaussian Process Regression model (:mod:`george` implementation).

    Note
    ----
    See :mod:`esmvaltool.diag_scripts.mlr.models`.

    """

    _CLF_TYPE = GeorgeGaussianProcessRegressor
    _GEORGE_CLF = True

    def print_kernel_info(self):
        """Print information of the fitted kernel of the GPR model."""
        self._check_fit_status('Printing kernel')
        clf = self._clf.steps[-1][1].regressor_
        logger.info("Fitted kernel: %s", clf.kernel)
        logger.info("All fitted log-hyperparameters:")
        for (hyper_param, value) in clf.get_george_params().items():
            logger.info("%s: %s", hyper_param, value)

    def _get_clf_parameters(self, deep=True):
        """Get parameters of regressor."""
        params = super()._get_clf_parameters(deep)
        prefix = f'{self._clf.steps[-1][0]}__regressor__'
        params.update(self._clf.steps[-1][1].regressor.get_george_params(
            include_frozen=True, prefix=prefix))
        return params
