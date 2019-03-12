"""Gaussian Process Regression model."""

import logging
import os

import numpy as np
from esmvaltool.diag_scripts.mlr.models import MLRModel
from george import GP
from scipy.optimize import fmin_l_bfgs_b
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils import check_random_state

logger = logging.getLogger(os.path.basename(__file__))


@MLRModel.register_mlr_model('sklearn_gpr')
class SklearnGPRModel(MLRModel):
    """Gaussian Process Regression model (:mod:`sklearn` implementation).

    Note
    ----
    See :mod:`esmvaltool.diag_scripts.mlr.models`.

    """

    _CLF_TYPE = GaussianProcessRegressor

    def print_kernel_info(self):
        """Print information of the fitted kernel of the GPR model."""
        if not self._is_fitted():
            logger.error("Printing kernel not possible because the model is "
                         "not fitted yet, call fit() first")
            return
        kernel = self._clf.named_steps['regressor'].regressor_.kernel_
        logger.info("Fitted kernel: %s (%i hyperparameters)", kernel,
                    kernel.n_dims)
        logger.info("Hyperparameters:")
        for hyper_param in kernel.hyperparameters:
            logger.info(hyper_param)
        logger.info("Theta:")
        for elem in kernel.theta:
            logger.info(elem)


class GeorgeGaussianProcessRegressor(BaseEstimator, RegressorMixin):
    """:mod:`sklearn` API for :mod:`george.GP`.

    Note
    ----
    :mod:`george` offers faster optimization functions useful for large data
    sets. The implementation of this class is based on :mod:`sklearn`
    <https://scikit-learn.org/stable/modules/generated/
    sklearn.gaussian_process.GaussianProcessRegressor.html>.

    """

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

        # george GP main object
        self._gp = GP(
            kernel=kernel,
            fit_kernel=fit_kernel,
            mean=mean,
            fit_mean=fit_mean,
            white_noise=white_noise,
            fit_white_noise=fit_white_noise,
            solver=solver,
            **kwargs)

        # Training data
        self._x_train = None
        self._y_train = None

        # Random number generator
        self._rng = check_random_state(self.random_state)

    def fit(self, x_train, y_train):
        """Fit regressor using given training data."""
        self._x_train = np.copy(x_train) if self.copy_X_train else x_train
        self._y_train = np.copy(y_train) if self.copy_X_train else y_train

        # Optimize hyperparameters of kernel if desired
        if self.optimizer is not None and self._gp.vector_size:
            self._gp.compute(self._x_train)

            print(self._gp)
            print(self._gp.get_parameter_dict())
            print(self._gp.get_parameter_bounds())

            # Objective function to minimize (- log-marginal likelihood)
            def obj_func(theta):
                self._gp.set_parameter_vector(theta)
                log_like = self._gp.log_likelihood(self._y_train, quiet=True)
                log_like = log_like if np.isfinite(log_like) else -np.inf
                grad_log_like = self._gp.grad_log_likelihood(
                    self._y_train, quiet=True)
                return (-log_like, -grad_log_like)

            # Start optimization from values specfied in kernel
            bounds = self._gp.get_parameter_bounds()
            optima = [
                self._constrained_optimization(obj_func,
                                               self._gp.get_parameter_vector(),
                                               bounds)
            ]

            # Additional runs (chosen from log-uniform intitial theta)
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer > "
                        "0) requires that all bounds are finite")
                for _ in range(self.n_restarts_optimizer):
                    theta_initial = self._rng.uniform(bounds[:, 0],
                                                      bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial,
                                                       bounds))

            # Select best run (with lowest negative log-marginal likelihood)
            log_like_vals = [opt[1] for opt in optima]
            theta_opt = optima[np.argmin(log_like_vals)][0]
            self._gp.set_parameter_vector(theta_opt)
        return self

    def predict(self, x_pred, **kwargs):
        """Predict for unknown data."""
        if not self._gp.computed:
            raise NotFittedError("Prediction not possible, model not fitted")
        return self._gp.predict(self._y_train, x_pred, **kwargs)

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        """Optimize hyperparameters.

        Note
        ----
        See implementation of :mod:`sklearn.gaussian_process.
        GaussianProcessRegressor`.

        """
        if self.optimizer == 'fmin_l_bfgs_b':
            (theta_opt, func_min, convergence_dict) = fmin_l_bfgs_b(
                obj_func, initial_theta, bounds=bounds)
            if convergence_dict["warnflag"] != 0:
                logger.warning(
                    "fmin_l_bfgs_b terminated abnormally with the state: %s",
                    convergence_dict)
        elif callable(self.optimizer):
            (theta_opt, func_min) = self.optimizer(
                obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")
        return (theta_opt, func_min)


@MLRModel.register_mlr_model('george_gpr')
class GeorgeGPRModel(MLRModel):
    """Gaussian Process Regression model (:mod:`george` implementation).

    Note
    ----
    See :mod:`esmvaltool.diag_scripts.mlr.models`.

    """

    _CLF_TYPE = GeorgeGaussianProcessRegressor

    def print_kernel_info(self):
        """Print information of the fitted kernel of the GPR model."""
        logger.error("PRINTING KERNEL NOT SUPPORTED YET")
