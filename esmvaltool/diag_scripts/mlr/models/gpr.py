"""Gaussian Process Regression model."""

import logging
import os

from sklearn.gaussian_process import GaussianProcessRegressor

from esmvaltool.diag_scripts.mlr.models import MLRModel

logger = logging.getLogger(os.path.basename(__file__))


@MLRModel.register_mlr_model('gpr')
class GPRModel(MLRModel):
    """Gaussian Process Regression model.

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
        kernel = self._clf.kernel_
        logger.info("Fitted kernel: %s (%i hyperparameters)", kernel,
                    kernel.n_dims)
        logger.info("Hyperparameters:")
        for hyper_param in kernel.hyperparameters:
            logger.info(hyper_param)
        logger.info("Theta:")
        for elem in kernel.theta:
            logger.info(elem)
