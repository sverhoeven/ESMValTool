"""Gaussian Process Regression model (using :mod:`sklearn`)."""

import logging
import os

from esmvaltool.diag_scripts.mlr.models import MLRModel
from sklearn.gaussian_process import GaussianProcessRegressor

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
        kernel = self._clf.named_steps[
            self._PIPELINE_FINAL_STEP].regressor_.kernel_
        logger.info("Fitted kernel: %s", kernel)
        logger.info("All fitted log-hyperparameters:")
        for (idx, hyper_param) in enumerate(kernel.hyperparameters):
            logger.info("%s: %s", hyper_param, kernel.theta[idx])
