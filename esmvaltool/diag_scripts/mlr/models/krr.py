"""Kernel Ridge Regression model."""

import logging
import os

from sklearn.kernel_ridge import KernelRidge

from esmvaltool.diag_scripts.mlr.models import MLRModel

logger = logging.getLogger(os.path.basename(__file__))


@MLRModel.register_mlr_model('krr')
class KRRModel(MLRModel):
    """Kernel Ridge Regression model.

    Note
    ----
    See :mod:`esmvaltool.diag_scripts.mlr.models`.

    """

    _CLF_TYPE = KernelRidge
