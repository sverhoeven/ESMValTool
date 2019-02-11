"""Support Vector Regression model."""

import logging
import os

from sklearn.svm import SVR

from esmvaltool.diag_scripts.mlr.models import MLRModel

logger = logging.getLogger(os.path.basename(__file__))


@MLRModel.register_mlr_model('svr')
class SVRModel(MLRModel):
    """Support Vector Regression model.

    Note
    ----
    See :mod:`esmvaltool.diag_scripts.mlr.models`.

    """

    _CLF_TYPE = SVR
