"""Linear Regression model."""

import logging
import os

from sklearn.linear_model import LinearRegression

from esmvaltool.diag_scripts.mlr.models import MLRModel

logger = logging.getLogger(os.path.basename(__file__))


@MLRModel.register_mlr_model('linear')
class LinearModel(MLRModel):
    """Linear Regression model.

    Note
    ----
    See :mod:`esmvaltool.diag_scripts.mlr.models`.

    """

    _CLF_TYPE = LinearRegression
