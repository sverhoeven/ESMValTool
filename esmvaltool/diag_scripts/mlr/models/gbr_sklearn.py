"""Gradient Boosting Regression model (using :mod:`sklearn´)."""

import logging
import os

from sklearn.ensemble import GradientBoostingRegressor

from esmvaltool.diag_scripts.mlr.models import MLRModel
from esmvaltool.diag_scripts.mlr.models.gbr import GBRModel

logger = logging.getLogger(os.path.basename(__file__))


@MLRModel.register_mlr_model('gbr_sklearn')
class SklearnGBRModel(GBRModel):
    """Gradient Boosting Regression model (:mod:`sklearn` implementation).

    Note
    ----
    See :mod:`esmvaltool.diag_scripts.mlr.models`.

    """

    _CLF_TYPE = GradientBoostingRegressor
