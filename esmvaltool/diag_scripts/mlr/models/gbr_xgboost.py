"""Gradient Boosting Regression model (using :mod:`xgboost´)."""

import logging
import os

from xgboost import XGBRegressor

from esmvaltool.diag_scripts.mlr.models import MLRModel
from esmvaltool.diag_scripts.mlr.models.gbr import GBRModel

logger = logging.getLogger(os.path.basename(__file__))


@MLRModel.register_mlr_model('gbr_xgboost')
class XGBoostGBRModel(GBRModel):
    """Gradient Boosting Regression model (:mod:`xgboost` implementation).

    Note
    ----
    See :mod:`esmvaltool.diag_scripts.mlr.models`.

    """

    _CLF_TYPE = XGBRegressor
