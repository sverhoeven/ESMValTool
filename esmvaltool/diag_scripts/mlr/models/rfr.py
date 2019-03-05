"""Random Forest Regression model."""

import logging
import os

from sklearn.ensemble import RandomForestRegressor

from esmvaltool.diag_scripts.mlr.models import MLRModel

logger = logging.getLogger(os.path.basename(__file__))


@MLRModel.register_mlr_model('rfr')
class RFRModel(MLRModel):
    """Random Forest Regression model.

    Note
    ----
    See :mod:`esmvaltool.diag_scripts.mlr.models`.

    """

    _CLF_TYPE = RandomForestRegressor
