"""Lasso Regression model with builtin CV."""

import logging
import os

from sklearn.linear_model import LassoCV

from esmvaltool.diag_scripts.mlr.models import MLRModel

logger = logging.getLogger(os.path.basename(__file__))


@MLRModel.register_mlr_model('lasso_cv')
class LassoCVModel(MLRModel):
    """Lasso Regression model with builtin CV.

    Note
    ----
    See :mod:`esmvaltool.diag_scripts.mlr.models`.

    """

    _CLF_TYPE = LassoCV
