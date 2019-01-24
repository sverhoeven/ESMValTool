"""Ridge Regression model."""

import logging
import os

from sklearn.linear_model import Ridge

from esmvaltool.diag_scripts.mlr.models import MLRModel

logger = logging.getLogger(os.path.basename(__file__))


@MLRModel.register_mlr_model('ridge')
class RidgeModel(MLRModel):
    """Ridge Regression model.

    Note
    ----
    See :mod:`esmvaltool.diag_scripts.mlr.models`.

    """

    _CLF_TYPE = Ridge

    def __init__(self, cfg, root_dir=None, **metadata):
        """Initialize child class members."""
        super().__init__(cfg, root_dir=root_dir, **metadata)
