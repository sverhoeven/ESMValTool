"""Gradient Boosting Regression model (using :mod:`sklearn´)."""

import logging
import os

import numpy as np
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

    def plot_prediction_error(self, filename=None):
        """Plot prediction error for training and (if possible) test data.

        Parameters
        ----------
        filename : str, optional (default: 'prediction_error')
            Name of the plot file.

        """
        clf = self._clf.steps[-1][1].regressor_
        train_score = clf.train_score_
        test_score = None
        if 'test' in self.data:
            test_score = np.zeros((len(clf.train_score_), ), dtype=np.float64)
            x_test = self._clf.transform_only(self.get_x_array('test'))
            y_test = self._clf.transform_target_only(self.get_y_array('test'))
            for (idx, y_pred) in enumerate(clf.staged_predict(x_test)):
                test_score[idx] = clf.loss_(y_test, y_pred)
        self._plot_prediction_error(train_score, test_score, filename)
