"""Gradient Boosting Regression model (using :mod:`sklearn´)."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence

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

    def plot_gbr_partial_dependences(self, filename=None):
        """Plot GBR partial dependences for every feature.

         Note
         ----
         In contrast to the general implementation of partial dependence using
         the module :mod:`skater`), this function uses the :mod:`sklearn`
         implementation of it, which is only supported for tree-based ensemble
         regression methods. See
         <https://scikit-learn.org/stable/modules/ensemble.html>

        Parameters
        ----------
        filename : str, optional (default: 'gbr_partial_dependece_{feature}')
            Name of the plot file.

        """
        if not self._is_ready_for_plotting():
            return
        logger.info("Plotting GBR partial dependences")
        if filename is None:
            filename = 'gbr_partial_dependece_{feature}'
        clf = self._clf.steps[-1][1].regressor_

        # Plot for every feature
        for (idx, feature_name) in enumerate(self.features):
            (_, [axes]) = plot_partial_dependence(clf, self.data['x_train'],
                                                  [idx])
            axes.set_title(f'Partial dependence ({str(self._CLF_TYPE)} Model)')
            axes.set_xlabel(f'(Scaled) {feature_name}')
            axes.set_ylabel(f'(Scaled) {self.label}')
            new_filename = (filename.format(feature=feature_name) + '.' +
                            self._cfg['output_file_type'])
            new_path = os.path.join(self._cfg['mlr_plot_dir'], new_filename)
            plt.savefig(new_path, orientation='landscape', bbox_inches='tight')
            logger.info("Wrote %s", new_path)
            plt.close()

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
        if 'x_test' in self.data and 'y_test' in self.data:
            test_score = np.zeros((len(clf.train_score_), ), dtype=np.float64)
            x_test = self._clf.transform_only(self.data['x_test'])
            y_test = self._clf.steps[-1][1].transformer_.transform(
                np.expand_dims(self.data['y_test'], axis=-1))
            y_test = y_test[:, 0]
            for (idx, y_pred) in enumerate(clf.staged_predict(x_test)):
                test_score[idx] = clf.loss_(y_test, y_pred)
        self._plot_prediction_error(train_score, test_score, filename)
