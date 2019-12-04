"""Gradient Boosting Regression model."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from esmvaltool.diag_scripts.mlr.models import MLRModel

logger = logging.getLogger(os.path.basename(__file__))


class GBRModel(MLRModel):
    """Base class for Gradient Boosting Regression models.

    Note
    ----
    See :mod:`esmvaltool.diag_scripts.mlr.models`.

    """

    _CLF_TYPE = None

    def plot_gbr_feature_importance(self, filename=None):
        """Plot GBR feature importance.

        Note
        ----
        In contrast to the general implementation of feature importance using
        model scoring or prediction variance (done via the module
        :mod:`skater`), this function uses properties of the GBR model based on
        the number of appearances of that feature in the regression trees and
        the improvements made by the individual splits (see Friedman, 2001).
        The features plotted here are not necessarily the real input features,
        but the ones after preprocessing. Thus, in the case of PCA this is the
        feature importance of the different principal components.

        Parameters
        ----------
        filename : str, optional (default: 'feature_importance_gbr')
           Name of the plot file.

        """
        if not self._is_ready_for_plotting():
            return
        logger.info("Plotting GBR feature importance")
        if filename is None:
            filename = 'gbr_feature_importance'
        (_, axes) = plt.subplots()
        clf = self._clf.steps[-1][1].regressor_

        # Plot
        feature_importance = clf.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + 0.5
        axes.barh(pos, feature_importance[sorted_idx], align='center')

        # Plot appearance
        y_tick_labels = self.features_after_preprocessing[sorted_idx]
        axes.set_title(
            f"Global feature importance ({self._cfg['mlr_model_name']})")
        axes.set_xlabel('Relative Importance')
        axes.set_yticks(pos)
        axes.set_yticklabels(y_tick_labels)
        new_filename = filename + '.' + self._cfg['output_file_type']
        plot_path = os.path.join(self._cfg['mlr_plot_dir'], new_filename)
        plt.savefig(plot_path, **self._cfg['savefig_kwargs'])
        logger.info("Wrote %s", plot_path)
        plt.close()

    def _plot_training_progress(self,
                                train_score,
                                test_score=None,
                                filename=None):
        """Plot training progress during fitting."""
        if not self._is_ready_for_plotting():
            return
        logger.info("Plotting training progress for GBR model")
        if filename is None:
            filename = 'training_progress'
        (_, axes) = plt.subplots()
        x_values = np.arange(len(train_score), dtype=np.float64) + 1.0

        # Plot train score
        axes.plot(x_values,
                  train_score,
                  color='b',
                  linestyle='-',
                  label='train data')

        # Plot test score if possible
        if test_score is not None:
            axes.plot(x_values,
                      test_score,
                      color='g',
                      linestyle='-',
                      label='test data')

        # Appearance
        ylim = axes.get_ylim()
        axes.set_ylim(0.0, ylim[1])
        axes.set_title(f"Training progress ({self._cfg['mlr_model_name']})")
        axes.set_xlabel('Boosting iterations')
        axes.set_ylabel('Normalized RMSE')
        axes.legend(loc='upper right')
        new_filename = filename + '.' + self._cfg['output_file_type']
        plot_path = os.path.join(self._cfg['mlr_plot_dir'], new_filename)
        plt.savefig(plot_path, **self._cfg['savefig_kwargs'])
        logger.info("Wrote %s", plot_path)
        plt.close()
