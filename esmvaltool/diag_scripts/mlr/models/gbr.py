"""Gradient Boosting Regression model."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence

from esmvaltool.diag_scripts.mlr.models import MLRModel

logger = logging.getLogger(os.path.basename(__file__))


@MLRModel.register_mlr_model('gbr')
class GBRModel(MLRModel):
    """Gradient Boosting Regression model.

    Note
    ----
    See :mod:`esmvaltool.diag_scripts.mlr.models`.

    """

    _CLF_TYPE = GradientBoostingRegressor

    def plot_gbr_feature_importance(self, filename=None):
        """Plot GBR feature importance.

        Note
        ----
        In contrast to the general implementation of feature importance using
        model scoring or prediction variance (done via the module
        :mod:`skater`), this function uses a specific procedure for GBR based
        on the number of appearances of that feature in the regression trees
        and the improvements made by the individual splits (see Friedman,
        2001).

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
        clf = self._clf.named_steps[self._PIPELINE_FINAL_STEP].regressor_

        # Plot
        feature_importance = clf.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + 0.5
        axes.barh(pos, feature_importance[sorted_idx], align='center')
        axes.set_title(
            f'Variable Importance ({self._CLF_TYPE.__name__} Model)')
        axes.set_xlabel('Relative Importance')
        axes.set_yticks(pos)
        axes.set_yticklabels(self.classes['features'][sorted_idx])
        new_filename = filename + '.' + self._cfg['output_file_type']
        new_path = os.path.join(self._cfg['mlr_plot_dir'], new_filename)
        plt.savefig(new_path, orientation='landscape', bbox_inches='tight')
        logger.info("Wrote %s", new_path)
        plt.close()

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
        clf = self._clf.named_steps[self._PIPELINE_FINAL_STEP].regressor_

        # Plot for every feature
        for (idx, feature_name) in enumerate(self.classes['features']):
            (_, [axes]) = plot_partial_dependence(clf, self._data['x_train'],
                                                  [idx])
            axes.set_title(
                f'Partial dependence ({self._CLF_TYPE.__name__} Model)')
            axes.set_xlabel(f'(Scaled) {feature_name}')
            axes.set_ylabel(f"(Scaled) {self.classes['label']}")
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
        if not self._is_ready_for_plotting():
            return
        logger.info("Plotting prediction error")
        if filename is None:
            filename = 'prediction_error'
        (_, axes) = plt.subplots()
        clf = self._clf.named_steps[self._PIPELINE_FINAL_STEP].regressor_

        # Plot train score
        axes.plot(np.arange(len(clf.train_score_)) + 1,
                  clf.train_score_,
                  'b-',
                  label='Training Set Deviance')

        # Plot test score if possible
        if 'x_test' in self._data:
            test_score = np.zeros((len(clf.train_score_), ), dtype=np.float64)
            x_test = self._clf.transform_only(self._data['x_test'])
            y_test = self._clf.named_steps[
                self._PIPELINE_FINAL_STEP].transformer_.transform(
                    np.expand_dims(self._data['y_test'], axis=-1))
            y_test = y_test[:, 0]
            for (idx, y_pred) in enumerate(clf.staged_predict(x_test)):
                test_score[idx] = clf.loss_(y_test, y_pred)
            axes.plot(np.arange(len(test_score)) + 1,
                      test_score,
                      'r-',
                      label='Test Set Deviance')
        axes.legend(loc='upper right')
        axes.set_title('Deviance ({} Model)'.format(self._CLF_TYPE.__name__))
        axes.set_xlabel('Boosting Iterations')
        axes.set_ylabel('Deviance')
        new_filename = filename + '.' + self._cfg['output_file_type']
        new_path = os.path.join(self._cfg['mlr_plot_dir'], new_filename)
        plt.savefig(new_path, orientation='landscape', bbox_inches='tight')
        logger.info("Wrote %s", new_path)
        plt.close()
