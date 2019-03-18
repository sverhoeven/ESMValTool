"""Gradient Boosting Regression model."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from esmvaltool.diag_scripts.mlr.models import MLRModel
from sklearn.ensemble import GradientBoostingRegressor

logger = logging.getLogger(os.path.basename(__file__))


@MLRModel.register_mlr_model('gbr')
class GBRModel(MLRModel):
    """Gradient Boosting Regression model.

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
        if not self._is_ready_for_plotting():
            return
        logger.info("Plotting prediction error")
        if filename is None:
            filename = 'prediction_error'
        (_, axes) = plt.subplots()
        clf = self._clf.named_steps[self._PIPELINE_FINAL_STEP].regressor_

        # Plot train score
        axes.plot(
            np.arange(len(clf.train_score_)) + 1,
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
            axes.plot(
                np.arange(len(test_score)) + 1,
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
        plt.close('all')
