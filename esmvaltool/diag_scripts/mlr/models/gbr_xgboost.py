"""Gradient Boosting Regression model (using :mod:`xgboost´)."""

import logging
import os

import numpy as np
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

    def plot_prediction_error(self, filename=None):
        """Plot prediction error for training and (if possible) test data.

        Parameters
        ----------
        filename : str, optional (default: 'prediction_error')
            Name of the plot file.

        """
        clf = self._clf.steps[-1][1].regressor_
        evals_result = clf.evals_result()
        train_score = evals_result['validation_0']['rmse']
        test_score = None
        if 'x_test' in self.data and 'y_test' in self.data:
            test_score = evals_result['validation_1']['rmse']
        self._plot_prediction_error(train_score, test_score, filename)

    def _update_fit_kwargs(self, fit_kwargs):
        """Add transformed training and test data as fit kwargs."""
        reduced_fit_kwargs = {}
        for (param_name, param_val) in fit_kwargs.items():
            reduced_fit_kwargs[param_name.replace(
                f'{self._clf.steps[-1][0]}__', '')] = param_val
        self._clf.fit_transformers_only(self.data['x_train'],
                                        self.data['y_train'],
                                        **reduced_fit_kwargs)
        self._clf.steps[-1][1].fit_transformer_only(self.data['y_train'],
                                                    **reduced_fit_kwargs)

        # Transform input data
        x_train = self._clf.transform_only(self.data['x_train'])
        y_train = self._clf.steps[-1][1].transformer_.transform(
            np.expand_dims(self.data['y_train'], axis=-1))
        eval_set = [(x_train, y_train)]
        if 'x_test' in self.data and 'y_test' in self.data:
            x_test = self._clf.transform_only(self.data['x_test'])
            y_test = self._clf.steps[-1][1].transformer_.transform(
                np.expand_dims(self.data['y_test'], axis=-1))
            eval_set.append((x_test, y_test))

        # Update kwargs
        fit_kwargs.update({
            f'{self._clf.steps[-1][0]}__regressor__eval_metric':
            'rmse',
            f'{self._clf.steps[-1][0]}__regressor__eval_set':
            eval_set,
        })
        logger.debug(
            "Updated keyword arguments of fit() function with training and "
            "(if possible) test datasets for evaluation of prediction error")
        return fit_kwargs
