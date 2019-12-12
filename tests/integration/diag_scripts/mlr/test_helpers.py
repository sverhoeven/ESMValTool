"""Integration tests for :mod:`esmvaltool.diag_scripts.mlr`."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from esmvaltool.diag_scripts import mlr


X_TRAIN = np.array([
    [3.0],
    [6.0],
    [10.0],
])
Y_TRAIN = np.array([10.0, 20.0, 30.0])


class StdLinearRegression(LinearRegression):
    """Expand :class:`sklearn.linear_model.CoolLinearRegression`."""

    def predict(self, X, return_std=False):
        """Expand :meth:`predict`."""
        pred = super().predict(X)
        if return_std:
            return (pred, np.ones(X.shape[0], dtype=X.dtype))
        return pred


class VarLinearRegression(LinearRegression):
    """Expand :class:`sklearn.linear_model.CoolLinearRegression`."""

    def predict(self, X, return_var=False, return_cov=False):
        """Expand :meth:`predict`."""
        pred = super().predict(X)
        if return_var or return_cov:
            return (pred, np.ones(X.shape[0], dtype=X.dtype))
        return pred


class NonStandardScaler(StandardScaler):
    """Expand :class:`sklearn.preprocessing.StandardScaler`."""

    def fit(self, X, y=None, f=0.0):
        """Expand :meth:`fit`."""
        return_value = super().fit(X, y)
        if self.mean_ is not None:
            self.mean_ += f
        return return_value


class TestAdvancedPipeline():
    """Tests for :class:`esmvaltool.diag_scripts.mlr.AdvancedPipeline`."""

    AREG = mlr.AdvancedTransformedTargetRegressor(
        transformer=NonStandardScaler(),
        regressor=LinearRegression(),
    )
    REG = TransformedTargetRegressor(
        transformer=NonStandardScaler(),
        regressor=LinearRegression(),
    )
    STEPS = [
        [('t', NonStandardScaler())],
        [('t', NonStandardScaler()), ('r', LinearRegression())],
        [('t', NonStandardScaler()), ('r', REG)],
        [('t', NonStandardScaler()), ('r', AREG)],
        [('t', NonStandardScaler()), ('r', AREG)],
        [('t', NonStandardScaler()), ('r', AREG)],
        [('t', NonStandardScaler()), ('r', AREG)],
    ]
    PIPELINES = [mlr.AdvancedPipeline(step) for step in STEPS]
    KW_X0 = {'a': 1, 't__f': 2.0}
    KW_X1 = {'b__a': 1, 't__f': 2.0}
    KW_X2 = {'t__wrongparam': 1, 't__f': 2.0}
    KW_X3 = {'r__wrongparam': 1, 't__f': 2.0}
    KW_X4 = {'r__wrongstep__f': 1, 't__f': 2.0}
    KW_X5 = {'r__regressor__wrongparam': 1, 't__f': 2.0}
    KW_0 = {'t__f': 2.0}
    KW_1 = {'t__f': 2.0, 'r__sample_weight': np.arange(3.0)}
    KW_2 = {'t__f': 2.0, 'r__transformer__f': 3.0}

    TEST_CHECK_FINAL_STEP = zip(
        PIPELINES,
        [TypeError, TypeError, TypeError, True, True, True, True, True],
    )

    @pytest.mark.parametrize('pipeline,output', TEST_CHECK_FINAL_STEP)
    def test_check_final_step(self, pipeline, output):
        """Test checking if final step."""
        pipeline = clone(pipeline)
        if isinstance(output, type):
            with pytest.raises(output):
                pipeline._check_final_step()
            return
        assert pipeline._check_final_step() is None

    TEST_FIT_TARGET_TRANSFORMER_ONLY = zip(
        PIPELINES,
        [{}, {}, {}, KW_X3, KW_X4, KW_0, KW_2],
        [TypeError, TypeError, TypeError, ValueError, ValueError,
         (20.0, 200.0 / 3.0), NotImplementedError],
    )

    @pytest.mark.parametrize('pipeline,kwargs,output',
                             TEST_FIT_TARGET_TRANSFORMER_ONLY)
    def test_fit_target_transformer_only(self, pipeline, kwargs, output):
        """Test fitting of target transformer only."""
        pipeline = clone(pipeline)
        if isinstance(output, type):
            with pytest.raises(output):
                pipeline.fit_target_transformer_only(Y_TRAIN, **kwargs)
            return
        pipeline.fit_target_transformer_only(Y_TRAIN, **kwargs)
        transformer = pipeline.steps[-1][1].transformer_
        np.testing.assert_allclose(transformer.mean_, output[0])
        np.testing.assert_allclose(transformer.var_, output[1])
        assert not hasattr(pipeline.steps[-1][1], 'regressor_')
        with pytest.raises(NotFittedError):
            pipeline.predict(X_TRAIN)
        with pytest.raises(NotFittedError):
            pipeline.steps[-1][1].predict(X_TRAIN)

    TEST_FIT_TRANSFORMERS_ONLY = zip(
        PIPELINES,
        [KW_0, KW_0, KW_1, {}, KW_X0, KW_X1, KW_2],
        [None,
         (np.array([8.333333]), np.array([8.222222])),
         (np.array([8.333333]), np.array([8.222222])),
         (np.array([6.333333]), np.array([8.222222])),
         ValueError,
         ValueError,
         (np.array([8.333333]), np.array([8.222222]))],
    )

    @pytest.mark.parametrize('pipeline,kwargs,output',
                             TEST_FIT_TRANSFORMERS_ONLY)
    def test_fit_transformers_only(self, pipeline, kwargs, output):
        """Test fitting transformers only."""
        pipeline = clone(pipeline)
        if isinstance(output, type):
            with pytest.raises(output):
                pipeline.fit_transformers_only(X_TRAIN, Y_TRAIN, **kwargs)
            return
        pipeline.fit_transformers_only(X_TRAIN, Y_TRAIN, **kwargs)
        transformer = pipeline.steps[0][1]
        if output is None:
            assert not hasattr(transformer, 'mean_')
            assert not hasattr(transformer, 'var_')
            return
        np.testing.assert_allclose(transformer.mean_, output[0])
        np.testing.assert_allclose(transformer.var_, output[1])
        with pytest.raises(NotFittedError):
            pipeline.predict(X_TRAIN)
        with pytest.raises(NotFittedError):
            pipeline.steps[-1][1].predict(X_TRAIN)

    TEST_TRANSFORM_ONLY = [
        (KW_X0, ValueError),
        (KW_X1, KeyError),
        ({}, np.array([[-1.162476], [-0.116248], [1.278724]])),
        (KW_0, np.array([[-3.162476], [-2.116248], [-0.721276]])),
    ]

    @pytest.mark.parametrize('kwargs,output', TEST_TRANSFORM_ONLY)
    def test_transform_only(self, kwargs, output):
        """Test transforming only."""
        pipeline = mlr.AdvancedPipeline([
            ('s', StandardScaler()),
            ('t', NonStandardScaler()),
            ('r', LinearRegression()),
        ])
        with pytest.raises(NotFittedError):
            pipeline.transform_only(X_TRAIN)
        if isinstance(output, type):
            with pytest.raises(output):
                pipeline.fit(X_TRAIN, Y_TRAIN, **kwargs)
            return
        pipeline.fit(X_TRAIN, Y_TRAIN, **kwargs)
        x_trans = pipeline.transform_only(X_TRAIN)
        np.testing.assert_allclose(x_trans, output, rtol=1e-5)

    TEST_TRANSFORM_TARGET_ONLY = zip(
        PIPELINES,
        [{}, {}, {}, {}, KW_X2, KW_0, KW_X5],
        [TypeError,
         TypeError,
         TypeError,
         np.array([-1.22474487, 0.0, 1.22474487]),
         np.array([-1.22474487, 0.0, 1.22474487]),
         np.array([-1.22474487, 0.0, 1.22474487]),
         np.array([-1.22474487, 0.0, 1.22474487])],
    )

    @pytest.mark.parametrize('pipeline,kwargs,output',
                             TEST_TRANSFORM_TARGET_ONLY)
    def test_transform_target_only(self, pipeline, kwargs, output):
        """Test transforming of target only."""
        pipeline = clone(pipeline)
        if isinstance(output, type):
            with pytest.raises(output):
                pipeline.fit_target_transformer_only(Y_TRAIN, **kwargs)
            return
        with pytest.raises(NotFittedError):
            pipeline.transform_target_only(Y_TRAIN)
        pipeline.fit_target_transformer_only(Y_TRAIN, **kwargs)
        y_trans = pipeline.transform_target_only(Y_TRAIN)
        np.testing.assert_allclose(y_trans, output)
        assert not hasattr(pipeline.steps[-1][1], 'regressor_')
        with pytest.raises(NotFittedError):
            pipeline.predict(X_TRAIN)
        with pytest.raises(NotFittedError):
            pipeline.steps[-1][1].predict(X_TRAIN)


class TestAdvancedTransformedTargetRegressor():
    """Tests for class ``AdvancedTransformedTargetRegressor``."""

    AREG = mlr.AdvancedTransformedTargetRegressor(
        transformer=NonStandardScaler(),
        regressor=LinearRegression(),
    )
    FIT_KWARGS = [
        {'a': 1},
        {'b__a': 1, 't__f': 2.0},
        {'regressor__wrongparam': 1},
        {'transformer__fails': 1},
        {},
        {'regressor__sample_weight': np.arange(3.0)},
    ]

    TEST_FIT = zip(
        FIT_KWARGS,
        [ValueError,
         ValueError,
         TypeError,
         NotImplementedError,
         (np.array([20.0]), np.array([200.0 / 3.0]), np.array([0.34756273]),
          -2.2012306472308283,
          np.array([10.54054054, 19.05405405, 30.40540541])),
         (np.array([20.0]), np.array([200.0 / 3.0]), np.array([0.30618622]),
          -1.8371173070873827, np.array([12.5, 20.0, 30.0]))],
    )

    @pytest.mark.parametrize('kwargs,output', TEST_FIT)
    def test_fit(self, kwargs, output):
        """Test fitting with kwargs."""
        reg = clone(self.AREG)
        if isinstance(output, type):
            with pytest.raises(output):
                reg.fit(X_TRAIN, Y_TRAIN, **kwargs)
            return
        reg.fit(X_TRAIN, Y_TRAIN, **kwargs)
        transformer = reg.transformer_
        regressor = reg.regressor_
        np.testing.assert_allclose(transformer.mean_, output[0])
        np.testing.assert_allclose(transformer.var_, output[1])
        np.testing.assert_allclose(regressor.coef_, output[2])
        np.testing.assert_allclose(regressor.intercept_, output[3])
        np.testing.assert_allclose(reg.predict(X_TRAIN), output[4])
