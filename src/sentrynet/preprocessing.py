"""Preprocessing pipeline: log1p, StandardScaler, and OneHotEncoder.

Fit on Normal training rows only. FittedPreprocessor wraps the fitted pipeline and refuses
to be re-fit, so validation and test data can only be transformed, never fit on.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from sentrynet.features import LABEL_COLUMNS, assert_no_labels


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """log1p transform for the heavy-tailed numeric features."""

    def fit(self, X, y=None):  # noqa: N803 - scikit-learn API
        X = self._as_array(X)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):  # noqa: N803 - scikit-learn API
        X = self._as_array(X)
        return np.log1p(np.maximum(X, 0.0))

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([f"log1p_x{i}" for i in range(getattr(self, "n_features_in_", 0))])
        return np.asarray([f"log1p_{name}" for name in input_features], dtype=object)

    @staticmethod
    def _as_array(X):  # noqa: N803 - scikit-learn API
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype="float64")
        return np.asarray(X, dtype="float64")


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    log1p_features: Sequence[str],
) -> ColumnTransformer:
    """Build the ColumnTransformer: log1p+scale, plain scale, and one-hot branches."""
    log_cols = [c for c in numeric_features if c in set(log1p_features)]
    plain_cols = [c for c in numeric_features if c not in set(log1p_features)]

    transformers = []
    if log_cols:
        transformers.append(
            (
                "log_numeric",
                Pipeline(
                    [
                        ("log1p", Log1pTransformer()),
                        ("scaler", StandardScaler()),
                    ]
                ),
                log_cols,
            )
        )
    if plain_cols:
        transformers.append(("plain_numeric", Pipeline([("scaler", StandardScaler())]), plain_cols))
    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical_features),
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


class FittedPreprocessor:
    """A fitted preprocessor that can only transform; re-fitting raises RuntimeError."""

    def __init__(self, column_transformer: ColumnTransformer, input_columns: Sequence[str]):
        check_is_fitted(column_transformer)
        self._ct = column_transformer
        self.input_columns = list(input_columns)
        self.feature_names_out = [str(n) for n in column_transformer.get_feature_names_out()]

    def fit(self, *args, **kwargs):
        raise RuntimeError("FittedPreprocessor is transform-only; it cannot be re-fit.")

    fit_transform = fit

    def transform(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803 - scikit-learn API
        assert_no_labels(X, LABEL_COLUMNS)
        missing = [c for c in self.input_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing feature columns required at fit time: {missing}")
        return np.asarray(self._ct.transform(X.loc[:, self.input_columns]), dtype="float64")

    @property
    def column_transformer(self) -> ColumnTransformer:
        return self._ct

    def uses_standard_scaler(self) -> bool:
        return any(
            isinstance(step, StandardScaler)
            for _, trans, _ in self._ct.transformers_
            if hasattr(trans, "named_steps")
            for step in trans.named_steps.values()
        )

    def uses_one_hot_encoder(self) -> bool:
        return any(isinstance(trans, OneHotEncoder) for _, trans, _ in self._ct.transformers_)


def fit_preprocessor_on_normal_train(
    normal_train_features: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    log1p_features: Sequence[str],
) -> FittedPreprocessor:
    """Fit the preprocessor on Normal training rows and return a transform-only wrapper."""
    assert_no_labels(normal_train_features, LABEL_COLUMNS)
    columns = list(numeric_features) + list(categorical_features)
    ct = build_preprocessor(numeric_features, categorical_features, log1p_features)
    ct.fit(normal_train_features.loc[:, columns])
    return FittedPreprocessor(ct, columns)


def transform_only(preprocessor: FittedPreprocessor, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
    """Transform ``X`` with an already-fitted preprocessor. Never fits."""
    if not isinstance(preprocessor, FittedPreprocessor):
        raise TypeError(
            "transform_only requires a FittedPreprocessor so that fitting is impossible."
        )
    return preprocessor.transform(X)
