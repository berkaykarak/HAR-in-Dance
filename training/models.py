from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

_import_error: BaseException | None = None
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError as e:  # pragma: no cover
    tf = None  # type: ignore[misc, assignment]
    keras = None  # type: ignore[misc, assignment]
    layers = None  # type: ignore[misc, assignment]
    _import_error = e


def _ensure_tf() -> None:
    if keras is None:  # pragma: no cover
        raise ImportError(
            "TensorFlow is required for CNN1D/LSTM. Install: pip install tensorflow"
        ) from _import_error


def _xgboost_for_har_features(
    n_estimators: int,
    max_depth: int | None,
    random_state: int,
):
    """Yuksek boyutlu UCI-tarzı ozellik vektorleri icin XGBoost ayarlari."""
    from xgboost import XGBClassifier

    depth = 6 if max_depth is None else max_depth
    return XGBClassifier(
        n_estimators=max(n_estimators, 300),
        max_depth=depth,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.75,
        min_child_weight=1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="mlogloss",
    )


def _xgboost_default(
    n_estimators: int,
    max_depth: int | None,
    random_state: int,
):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="mlogloss",
    )


def train_sklearn(
    model_key: Literal["rf", "xgboost"],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int,
    max_depth: int | None,
    random_state: int,
    *,
    xgboost_har_features: bool = False,
    xgb_tune_max_depth: bool = False,
    xgb_depth_candidates: tuple[int, ...] = (3, 4, 5, 6, 8, 10),
    xgb_cv_folds: int = 3,
    xgb_scoring: str = "accuracy",
) -> tuple[object, dict[str, float | str]]:
    if model_key == "rf":
        est = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        if xgboost_har_features:
            est = _xgboost_for_har_features(n_estimators, max_depth, random_state)
        else:
            est = _xgboost_default(n_estimators, max_depth, random_state)
        if xgb_tune_max_depth:
            search = GridSearchCV(
                estimator=est,
                param_grid={"max_depth": list(xgb_depth_candidates)},
                cv=xgb_cv_folds,
                scoring=xgb_scoring,
                n_jobs=-1,
                refit=True,
            )
            search.fit(x_train, y_train)
            est = search.best_estimator_
    est.fit(x_train, y_train)
    pred = est.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "report": classification_report(y_test, pred, digits=4),
        "best_max_depth": str(getattr(est, "max_depth", "n/a")),
    }
    return est, metrics


def build_cnn1d(
    input_steps: int,
    n_channels: int,
    num_classes: int,
    learning_rate: float,
) -> Any:
    _ensure_tf()
    inputs = keras.Input(shape=(input_steps, n_channels))
    x = layers.Conv1D(64, 5, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm(
    input_steps: int,
    n_channels: int,
    num_classes: int,
    learning_rate: float,
) -> Any:
    _ensure_tf()
    inputs = keras.Input(shape=(input_steps, n_channels))
    x = layers.LSTM(64, return_sequences=False)(inputs)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_keras(
    model_key: Literal["cnn1d", "lstm"],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    random_state: int,
) -> tuple[Any, dict[str, float | str]]:
    _ensure_tf()
    tf.keras.utils.set_random_seed(random_state)
    n_steps, n_ch = int(x_train.shape[1]), int(x_train.shape[2])
    num_classes = int(np.max(y_train)) + 1
    if model_key == "cnn1d":
        model = build_cnn1d(n_steps, n_ch, num_classes, learning_rate)
    else:
        model = build_lstm(n_steps, n_ch, num_classes, learning_rate)
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )
    pred = np.argmax(model.predict(x_test, batch_size=batch_size, verbose=0), axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "report": classification_report(y_test, pred, digits=4),
    }
    return model, metrics
