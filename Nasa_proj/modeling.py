"""Train and evaluate a baseline RUL prediction model."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


MODEL_TYPES = {
    'xgboost': 'xgboost',
    'random_forest': 'random_forest',
    'linear': 'linear',
}


def _feature_target_split(data):
    """Return model features and target from a normalized dataset."""
    feature_columns = [
        column
        for column in data.columns
        if column not in {'unit', 'cycle', 'RUL'}
    ]
    if not feature_columns:
        raise ValueError('No feature columns were found in the dataset.')
    return data[feature_columns], data['RUL']


def _build_model(model_type):
    """Create a baseline regressor for remaining useful life estimation."""
    model_choice = model_type.lower()
    if model_choice not in MODEL_TYPES:
        raise ValueError(
            f'Unsupported model type: {model_type}. Choose from: '
            f'{", ".join(MODEL_TYPES)}.'
        )

    if model_choice == 'xgboost':
        return XGBRegressor(
            objective='reg:squarederror',
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
    if model_choice == 'random_forest':
        return RandomForestRegressor(
            n_estimators=250,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
    return LinearRegression()


def train_and_evaluate_model(train_df, test_df, output_dir, model_type='xgboost'):
    """Fit a model on the training split and evaluate it on the test split."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = _feature_target_split(train_df)
    X_test, y_test = _feature_target_split(test_df)

    model = _build_model(model_type)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        'model_type': model_type,
        'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
        'mae': float(mean_absolute_error(y_test, predictions)),
        'r2': float(r2_score(y_test, predictions)),
        'mape': float(np.mean(np.abs((y_test - predictions) / y_test))) if np.all(y_test != 0) else None,
    }

    joblib.dump(model, output_dir / 'model.joblib')
    pd.DataFrame({'actual': y_test, 'predicted': predictions}).to_csv(
        output_dir / 'model_predictions.csv', index=False
    )

    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.Series(
            model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)
        feature_importance.to_csv(output_dir / 'feature_importance.csv')

    with (output_dir / 'model_metrics.json').open('w', encoding='utf-8') as file:
        json.dump(metrics, file, indent=2)

    print(f'Model trained with {model_type} and saved in: {output_dir}')
    print(
        'Metrics: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}'.format(
            rmse=metrics['rmse'],
            mae=metrics['mae'],
            r2=metrics['r2'],
        )
    )
    return model, metrics
