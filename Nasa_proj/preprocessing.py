"""Scale sensor and operating-setting data."""

import json
from pathlib import Path

import joblib
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

try:
    from .eda import get_feature_columns
except ImportError:
    from eda import get_feature_columns


SCALERS = {
    'standard': StandardScaler,
    'minmax': MinMaxScaler,
    'robust': RobustScaler,
}


def save_normalized_data(train, test, output_dir, scaling='standard'):
    """Scale features and save the data and fitted scaler."""
    if scaling not in SCALERS:
        raise ValueError(f'Choose a scaler from: {", ".join(SCALERS)}')

    features = get_feature_columns(train)
    if not features:
        raise ValueError('No sensor or operating-setting columns were found.')
    if train[features].isna().any().any() or test[features].isna().any().any():
        raise ValueError('Feature columns contain missing values.')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler = SCALERS[scaling]()

    # Fit on training data only, then use the same scaler for the test data.
    train_scaled = train.copy()
    test_scaled = test.copy()
    train_scaled[features] = train[features].astype(float)
    test_scaled[features] = test[features].astype(float)
    train_scaled[features] = scaler.fit_transform(train[features])
    test_scaled[features] = scaler.transform(test[features])

    train_scaled.to_csv(output_dir / 'train_normalized.csv', index=False)
    test_scaled.to_csv(output_dir / 'test_normalized.csv', index=False)
    joblib.dump(scaler, output_dir / 'scaler.joblib')

    with (output_dir / 'feature_columns.json').open('w', encoding='utf-8') as file:
        json.dump({'scaling': scaling, 'columns': features}, file, indent=2)
