"""Create simple tables and plots for exploring the C-MAPSS data."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_feature_columns(data):
    """Return sensor and operating-setting column names."""
    return [
        column for column in data.columns
        if column.startswith('sensor_') or column.startswith('op_setting_')
    ]


def _split_summary(data):
    """Summarize the size and basic quality of a train or test dataset."""
    return {
        'rows': int(len(data)),
        'engines': int(data['unit'].nunique()), # number of unique engines
        'missing_values': int(data.isna().sum().sum()),
        'duplicate_rows': int(data.duplicated().sum()),
        'shortest_engine_life': int(data.groupby('unit')['cycle'].max().min()),
        'longest_engine_life': int(data.groupby('unit')['cycle'].max().max()),
    }


def create_eda_reports(train, test, output_dir, subset):
    """Save summary files and plots for the training and test data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = get_feature_columns(train)

    train[features].describe().T.to_csv(output_dir / 'feature_summary.csv')
    train[features].corr().to_csv(output_dir / 'feature_correlations.csv')

    summary = {
        'subset': subset,
        'train': _split_summary(train),
        'test': _split_summary(test),
        'train_rul': {
            'min': int(train['RUL'].min()),
            'median': float(train['RUL'].median()),
            'max': int(train['RUL'].max()),
        },
        'test_rul': {
            'min': int(test['RUL'].min()),
            'median': float(test['RUL'].median()),
            'max': int(test['RUL'].max()),
        },
    }
    with (output_dir / 'eda_summary.json').open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)

    sns.set_theme(style='whitegrid')

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    train_cycles = train.groupby('unit')['cycle'].max()
    axes[0].hist(train_cycles, bins=20)
    axes[0].set(title='Training cycles per engine', xlabel='Cycles', ylabel='Engines')
    axes[1].hist(train['RUL'], bins=30)
    axes[1].set(title='Training RUL', xlabel='RUL (cycles)', ylabel='Rows')
    figure.tight_layout()
    figure.savefig(output_dir / 'cycle_and_rul_distributions.png', dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        train[features].corr(),
        cmap='coolwarm',
        center=0,
        ax=axis,
    )
    axis.set_title(f'{subset} feature correlations')
    figure.tight_layout()
    figure.savefig(output_dir / 'feature_correlations.png', dpi=150)
    plt.close(figure)
