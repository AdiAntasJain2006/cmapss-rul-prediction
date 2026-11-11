"""Run EDA and preprocessing for a NASA C-MAPSS dataset."""

import argparse
from pathlib import Path

import pandas as pd

try:
    from .dataset import calculate_rul, load_data
    from .eda import create_eda_reports
    from .modeling import train_and_evaluate_model
    from .preprocessing import save_normalized_data
except ImportError:
    from dataset import calculate_rul, load_data
    from eda import create_eda_reports
    from modeling import train_and_evaluate_model
    from preprocessing import save_normalized_data


def run_pipeline(
    subset='FD001',
    scaling='standard',
    data_dir=None,
    output_dir=None,
    model_type='xgboost',
    train_model=False,
):
    """Load, explore, normalize, and optionally train a model for one subset."""
    train_raw, test_raw, test_rul = load_data(subset=subset, data_dir=data_dir)
    train = calculate_rul(train_raw)
    test = calculate_rul(test_raw, final_rul=test_rul['RUL'])

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / 'outputs' / subset / scaling
    else:
        output_dir = Path(output_dir)

    create_eda_reports(train, test, output_dir, subset)
    save_normalized_data(train, test, output_dir, scaling)

    if train_model:
        train_normalized = output_dir / 'train_normalized.csv'
        test_normalized = output_dir / 'test_normalized.csv'
        if not train_normalized.is_file() or not test_normalized.is_file():
            raise FileNotFoundError('Normalized data files were not created.')

        train_df = pd.read_csv(train_normalized)
        test_df = pd.read_csv(test_normalized)
        train_and_evaluate_model(train_df, test_df, output_dir, model_type=model_type)

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Explore, normalize, and optionally train a NASA C-MAPSS RUL model.'
    )
    parser.add_argument(
        '--subset',
        choices=['FD001', 'FD002', 'FD003', 'FD004'],
        default='FD001',
        help='Dataset to use. Default: FD001.',
    )
    parser.add_argument(
        '--scaling',
        choices=['standard', 'minmax', 'robust'],
        default='standard',
        help='How to scale sensor values. Default: standard.',
    )
    parser.add_argument(
        '--model-type',
        choices=['xgboost', 'random_forest', 'linear'],
        default='xgboost',
        help='Type of regressor to train. Default: xgboost.',
    )
    parser.add_argument(
        '--train-model',
        action='store_true',
        help='Train and evaluate a baseline model after preprocessing.',
    )
    parser.add_argument('--data-dir', type=Path, help='Folder containing the data files.')
    parser.add_argument('--output-dir', type=Path, help='Folder for reports and CSV files.')
    args = parser.parse_args()

    output_dir = run_pipeline(
        subset=args.subset,
        scaling=args.scaling,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        train_model=args.train_model,
    )
    print(f'Reports and normalized data saved in: {output_dir}')


if __name__ == '__main__':
    main()
