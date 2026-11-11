from pathlib import Path
from zipfile import ZipFile
import pandas as pd
import numpy as np


def extract_data(zip_path, extract_to):
    """
    Extracts the contents of a zip file to a specified directory.

    Parameters:
    zip_path (str or Path): The path to the zip file.
    extract_to (str or Path): The directory where the contents will be extracted.

    Returns:
    None
    """
    try:
        with ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_to)
        print(f"Data extracted to {extract_to}")
    except Exception as e:
        print(f"Error extracting data: {e}")
        
        

def load_data(subset='FD001', data_dir=None):
    """Load a C-MAPSS training, test, and remaining-useful-life dataset.

    Args:
        subset: One of ``FD001``, ``FD002``, ``FD003``, or ``FD004``.
        data_dir: Directory containing the extracted C-MAPSS text files.
            Defaults to the workspace's ``extracted/CMaps`` directory.

    Returns:
        The training data, test data, and RUL data, in that order.

    Raises:
        ValueError: If ``subset`` is not a supported C-MAPSS subset.
        FileNotFoundError: If any required dataset file is missing.
    """
    subset = subset.upper()
    if subset not in {'FD001', 'FD002', 'FD003', 'FD004'}:
        raise ValueError('subset must be one of FD001, FD002, FD003, or FD004')

    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / 'extracted' / 'CMaps'
    else:
        data_dir = Path(data_dir)

    required_files = {
        'train': data_dir / f'train_{subset}.txt',
        'test': data_dir / f'test_{subset}.txt',
        'rul': data_dir / f'RUL_{subset}.txt',
    }
    missing_files = [path for path in required_files.values() if not path.is_file()]
    if missing_files:
        missing_list = ', '.join(str(path) for path in missing_files)
        raise FileNotFoundError(f'Missing required C-MAPSS file(s): {missing_list}')

    columns = (
        ['unit', 'cycle']
        + [f'op_setting_{i}' for i in range(1, 4)]
        + [f'sensor_{i}' for i in range(1, 22)]
    )
    read_options = {'sep': r'\s+', 'header': None, 'names': columns}

    train_df = pd.read_csv(required_files['train'], **read_options)
    test_df = pd.read_csv(required_files['test'], **read_options)
    rul_df = pd.read_csv(required_files['rul'], sep=r'\s+', header=None, names=['RUL'])
    return train_df, test_df, rul_df


def calculate_rul(data, final_rul=None):
    """Add a remaining-useful-life (RUL) column to engine-cycle data.

    With training data, RUL is each unit's last observed cycle minus the
    current cycle. For test data, pass the FD001 RUL values as ``final_rul``;
    these are applied to each unit's last observed test cycle in unit order.
    Returns a copy and leaves ``data`` unchanged.
    """
    result = data.copy()
    last_cycle = result.groupby('unit')['cycle'].transform('max')

    if final_rul is None:
        result['RUL'] = last_cycle - result['cycle']
    else:
        units = np.sort(result['unit'].unique())
        final_rul = np.asarray(final_rul).reshape(-1)
        if len(final_rul) != len(units):
            raise ValueError('final_rul must have one value per unit')
        rul_by_unit = dict(zip(units, final_rul))
        result['RUL'] = (
            last_cycle + result['unit'].map(rul_by_unit) - result['cycle']
        )

    return result
    
