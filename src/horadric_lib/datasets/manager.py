import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd
import structlog
from binpickle import dump, load
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from horadric_lib.config import HoradricConfig
from horadric_lib.logging import configure_logging

logger = structlog.getLogger('DatasetLoader')

DATA_DIR = Path('data')
REGISTRY_FILE = DATA_DIR / 'dataset_registry.json'
SIX_MONTHS_DAYS = 180


def calculate_sha256(filepath: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b''):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_registry() -> dict:
    if REGISTRY_FILE.exists():
        try:
            with open(REGISTRY_FILE) as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning('Registry file corrupted. Creating a new one.')
    return {'files': {}, 'latest': None}


def save_registry(registry: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=4)


def fetch_huggingface_dataset(dataset_id: str, output_path: Path) -> None:
    """Downloads a dataset from HuggingFace and saves it as binpickle."""
    logger.info('downloading_hf_dataset', dataset_id=dataset_id)
    try:
        dataset = load_dataset(dataset_id, split='train')
    except Exception as e:
        logger.error('hf_download_failed', error=str(e))
        raise

    df = dataset.to_pandas(batched=False)
    df = cast(pd.DataFrame, df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump(df, output_path)
    logger.info('hf_dataset_saved', total_size=len(df), path=str(output_path))


def get_fresh_remote_dataset(dataset_id: str, registry: dict) -> Path:
    """Handles hashing and registry updating for a fresh fetch."""
    timestamp_str = datetime.now().strftime('%Y%m%d%H%M')
    safe_name = dataset_id.replace('/', '_')
    filename = f'{timestamp_str}_{safe_name}_full.bpk'
    filepath = DATA_DIR / filename

    fetch_huggingface_dataset(dataset_id, filepath)

    file_size = filepath.stat().st_size
    file_hash = calculate_sha256(filepath)

    registry['files'][filename] = {
        'created_at': datetime.now().isoformat(),
        'size': file_size,
        'hash': file_hash,
        'dataset_id': dataset_id,
    }
    registry['latest'] = filename
    save_registry(registry)

    return filepath


def load_raw_dataset(source: str, force_download: bool = False, target_filename: str | None = None) -> pd.DataFrame:
    """Main ingestion router.

    If `source` is a valid local path, it loads it directly without copying.
    If `source` is a HuggingFace ID, it manages the caching registry.
    """
    source_path = Path(source)
    if source_path.exists() and source_path.is_file():
        logger.info('loading_local_file', path=str(source_path))
        if source_path.suffix == '.parquet':
            return pd.read_parquet(source_path)
        elif source_path.suffix == '.csv':
            return pd.read_csv(source_path)
        elif source_path.suffix == '.bpk':
            return load(source_path)
        else:
            raise ValueError(f'Unsupported local file format: {source_path.suffix}')

    # Fallback to HuggingFace remote handling
    registry = load_registry()

    if force_download:
        filepath = get_fresh_remote_dataset(source, registry)
        return load(filepath)

    filename_to_check = target_filename if target_filename else registry.get('latest')

    if not filename_to_check or filename_to_check not in registry['files']:
        filepath = get_fresh_remote_dataset(source, registry)
        return load(filepath)

    filepath = DATA_DIR / filename_to_check
    file_info = registry['files'][filename_to_check]

    if not filepath.exists() or filepath.stat().st_size != file_info['size']:
        filepath = get_fresh_remote_dataset(source, registry)
        return load(filepath)

    age_days = (datetime.now() - datetime.fromisoformat(file_info['created_at'])).days
    if age_days > SIX_MONTHS_DAYS:
        choice = input(f'Cache {filename_to_check} is > 6 months old. Redownload? [y/N]: ').lower()
        if choice == 'y':
            filepath = get_fresh_remote_dataset(source, registry)

    logger.info('loading_cached_dataset', filename=filename_to_check)
    return load(filepath)


def create_dataset_slice(
    df: pd.DataFrame,
    dataset_name: str,
    slice_name: str,
    sort_col: str | None = None,
    top_p: float = 1.0,
    ascending: bool = False,
    test_size: float = 0.0,
    random_seed: int = 42,
) -> Path:
    """Slices the DataFrame dynamically and saves to {DATA_DIR}/{dataset_name}/{slice_name}/[train|test].parquet."""
    df = df.reset_index(names='original_row_id')

    if sort_col and sort_col in df.columns:
        logger.info('sorting_data', col=sort_col, top_p=top_p, ascending=ascending)
        df_slice = df.sort_values(sort_col, ascending=ascending).head(int(len(df) * top_p))
    else:
        if top_p < 1.0:
            logger.info('random_sampling_data', top_p=top_p)
            df_slice = df.sample(frac=top_p, random_state=random_seed)
        else:
            df_slice = df

    output_dir = DATA_DIR / dataset_name / slice_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if test_size > 0.0:
        logger.info('splitting_train_test', test_size=test_size)
        train_df, test_df = train_test_split(df_slice, test_size=test_size, random_state=random_seed)

        train_path = output_dir / 'train.parquet'
        test_path = output_dir / 'test.parquet'

        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        logger.info('saved_splits', train=str(train_path), test=str(test_path))
    else:
        single_path = output_dir / 'full_slice.parquet'
        df_slice.to_parquet(single_path, index=False)
        logger.info('saved_single_slice', path=str(single_path))

    return output_dir


def main() -> None:
    conf = HoradricConfig.load_or_create()
    configure_logging(log_dir=conf.log_dir)

    parser = argparse.ArgumentParser(description='Generic Dataset Fetcher and Slicer')
    parser.add_argument(
        '--source',
        type=str,
        default=conf.default_hf_source,
        help='HF ID or Local file path',
    )
    parser.add_argument('--data-dir', type=str, default=conf.data_dir, help='Path to the data')
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='SATBench',
        help='Root folder name for this dataset',
    )
    parser.add_argument(
        '--slice-name',
        type=str,
        default='hard_10_percent',
        help='Subfolder name for this slice',
    )
    parser.add_argument(
        '--sort-col',
        type=str,
        default='num_clauses',
        help='Column to sort by before slicing',
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.10,
        help='Percentage of data to keep (e.g., 0.10 for 10%)',
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.20,
        help='Percentage to reserve for test.parquet',
    )
    parser.add_argument(
        '--save-config',
        action='store_true',
        help='Save these CLI args back to horadric_conf.json',
    )
    parser.add_argument('--force-download', action='store_true')

    try:
        args = parser.parse_args()

        raw_df = load_raw_dataset(source=args.source, force_download=args.force_download)
        if args.dataset_name == 'SATBench':
            raw_df = raw_df.rename(columns={'readable': 'original_cnf', 'scenario': 'scenario_text'})

        out_dir = create_dataset_slice(
            df=raw_df,
            dataset_name=args.dataset_name,
            slice_name=args.slice_name,
            sort_col=args.sort_col,
            top_p=args.top_p,
            ascending=False,  # False to get the "hardest" (highest num_clauses)
            test_size=args.test_size,
        )

        logger.info('pipeline_complete', target_directory=str(out_dir))
    except FileNotFoundError as e:
        logger.error(f'Could not find the specified file: {e}')
        print('\n')
        parser.print_help()
        sys.exit(1)

    except ValueError as e:
        logger.error(f'Invalid input provided: {e}')
        print('\n')
        parser.print_help()
        sys.exit(1)

    except Exception as e:
        logger.exception(f'A fatal error occurred: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
