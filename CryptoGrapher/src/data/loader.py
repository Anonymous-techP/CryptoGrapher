"""Data loader for CRYPTO_1D_ALL."""
import os
import pickle
import sys
import numpy as np


DEFAULT_MARKET = 'CRYPTO_1D_ALL'
DEFAULT_DATA_PATH = '../dataset'


def load_data(data_path: str = DEFAULT_DATA_PATH,
              market_name: str = DEFAULT_MARKET) -> dict:
    dataset_path = os.path.join(data_path, market_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    def _load(name):
        with open(os.path.join(dataset_path, name), "rb") as f:
            return pickle.load(f)

    try:
        eod_data = _load("eod_data.pkl")
    except ModuleNotFoundError as e:
        if 'numpy._core' not in str(e):
            raise
        sys.modules['numpy._core'] = np.core
        eod_data = _load("eod_data.pkl")

    mask_data = _load("mask_data.pkl")
    gt_data = _load("gt_data.pkl")
    price_data = _load("price_data.pkl")

    trade_dates = min(mask_data.shape[1], gt_data.shape[1])
    if trade_dates > 0 and np.all(gt_data[:, trade_dates - 1] == 0):
        trade_dates -= 1

    return {
        'eod_data': eod_data,
        'mask_data': mask_data,
        'gt_data': gt_data,
        'price_data': price_data,
        'trade_dates': trade_dates,
        'stock_num': eod_data.shape[0],
        'fea_num': eod_data.shape[2],
    }


def get_batch(data: dict, offset: int, lookback: int = 30, steps: int = 1):
    eod = data['eod_data']
    mask = data['mask_data']
    price = data['price_data']
    gt = data['gt_data']

    mask_window = np.min(mask[:, offset:offset + lookback + steps], axis=1)

    x = eod[:, offset:offset + lookback, :].astype(np.float32)
    return (
        x,
        np.expand_dims(mask_window, axis=1).astype(np.float32),
        np.expand_dims(price[:, offset + lookback - 1], axis=1).astype(np.float32),
        np.expand_dims(gt[:, offset + lookback + steps - 1], axis=1).astype(np.float32),
    )
