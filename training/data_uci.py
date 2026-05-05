from __future__ import annotations

from pathlib import Path

import numpy as np

RAW_FILES_ORDER = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)


def _read_y(path: Path) -> np.ndarray:
    y = np.loadtxt(path, dtype=np.int64)
    if y.ndim > 1:
        y = y.ravel()
    return y - 1


def load_uci_tabular(uci_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_dir = uci_root / "train"
    test_dir = uci_root / "test"
    x_tr = np.loadtxt(train_dir / "X_train.txt")
    x_te = np.loadtxt(test_dir / "X_test.txt")
    y_tr = _read_y(train_dir / "y_train.txt")
    y_te = _read_y(test_dir / "y_test.txt")
    return x_tr, x_te, y_tr, y_te


def load_uci_raw_windows(uci_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Her örnek (zaman_adımı=128, kanal=9) — Conv1D/LSTM için."""
    train_dir = uci_root / "train" / "Inertial Signals"
    test_dir = uci_root / "test" / "Inertial Signals"

    def stack_split(split_dir: Path, split: str) -> np.ndarray:
        channels = []
        for name in RAW_FILES_ORDER:
            p = split_dir / f"{name}_{split}.txt"
            arr = np.loadtxt(p)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            channels.append(arr[:, :, np.newaxis])
        return np.concatenate(channels, axis=2)

    x_tr = stack_split(train_dir, "train")
    x_te = stack_split(test_dir, "test")
    y_tr = _read_y(uci_root / "train" / "y_train.txt")
    y_te = _read_y(uci_root / "test" / "y_test.txt")
    return x_tr, x_te, y_tr, y_te


def normalize_channels(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (x_train - mean) / std, (x_test - mean) / std
