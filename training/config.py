from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DatasetKind = Literal["uci_har", "csv"]
ModelKind = Literal["rf", "xgboost", "cnn1d", "lstm"]
CsvLayout = Literal["tabular", "grouped_windows", "har_processor"]
HarNodeMode = Literal["single_csv", "multi_files", "four_files"]


@dataclass
class TrainConfig:
    """Paths and hyperparameters; edit the CFG block in train.py."""

    project_root: Path | None = None

    dataset: DatasetKind = "uci_har"
    uci_dataset_root: Path = Path("UCI HAR Dataset")

    csv_path: Path | None = None
    csv_target_column: str = "label"
    csv_group_column: str | None = None
    csv_time_column: str | None = None
    csv_layout: CsvLayout = "tabular"
    csv_feature_columns: list[str] | None = None
    csv_exclude_columns: tuple[str, ...] = ()
    csv_window_length: int = 128
    csv_window_stride: int = 64
    csv_split_by_group: bool = True
    csv_time_format: str | None = None
    csv_time_utc: bool = False

    csv_har_node_mode: HarNodeMode = "single_csv"
    csv_har_node_files: tuple[Path, ...] = ()
    csv_node_column: str = "Node"
    csv_har_node_values: tuple[str, ...] = ("1", "2", "3", "4", "5")
    csv_har_sensor_columns: tuple[str, ...] = ("Ax", "Ay", "Az", "Gx", "Gy", "Gz")
    csv_fs_hz: float = 50.0
    csv_har_lpf_cutoff_hz: float = 20.0
    csv_har_merge_tolerance_ms: float = 25.0
    csv_har_use_body_acc: bool = False

    model: ModelKind = "rf"

    random_state: int = 42
    test_size: float = 0.2

    n_estimators: int = 200
    max_depth: int | None = None
    xgb_tune_max_depth: bool = False
    xgb_tuning_method: str = "optuna"
    xgb_depth_candidates: tuple[int, ...] = (3, 4, 5, 6, 8, 10)
    xgb_cv_folds: int = 3
    xgb_scoring: str = "accuracy"
    xgb_optuna_trials: int = 20

    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3

    save_model: bool = True
    models_dir: Path = Path("models")
