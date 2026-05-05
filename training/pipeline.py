from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

from training.config import TrainConfig
from training.data_prep.processor import (
    load_csv_grouped_windows,
    load_csv_tabular,
    process_har_fusion_pipeline,
)
from training.data_uci import load_uci_raw_windows, load_uci_tabular, normalize_channels
from training.models import train_keras, train_sklearn


def _root(cfg: TrainConfig) -> Path:
    return (cfg.project_root or Path.cwd()).resolve()


def _ensure_paths(cfg: TrainConfig) -> Path:
    root = _root(cfg)
    return (root / cfg.uci_dataset_root).resolve()


def _dataset_name(cfg: TrainConfig, csv_path: Path | None) -> str:
    if cfg.dataset == "csv" and csv_path is not None:
        return csv_path.stem
    return cfg.dataset


def _run_output_dirs(root: Path, cfg: TrainConfig, dataset_name: str, model_name: str) -> tuple[Path, Path]:
    base_dir = (root / cfg.models_dir).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{dataset_name}_{model_name}_{stamp}"
    scores_dir = run_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, scores_dir


def _ensure_project_layout(root: Path) -> None:
    for folder in (
        root / "data" / "raw",
        root / "data" / "processed",
        root / "notebooks",
        root / "src" / "training",
        root / "src" / "preprocessing",
        root / "src" / "utils",
    ):
        folder.mkdir(parents=True, exist_ok=True)


def _build_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("training.pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _load_uci_activity_labels(uci_root: Path) -> list[str] | None:
    labels_path = uci_root / "activity_labels.txt"
    if not labels_path.is_file():
        return None
    names: list[str] = []
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        names.append(parts[1])
    return names or None


def _resolve_class_names(
    cfg: TrainConfig,
    label_names: list[str] | None,
    y_true: np.ndarray,
    y_score: np.ndarray,
    uci_root: Path | None,
) -> list[str]:
    if label_names is not None and len(label_names) == y_score.shape[1]:
        return label_names
    if cfg.dataset == "uci_har" and uci_root is not None:
        uci_names = _load_uci_activity_labels(uci_root)
        if uci_names is not None and len(uci_names) == y_score.shape[1]:
            return uci_names
    unique_labels = sorted(set(int(v) for v in np.unique(y_true)))
    if len(unique_labels) == y_score.shape[1]:
        return [str(v) for v in unique_labels]
    return [str(i) for i in range(y_score.shape[1])]


def _save_ovr_roc_plot(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_labels: list[str],
    out_path: Path,
    model_name: str,
    logger: logging.Logger,
) -> tuple[Path, dict[str, float]]:
    n_classes = y_score.shape[1]
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    fig, ax = plt.subplots(figsize=(10, 7))
    auc_by_class: dict[str, float] = {}
    for idx in range(n_classes):
        y_true_cls = y_true_bin[:, idx]
        # ROC requires both positive and negative samples for each one-vs-rest slice.
        if len(np.unique(y_true_cls)) < 2:
            logger.info("[ROC] class=%s skipped (test set has a single label state).", class_labels[idx])
            continue
        fpr, tpr, _ = roc_curve(y_true_cls, y_score[:, idx])
        class_auc = auc(fpr, tpr)
        logger.info("[ROC] class=%s | AUC=%.4f", class_labels[idx], class_auc)
        ax.plot(fpr, tpr, lw=2, label=f"{class_labels[idx]} (AUC={class_auc:.3f})")
        auc_by_class[class_labels[idx]] = float(class_auc)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_title(f"OVR ROC Curves ({model_name})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    if not auc_by_class:
        logger.info("[ROC] No class had both positive and negative samples; saved baseline-only ROC plot.")
    logger.info("saved ROC plot: %s", out_path)
    return out_path, auc_by_class


def _save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
    out_path: Path,
    logger: logging.Logger,
) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_labels)))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    logger.info("saved confusion matrix: %s", out_path)
    return out_path


def _write_metrics_file(
    out_path: Path,
    accuracy: float,
    report: str,
    auc_by_class: dict[str, float],
    logger: logging.Logger,
) -> Path:
    lines = [
        f"accuracy: {accuracy:.4f}",
        "",
        "classification_report:",
        report.rstrip(),
        "",
        "roc_auc_by_class:",
    ]
    if auc_by_class:
        lines.extend(f"{name}: {score:.6f}" for name, score in auc_by_class.items())
    else:
        lines.append("N/A")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("saved metrics: %s", out_path)
    return out_path


def _append_summary_row(summary_path: Path, row: dict[str, str]) -> None:
    fieldnames = list(row.keys())
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run(cfg: TrainConfig) -> int:
    root = _root(cfg)
    model_name = cfg.model
    xgboost_har = cfg.dataset == "csv" and cfg.csv_layout == "har_processor" and model_name == "xgboost"
    label_names: list[str] | None = None
    csv_path: Path | None = None
    uci: Path | None = None

    if cfg.dataset == "csv":
        if cfg.csv_path is None:
            raise ValueError('dataset="csv" requires csv_path.')
        path = (
            (root / cfg.csv_path).resolve()
            if not cfg.csv_path.is_absolute()
            else cfg.csv_path
        )
        csv_path = path
        exclude = frozenset(cfg.csv_exclude_columns)

        if cfg.csv_layout == "har_processor":
            if model_name in ("cnn1d", "lstm"):
                raise ValueError("har_processor outputs tabular feature vectors; use rf or xgboost.")
            if not cfg.csv_group_column or not cfg.csv_time_column:
                raise ValueError("har_processor requires csv_group_column and csv_time_column.")
            x_tr, x_te, y_tr, y_te, _le = process_har_fusion_pipeline(
                project_root=root,
                csv_path=path,
                node_csv_paths=cfg.csv_har_node_files,
                node_mode=cfg.csv_har_node_mode,
                node_column=cfg.csv_node_column,
                node_values=cfg.csv_har_node_values,
                time_column=cfg.csv_time_column,
                target_column=cfg.csv_target_column,
                group_column=cfg.csv_group_column,
                sensor_columns=cfg.csv_har_sensor_columns,
                exclude_columns=exclude,
                fs_hz=cfg.csv_fs_hz,
                lpf_cutoff_hz=cfg.csv_har_lpf_cutoff_hz,
                use_body_acc_from_triax=cfg.csv_har_use_body_acc,
                merge_tolerance_ms=cfg.csv_har_merge_tolerance_ms,
                test_size=cfg.test_size,
                random_state=cfg.random_state,
                split_by_group=cfg.csv_split_by_group,
                time_format=cfg.csv_time_format,
                time_utc=cfg.csv_time_utc,
            )
            label_names = [str(v) for v in _le.classes_]
        elif cfg.csv_layout == "grouped_windows":
            if not cfg.csv_group_column:
                raise ValueError('grouped_windows requires csv_group_column.')
            x_tr, x_te, y_tr, y_te, _le = load_csv_grouped_windows(
                path,
                cfg.csv_target_column,
                cfg.csv_group_column,
                cfg.csv_time_column,
                cfg.csv_feature_columns,
                exclude,
                cfg.csv_window_length,
                cfg.csv_window_stride,
                cfg.test_size,
                cfg.random_state,
                cfg.csv_split_by_group,
                time_format=cfg.csv_time_format,
                time_utc=cfg.csv_time_utc,
            )
            label_names = [str(v) for v in _le.classes_]
            if model_name in ("rf", "xgboost"):
                x_tr = x_tr.reshape(x_tr.shape[0], -1)
                x_te = x_te.reshape(x_te.shape[0], -1)
            else:
                x_tr, x_te = normalize_channels(x_tr, x_te)
        else:
            if model_name in ("cnn1d", "lstm"):
                raise ValueError(
                    "CSV tabular does not support CNN1D/LSTM. "
                    "Use grouped_windows or uci_har, or har_processor with tree models."
                )
            x_tr, x_te, y_tr, y_te, _le = load_csv_tabular(
                path,
                cfg.csv_target_column,
                cfg.test_size,
                cfg.random_state,
                exclude,
            )
            label_names = [str(v) for v in _le.classes_]
    else:
        uci = _ensure_paths(cfg)
        if not (uci / "train" / "X_train.txt").is_file():
            raise FileNotFoundError(f"UCI HAR root missing or invalid: {uci}")
        if model_name in ("cnn1d", "lstm"):
            x_tr, x_te, y_tr, y_te = load_uci_raw_windows(uci)
            x_tr, x_te = normalize_channels(x_tr, x_te)
        else:
            x_tr, x_te, y_tr, y_te = load_uci_tabular(uci)

    layout = cfg.csv_layout if cfg.dataset == "csv" else "-"
    _ensure_project_layout(root)
    dataset_name = _dataset_name(cfg, csv_path)
    run_dir, scores_dir = _run_output_dirs(root, cfg, dataset_name, model_name)
    logger = _build_logger(run_dir)
    logger.info("dataset=%s | csv_layout=%s | model=%s", cfg.dataset, layout, model_name)
    logger.info("train %s | test %s", x_tr.shape, x_te.shape)
    roc_path = scores_dir / "roc_auc.png"
    cm_path = scores_dir / "confusion_matrix.png"
    metrics_path = scores_dir / "metrics.txt"
    summary_path = (root / cfg.models_dir).resolve() / "summary_results.csv"
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if model_name in ("rf", "xgboost"):
        est, _metrics = train_sklearn(
            model_name,
            x_tr,
            y_tr,
            x_te,
            y_te,
            cfg.n_estimators,
            cfg.max_depth,
            cfg.random_state,
            xgboost_har_features=xgboost_har,
            xgb_tune_max_depth=cfg.xgb_tune_max_depth,
            xgb_depth_candidates=cfg.xgb_depth_candidates,
            xgb_cv_folds=cfg.xgb_cv_folds,
            xgb_scoring=cfg.xgb_scoring,
        )
        y_pred = est.predict(x_te)
        y_score = est.predict_proba(x_te)
        class_names = _resolve_class_names(cfg, label_names, y_te, y_score, uci)
        report = classification_report(y_te, y_pred, target_names=class_names, digits=4)
        acc = float(accuracy_score(y_te, y_pred))
        logger.info("test accuracy: %.4f", acc)
        logger.info("\n%s", report)
        _, auc_by_class = _save_ovr_roc_plot(
            y_true=y_te,
            y_score=y_score,
            class_labels=class_names,
            out_path=roc_path,
            model_name=model_name,
            logger=logger,
        )
        _save_confusion_matrix_plot(y_te, y_pred, class_names, cm_path, logger)
        _write_metrics_file(metrics_path, acc, report, auc_by_class, logger)
        model_path = run_dir / f"{model_name}.joblib"
        joblib.dump(est, model_path)
        logger.info("saved model: %s", model_path)
    else:
        model, _metrics = train_keras(
            model_name,
            x_tr,
            y_tr,
            x_te,
            y_te,
            cfg.epochs,
            cfg.batch_size,
            cfg.learning_rate,
            cfg.random_state,
        )
        y_score = model.predict(x_te, batch_size=cfg.batch_size, verbose=0)
        y_pred = np.argmax(y_score, axis=1)
        class_names = _resolve_class_names(cfg, label_names, y_te, y_score, uci)
        report = classification_report(y_te, y_pred, target_names=class_names, digits=4)
        acc = float(accuracy_score(y_te, y_pred))
        logger.info("test accuracy: %.4f", acc)
        logger.info("\n%s", report)
        _, auc_by_class = _save_ovr_roc_plot(
            y_true=y_te,
            y_score=y_score,
            class_labels=class_names,
            out_path=roc_path,
            model_name=model_name,
            logger=logger,
        )
        _save_confusion_matrix_plot(y_te, y_pred, class_names, cm_path, logger)
        _write_metrics_file(metrics_path, acc, report, auc_by_class, logger)
        model_path = run_dir / f"{model_name}.keras"
        model.save(model_path)
        logger.info("saved model: %s", model_path)

    _append_summary_row(
        summary_path,
        {
            "timestamp": start_ts,
            "run_dir": str(run_dir),
            "dataset": cfg.dataset,
            "dataset_name": dataset_name,
            "model": model_name,
            "csv_layout": layout,
            "accuracy": f"{acc:.6f}",
            "n_estimators": str(cfg.n_estimators),
            "max_depth": str(getattr(est, "max_depth", cfg.max_depth)) if model_name in ("rf", "xgboost") else "",
            "xgb_tune_max_depth": str(cfg.xgb_tune_max_depth),
            "xgb_depth_candidates": "|".join(str(v) for v in cfg.xgb_depth_candidates),
            "xgb_cv_folds": str(cfg.xgb_cv_folds),
            "xgb_scoring": cfg.xgb_scoring,
            "csv_window_length": str(cfg.csv_window_length),
            "csv_window_stride": str(cfg.csv_window_stride),
            "metrics_file": str(metrics_path),
            "roc_file": str(roc_path),
            "confusion_matrix_file": str(cm_path),
        },
    )
    logger.info("updated summary csv: %s", summary_path)

    return 0
