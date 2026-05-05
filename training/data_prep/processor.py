"""
UCI HAR README ile uyumlu is akisi: 50 Hz, 128 ornek / 2.56 s, %%50 overlap (stride 64),
Butterworth alcak geciren filtre, coklu Node senkronu, zaman + frekans ozellikleri.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- UCI HAR README referansi ---
FS_HZ_DEFAULT = 50.0
WINDOW_SAMPLES_UCI = 128
WINDOW_STRIDE_UCI = 64
GRAVITY_CUTOFF_HZ = 0.3


def coerce_time_column(
    df: pd.DataFrame,
    column: str,
    *,
    time_format: str | None = None,
    utc: bool = False,
) -> None:
    if column not in df.columns:
        raise KeyError(column)
    s = df[column]
    if pd.api.types.is_datetime64_any_dtype(s):
        return
    if pd.api.types.is_numeric_dtype(s):
        return
    parsed = pd.to_datetime(s, format=time_format, errors="coerce", utc=utc)
    if parsed.isna().any():
        n_bad = int(parsed.isna().sum())
        raise ValueError(
            f"Time column {column!r}: {n_bad} value(s) could not parse as datetime."
        )
    df[column] = parsed


def butter_lowpass_filtfilt(
    x: np.ndarray,
    cutoff_hz: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Sifir faz Butterworth alcak geciren (gurultu / yuksek frekans baskilama)."""
    if x.size == 0:
        return x
    if cutoff_hz <= 0 or cutoff_hz >= fs / 2:
        raise ValueError(f"cutoff_hz must be in (0, Nyquist). Got {cutoff_hz}, fs={fs}")
    wn = cutoff_hz / (fs / 2.0)
    b, a = signal.butter(order, wn, btype="low")
    padlen = 3 * max(len(a), len(b))
    if x.shape[0] <= padlen:
        return signal.filtfilt(b, a, x)
    return signal.filtfilt(b, a, x)


def butter_lowpass_gravity_component(acc_axis: np.ndarray, fs: float, order: int = 4) -> np.ndarray:
    """README: 0.3 Hz alti yer cekimi bileseni (tek eksen)."""
    return butter_lowpass_filtfilt(acc_axis, GRAVITY_CUTOFF_HZ, fs, order=order)


def body_acceleration_from_total(total_triax: np.ndarray, fs: float) -> np.ndarray:
    """total_triax (T,3) -> body (T,3) = total - gravity, gravity eksen bazli 0.3 Hz LPF."""
    g = np.stack(
        [butter_lowpass_gravity_component(total_triax[:, i], fs) for i in range(3)],
        axis=1,
    )
    return total_triax - g


def resample_to_uniform_fs(
    df: pd.DataFrame,
    time_col: str,
    value_cols: list[str],
    fs_hz: float,
) -> pd.DataFrame:
    """Zaman damgasina gore fs_hz sabit aralikli seri (UCI 50 Hz)."""
    d = df[[time_col] + value_cols].dropna(subset=[time_col]).copy()
    d = d.sort_values(time_col).drop_duplicates(subset=[time_col], keep="last")
    d = d.set_index(time_col)
    period = f"{int(round(1000 / fs_hz))}ms"
    out = d.resample(period).mean().interpolate(limit_direction="both")
    out = out.reset_index()
    if out.columns[0] != time_col:
        out = out.rename(columns={out.columns[0]: time_col})
    return out


def resample_uniform_har(
    df: pd.DataFrame,
    time_column: str,
    feature_cols: list[str],
    target_column: str,
    fs_hz: float,
) -> pd.DataFrame:
    """Sensor ortalama + etiket ffill ile 50 Hz hizaya (README). Sayisal Time -> saniye datetime."""
    cols = [time_column] + feature_cols + [target_column]
    d = df[cols].dropna(subset=[time_column]).sort_values(time_column)
    d = d.drop_duplicates(subset=[time_column], keep="last")
    if pd.api.types.is_datetime64_any_dtype(d[time_column]):
        idx = d[time_column]
    else:
        t = d[time_column].astype(np.float64).to_numpy()
        t0 = float(np.min(t))
        sec_from_start = (t - t0) / float(fs_hz)
        idx = pd.Timestamp("2000-01-01") + pd.to_timedelta(sec_from_start, unit="s")
    d = d.drop(columns=[time_column]).set_index(idx)
    period = f"{int(round(1000.0 / fs_hz))}ms"
    xnum = d[feature_cols].resample(period).mean().interpolate(limit_direction="both")
    ycat = d[[target_column]].resample(period).ffill()
    out = xnum.join(ycat, how="outer")
    for c in feature_cols:
        out[c] = out[c].interpolate(limit_direction="both")
    out[target_column] = out[target_column].ffill().bfill()
    out = out.reset_index().rename(columns={"index": time_column})
    if out.columns[0] != time_column and len(out.columns):
        out = out.rename(columns={out.columns[0]: time_column})
    return out


def sliding_windows_2d(matrix: np.ndarray, window_length: int, window_stride: int) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError(f"Expected (T, C), got {matrix.shape}")
    t, c = matrix.shape
    if t < window_length:
        return np.empty((0, window_length, c), dtype=np.float64)
    starts = list(range(0, t - window_length + 1, window_stride))
    if not starts:
        return np.empty((0, window_length, c), dtype=np.float64)
    return np.stack([matrix[s : s + window_length] for s in starts], axis=0)


def _stats_time_axis(t: np.ndarray) -> list[float]:
    t = np.asarray(t, dtype=np.float64)
    return [
        float(np.mean(t)),
        float(np.std(t)),
        float(np.max(t)),
        float(np.min(t)),
        float(np.sum(t * t) / max(len(t), 1)),
    ]


def _stats_freq_axis(t: np.ndarray) -> list[float]:
    mag = np.abs(np.fft.rfft(t))
    if mag.size == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    return _stats_time_axis(mag)


def signal_magnitude_area_triax(ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> float:
    """SMA: ortalama (|ax|+|ay|+|az|) zaman boyunca."""
    return float(np.mean(np.abs(ax) + np.abs(ay) + np.abs(az)))


def extract_window_time_frequency_features(window: np.ndarray) -> np.ndarray:
    """
    window: (WINDOW_SAMPLES_UCI, C)
    Kanal basina zaman + |RFFT| uzerinde mean, std, max, min, energy;
    her 3 kanallik blokta SMA (triaks varsayimi).
    """
    w = np.asarray(window, dtype=np.float64)
    _, c = w.shape
    feats: list[float] = []
    for ci in range(c):
        col = w[:, ci]
        feats.extend(_stats_time_axis(col))
        feats.extend(_stats_freq_axis(col))
    for start in range(0, c - 2, 3):
        feats.append(
            signal_magnitude_area_triax(w[:, start], w[:, start + 1], w[:, start + 2])
        )
    if c % 3 != 0:
        for ci in range(c - (c % 3), c):
            feats.append(float(np.mean(np.abs(w[:, ci]))))
    return np.asarray(feats, dtype=np.float64)


def _merge_asof_tolerance(series: pd.Series, tolerance_ms: float) -> pd.Timedelta | float:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.Timedelta(milliseconds=tolerance_ms)
    return float(tolerance_ms) / 1000.0


def fuse_nodes_from_node_files(
    paths: tuple[Path, ...],
    time_col: str,
    sensor_cols: tuple[str, ...],
    tolerance_ms: float,
    time_format: str | None,
    time_utc: bool,
    expected_node_count: int = 5,
) -> pd.DataFrame:
    """Her Node CSV'sini time_col uzerinden merge_asof ile yatay senkronlar."""
    if expected_node_count > 0 and len(paths) != expected_node_count:
        raise ValueError(
            f"Expected {expected_node_count} node CSV files, got {len(paths)}."
        )
    if len(paths) < 2:
        raise ValueError("fuse_nodes_from_node_files expects at least 2 CSV paths.")
    frames: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        if time_col not in df.columns:
            raise KeyError(f"{time_col} missing in {p}")
        for s in sensor_cols:
            if s not in df.columns:
                raise KeyError(f"{s} missing in {p}")
        coerce_time_column(df, time_col, time_format=time_format, utc=time_utc)
        frames.append(df[[time_col] + list(sensor_cols)].copy())

    base = frames[0].sort_values(time_col).rename(
        columns={s: f"n1_{s}" for s in sensor_cols}
    )
    tol = _merge_asof_tolerance(base[time_col], tolerance_ms)
    for idx, fr in enumerate(frames[1:], start=2):
        right = fr.sort_values(time_col).rename(
            columns={s: f"n{idx}_{s}" for s in sensor_cols}
        )
        base = pd.merge_asof(
            base,
            right,
            on=time_col,
            direction="nearest",
            tolerance=tol,
        )
    return base.sort_values(time_col).reset_index(drop=True)


def fuse_nodes_from_single_csv(
    df: pd.DataFrame,
    node_col: str,
    node_values: tuple[str, ...],
    time_col: str,
    sensor_cols: tuple[str, ...],
    tolerance_ms: float,
    time_format: str | None,
    time_utc: bool,
) -> pd.DataFrame:
    """Tek CSV: Node sutununa gore ayristirip zaman ekseninde yan yana birlestirir."""
    if node_col not in df.columns:
        raise KeyError(node_col)
    coerce_time_column(df, time_col, time_format=time_format, utc=time_utc)
    parts: list[pd.DataFrame] = []
    for i, nv in enumerate(node_values, start=1):
        sub = df[df[node_col].astype(str) == str(nv)][[time_col] + list(sensor_cols)].copy()
        if sub.empty:
            raise ValueError(f"No rows for {node_col}={nv!r}")
        sub = sub.sort_values(time_col).rename(
            columns={s: f"n{i}_{s}" for s in sensor_cols}
        )
        parts.append(sub)
    base = parts[0].sort_values(time_col)
    tol = _merge_asof_tolerance(base[time_col], tolerance_ms)
    for right in parts[1:]:
        base = pd.merge_asof(
            base,
            right.sort_values(time_col),
            on=time_col,
            direction="nearest",
            tolerance=tol,
        )
    return base.sort_values(time_col).reset_index(drop=True)


def _numeric_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    exclude: frozenset[str],
) -> list[str]:
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    out = [c for c in num if c != target_column and c not in exclude]
    if not out:
        raise ValueError("No numeric feature columns after target/exclude.")
    return out


def read_raw_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_csv_tabular(
    csv_path: Path,
    target_column: str,
    test_size: float,
    random_state: int,
    exclude_columns: frozenset[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    df = read_raw_csv(csv_path)
    ex = exclude_columns or frozenset()
    if target_column not in df.columns:
        raise KeyError(f"Missing target column {target_column!r}. Columns: {list(df.columns)}")
    missing_ex = ex - frozenset(df.columns)
    if missing_ex:
        raise KeyError(f"exclude_columns not in CSV: {sorted(missing_ex)}")
    feat_cols = _numeric_feature_columns(df, target_column, ex)
    x = df[feat_cols].to_numpy(dtype=np.float64)
    le = LabelEncoder()
    y = le.fit_transform(df[target_column].astype(str))
    _, counts = np.unique(y, return_counts=True)
    stratify = y if counts.min() >= 2 else None
    return train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=stratify
    ) + (le,)


def grouped_sliding_windows(
    df: pd.DataFrame,
    group_column: str,
    feature_columns: list[str],
    target_column: str,
    window_length: int,
    window_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_list: list[np.ndarray] = []
    y_list: list[str] = []
    gid_list: list[object] = []
    for gid, sub in df.groupby(group_column, sort=False):
        mat = sub[feature_columns].to_numpy(dtype=np.float64)
        y_sub = sub[target_column]
        wins = sliding_windows_2d(mat, window_length, window_stride)
        if wins.shape[0] == 0:
            continue
        for k in range(wins.shape[0]):
            end_row = (k * window_stride) + window_length - 1
            x_list.append(wins[k])
            y_list.append(str(y_sub.iloc[end_row]))
            gid_list.append(gid)
    if not x_list:
        return (
            np.empty((0, window_length, len(feature_columns)), dtype=np.float64),
            np.array([], dtype=object),
            np.array([], dtype=object),
        )
    return (
        np.stack(x_list, axis=0),
        np.array(y_list, dtype=object),
        np.array(gid_list, dtype=object),
    )


def load_csv_grouped_windows(
    csv_path: Path,
    target_column: str,
    group_column: str,
    time_column: str | None,
    feature_columns: list[str] | None,
    exclude_columns: frozenset[str],
    window_length: int,
    window_stride: int,
    test_size: float,
    random_state: int,
    split_by_group: bool,
    time_format: str | None = None,
    time_utc: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    if window_length < 2 or window_stride < 1:
        raise ValueError("Invalid window_length or window_stride.")
    df = read_raw_csv(csv_path)
    for col in (target_column, group_column):
        if col not in df.columns:
            raise KeyError(f"Missing column {col!r}. Columns: {list(df.columns)}")
    if time_column is not None and time_column not in df.columns:
        raise KeyError(time_column)
    ex = exclude_columns | {target_column, group_column}
    if time_column:
        ex = ex | {time_column}
    if feature_columns is not None:
        for c in feature_columns:
            if c not in df.columns:
                raise KeyError(c)
        feat_cols = [c for c in feature_columns if c not in ex]
        non_num = [c for c in feat_cols if c not in df.select_dtypes(include=[np.number]).columns]
        if non_num:
            raise ValueError(f"Non-numeric feature columns: {non_num}")
    else:
        feat_cols = _numeric_feature_columns(df, target_column, frozenset(ex))
    if time_column is not None:
        coerce_time_column(df, time_column, time_format=time_format, utc=time_utc)
    sort_cols = [group_column] + ([time_column] if time_column else [])
    df = df.sort_values(sort_cols).reset_index(drop=True)
    X, y_raw_arr, gids = grouped_sliding_windows(
        df, group_column, feat_cols, target_column, window_length, window_stride
    )
    if X.shape[0] == 0:
        raise ValueError("No windows built.")
    if split_by_group:
        unique_groups = np.unique(gids)
        if len(unique_groups) < 2:
            raise ValueError("split_by_group needs at least 2 distinct group IDs.")
        g_tr, g_te = train_test_split(unique_groups, test_size=test_size, random_state=random_state)
        set_tr, set_te = set(g_tr.tolist()), set(g_te.tolist())
        m_tr = np.array([g in set_tr for g in gids], dtype=bool)
        m_te = np.array([g in set_te for g in gids], dtype=bool)
    else:
        n = X.shape[0]
        idx = np.arange(n)
        i_tr, i_te = train_test_split(idx, test_size=test_size, random_state=random_state)
        m_tr, m_te = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        m_tr[i_tr], m_te[i_te] = True, True
    x_tr, x_te = X[m_tr], X[m_te]
    y_tr_raw, y_te_raw = y_raw_arr[m_tr], y_raw_arr[m_te]
    le = LabelEncoder()
    y_tr = le.fit_transform(y_tr_raw)
    try:
        y_te = le.transform(y_te_raw)
    except ValueError as e:
        raise ValueError(
            "Test labels not seen in train. Use split_by_group=True or adjust data."
        ) from e
    return x_tr, x_te, y_tr, y_te, le


NodeMode = Literal["single_csv", "multi_files", "four_files"]


def _resolve(root: Path, p: Path | None) -> Path | None:
    if p is None:
        return None
    return (root / p).resolve() if not p.is_absolute() else p


def process_har_fusion_pipeline(
    *,
    project_root: Path,
    csv_path: Path | None,
    node_csv_paths: tuple[Path, ...],
    node_mode: NodeMode,
    node_column: str,
    node_values: tuple[str, ...],
    time_column: str,
    target_column: str,
    group_column: str,
    sensor_columns: tuple[str, ...],
    exclude_columns: frozenset[str],
    fs_hz: float,
    lpf_cutoff_hz: float,
    use_body_acc_from_triax: bool,
    merge_tolerance_ms: float,
    test_size: float,
    random_state: int,
    split_by_group: bool,
    time_format: str | None,
    time_utc: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Ham CSV -> Node senkronu -> 50 Hz -> Butterworth LPF -> 128 ornek / %%50 overlap ->
    zaman + frekans ozellik vektorleri (README: 50 Hz, 128 ornek/pencere).
    """
    root = project_root.resolve()
    path_abs = _resolve(root, csv_path)
    node_paths_resolved = tuple(_resolve(root, p) for p in node_csv_paths if p is not None)
    if node_mode in ("multi_files", "four_files"):
        if len(node_paths_resolved) < 2:
            raise ValueError(
                "node_mode='multi_files' needs at least 2 paths in csv_har_node_files."
            )
        if path_abs is None:
            raise ValueError(
                "node_mode='multi_files' needs csv_path as label/reference (Time, ID, Activity)."
            )
        expected_nodes = 5 if node_mode == "multi_files" else 0
        fused_s = fuse_nodes_from_node_files(
            node_paths_resolved,
            time_column,
            sensor_columns,
            merge_tolerance_ms,
            time_format,
            time_utc,
            expected_node_count=expected_nodes,
        )
        ref = read_raw_csv(path_abs)
        for c in (time_column, group_column, target_column):
            if c not in ref.columns:
                raise KeyError(f"Reference csv_path must contain {c!r}. Got: {list(ref.columns)}")
        coerce_time_column(ref, time_column, time_format=time_format, utc=time_utc)
        meta = ref[[time_column, group_column, target_column]].drop_duplicates(
            subset=[time_column], keep="last"
        )
        left = fused_s.sort_values(time_column)
        tol = _merge_asof_tolerance(left[time_column], merge_tolerance_ms)
        fused = pd.merge_asof(
            left,
            meta.sort_values(time_column),
            on=time_column,
            direction="nearest",
            tolerance=tol,
        )
    else:
        if path_abs is None:
            raise ValueError("node_mode='single_csv' requires csv_path.")
        df0 = read_raw_csv(path_abs)
        for c in (node_column, time_column, group_column, target_column):
            if c not in df0.columns:
                raise KeyError(f"Missing {c!r} in CSV. Columns: {list(df0.columns)}")
        pieces: list[pd.DataFrame] = []
        for gid, dg in df0.groupby(group_column, sort=False):
            fused_s = fuse_nodes_from_single_csv(
                dg,
                node_column,
                node_values,
                time_column,
                sensor_columns,
                merge_tolerance_ms,
                time_format,
                time_utc,
            )
            meta = (
                dg[[time_column, target_column]]
                .drop_duplicates(subset=[time_column], keep="last")
                .sort_values(time_column)
            )
            left = fused_s.sort_values(time_column)
            tol = _merge_asof_tolerance(left[time_column], merge_tolerance_ms)
            fused_g = pd.merge_asof(
                left,
                meta,
                on=time_column,
                direction="nearest",
                tolerance=tol,
            )
            fused_g[group_column] = gid
            pieces.append(fused_g)
        fused = pd.concat(pieces, ignore_index=True)

    fused = fused.dropna(subset=[group_column, target_column])
    if fused.empty:
        raise ValueError("Fused frame empty after aligning labels.")

    feat_cols = [
        c
        for c in fused.columns
        if c not in (time_column, target_column, group_column) and c not in exclude_columns
    ]
    if not feat_cols:
        raise ValueError("No sensor feature columns after fusion.")

    X_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    g_rows: list[object] = []

    for gid, sub in fused.groupby(group_column, sort=False):
        sub = sub.sort_values(time_column)
        rs = resample_uniform_har(sub, time_column, feat_cols, target_column, fs_hz)
        if rs.shape[0] < WINDOW_SAMPLES_UCI:
            continue
        # 128 ornek / pencere kurali, Node'lar birlestirilmis genis (wide) matris uzerinde uygulanir.
        mat = rs[feat_cols].to_numpy(dtype=np.float64)
        if use_body_acc_from_triax and mat.shape[1] >= 3:
            body = body_acceleration_from_total(mat[:, :3], fs_hz)
            mat = np.hstack([body, mat[:, 3:]])
        for j in range(mat.shape[1]):
            mat[:, j] = butter_lowpass_filtfilt(mat[:, j], lpf_cutoff_hz, fs_hz)
        y_arr = rs[target_column].astype(str).to_numpy()
        wins = sliding_windows_2d(mat, WINDOW_SAMPLES_UCI, WINDOW_STRIDE_UCI)
        if wins.shape[0] == 0:
            continue
        for k in range(wins.shape[0]):
            end = (k * WINDOW_STRIDE_UCI) + WINDOW_SAMPLES_UCI - 1
            X_rows.append(extract_window_time_frequency_features(wins[k]))
            y_rows.append(str(y_arr[end]) if end < len(y_arr) else str(y_arr[-1]))
            g_rows.append(gid)

    if not X_rows:
        raise ValueError("No HAR feature windows. Check groups, fs_hz, and recording length.")

    X = np.stack(X_rows, axis=0)
    y_raw = np.array(y_rows, dtype=object)
    gids = np.array(g_rows, dtype=object)

    if split_by_group:
        unique_groups = np.unique(gids)
        if len(unique_groups) < 2:
            raise ValueError("Need at least 2 groups for split_by_group.")
        g_tr, g_te = train_test_split(unique_groups, test_size=test_size, random_state=random_state)
        set_tr, set_te = set(g_tr.tolist()), set(g_te.tolist())
        m_tr = np.array([g in set_tr for g in gids], dtype=bool)
        m_te = np.array([g in set_te for g in gids], dtype=bool)
    else:
        n = X.shape[0]
        idx = np.arange(n)
        i_tr, i_te = train_test_split(idx, test_size=test_size, random_state=random_state)
        m_tr, m_te = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        m_tr[i_tr], m_te[i_te] = True, True

    if split_by_group and m_tr.any() and m_te.any():
        u_tr = np.unique(y_raw[m_tr])
        u_te = np.unique(y_raw[m_te])
        if len(u_tr) == 1 and len(u_te) == 1 and u_tr[0] != u_te[0]:
            n = X.shape[0]
            idx = np.arange(n)
            try:
                i_tr, i_te = train_test_split(
                    idx,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=y_raw,
                )
            except ValueError:
                i_tr, i_te = train_test_split(
                    idx, test_size=test_size, random_state=random_state
                )
            m_tr[:] = False
            m_te[:] = False
            m_tr[i_tr] = True
            m_te[i_te] = True

    le = LabelEncoder()
    le.fit(y_raw)
    y_int = le.transform(y_raw)
    y_tr = y_int[m_tr]
    y_te = y_int[m_te]
    return X[m_tr], X[m_te], y_tr, y_te, le
