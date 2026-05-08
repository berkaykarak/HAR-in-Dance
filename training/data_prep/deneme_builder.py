from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_FILENAME_RE = re.compile(
    r"^ID_(?P<node>\d+)_(?P<subject>[^_]+)_(?P<activity>[^_]+)_(?P<trial>\d+)_(?P<clock>\d+)\.csv$",
    re.IGNORECASE,
)


def build_single_csv_from_deneme(
    raw_root: Path,
    output_csv: Path,
    required_nodes: tuple[str, ...] = ("1", "2", "3", "5"),
) -> tuple[Path, int]:
    """
    Convert nested raw/deneme recordings into one har_processor-compatible CSV.

    Output columns include:
    - Time
    - Node
    - Recording (group id for windowing/splitting)
    - Activity (target)
    - Subject
    - Ax, Ay, Az, Gx, Gy, Gz (+ any passthrough columns from source files)
    """
    raw_root = raw_root.resolve()
    csv_files = sorted(raw_root.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {raw_root}")

    required_node_set = set(required_nodes)
    recording_nodes: dict[str, set[str]] = {}
    indexed: list[tuple[Path, re.Match[str]]] = []
    for csv_path in csv_files:
        match = _FILENAME_RE.match(csv_path.name)
        if not match:
            continue
        node = match.group("node")
        subject = match.group("subject").lower()
        activity = csv_path.parent.name.lower()
        trial = match.group("trial")
        clock = match.group("clock")
        recording = f"{subject}_{activity}_{trial}_{clock}"
        recording_nodes.setdefault(recording, set()).add(node)
        indexed.append((csv_path, match))

    complete_recordings = {
        rec for rec, nodes in recording_nodes.items() if required_node_set.issubset(nodes)
    }
    if not complete_recordings:
        raise ValueError(
            f"No complete recordings found with required nodes={sorted(required_node_set)} "
            f"under {raw_root}"
        )

    frames: list[pd.DataFrame] = []
    for csv_path, match in indexed:
        node = match.group("node")
        subject = match.group("subject").lower()
        activity = csv_path.parent.name.lower()
        trial = match.group("trial")
        clock = match.group("clock")
        recording = f"{subject}_{activity}_{trial}_{clock}"
        if recording not in complete_recordings:
            continue

        df = pd.read_csv(csv_path)
        if "Time" not in df.columns:
            raise KeyError(f"Missing 'Time' in {csv_path}")
        for col in ("Ax", "Ay", "Az", "Gx", "Gy", "Gz"):
            if col not in df.columns:
                raise KeyError(f"Missing {col!r} in {csv_path}")

        df = df.copy()
        df["Node"] = str(node)
        df["Subject"] = subject
        df["Activity"] = activity
        df["Recording"] = recording
        frames.append(df)

    if not frames:
        raise ValueError(
            "No deneme CSV matched expected filename format: "
            "ID_<node>_<subject>_<activity>_<trial>_<clock>.csv"
        )

    merged = pd.concat(frames, ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return output_csv, int(len(merged))
