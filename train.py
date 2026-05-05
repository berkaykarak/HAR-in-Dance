"""
Eğitim giriş noktası — aşağıdaki CFG bloğunu düzenlemen yeterli (path, model, hiperparametreler).
Çalıştırma (proje kökünden): python train.py
"""

from pathlib import Path

from training import TrainConfig, run

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent

    CFG = TrainConfig(
        project_root=ROOT,
        dataset="uci_har",
        uci_dataset_root=ROOT / "UCI HAR Dataset",
        # --- CSV tabular (satir = ornek, hedef + sayisal ozellikler) ---
        # dataset="csv",
        # csv_layout="tabular",
        # csv_path=ROOT / "data" / "mydata.csv",
        # csv_target_column="Activity",
        # csv_exclude_columns=("Time", "ID"),
        # test_size=0.2,
        # --- CSV gruplu zaman serisi: Time, ID, Ax..Gz, Height, Weight + hedef sutun ---
        # Ornek: Time,ID,Ax,...,Gz,Height,Weight + hedef sutun (Activity vb.) zorunlu
        # Hedef yoksa: etiket sutunu ekleyin veya baska dosyadan ID/Time ile merge edin
        # dataset="csv",
        # csv_layout="grouped_windows",
        # csv_path=ROOT / "data" / "sensor.csv",
        # csv_target_column="Activity",
        # csv_group_column="ID",
        # csv_time_column="Time",
        # Zaman metin/timestamp ise: csv_time_format="%Y-%m-%d %H:%M:%S" veya None (otomatik)
        # csv_time_utc=True  # UTC epoch / ISO icin gerekiyorsa
        # csv_feature_columns=None,
        # csv_exclude_columns=(),
        # csv_window_length=128,
        # csv_window_stride=64,
        # csv_split_by_group=True,
        # model="lstm",
        # --- UCI README tarzi: Butterworth LPF, 50 Hz, 128 ornek / 50 percent overlap, Node 1-4 fusion ---
        # dataset="csv",
        # csv_layout="har_processor",
        # csv_path=ROOT / "data" / "long_format.csv",
        # csv_time_column="Time",
        # csv_group_column="ID",
        # csv_target_column="Activity",
        # csv_har_node_mode="single_csv",
        # csv_node_column="Node",
        # csv_har_node_values=("1", "2", "3", "4"),
        # csv_har_sensor_columns=("Ax", "Ay", "Az", "Gx", "Gy", "Gz"),
        # csv_har_node_files=(ROOT/"data"/"n1.csv", ... ),
        # csv_har_node_mode="four_files",
        # csv_fs_hz=50.0,
        # csv_har_lpf_cutoff_hz=20.0,
        # csv_har_use_body_acc=False,
         model="xgboost",
        #model="rf",
        # model="xgboost"
        # model="cnn1d"
        # model="lstm"
        random_state=42,
        n_estimators=300,
        # XGBoost max_depth tuning (GridSearchCV)
        xgb_tune_max_depth=True,
        xgb_depth_candidates=(3, 4, 5, 6, 8, 10),
        xgb_cv_folds=3,
        xgb_scoring="accuracy",
        epochs=30,
        batch_size=64,
        learning_rate=1e-3,
        save_model=True,
        models_dir=ROOT / "models",
    )

    raise SystemExit(run(CFG))
