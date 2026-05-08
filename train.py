"""
Eğitim giriş noktası — Dans Projesi (5 Node Fusion) için güncellendi.
Çalıştırma (proje kökünden): python train.py
"""

from pathlib import Path
from training import TrainConfig, run
from training.data_prep.deneme_builder import build_single_csv_from_deneme

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    RAW_DENEME = ROOT / "data" / "raw" / "deneme"
    PROCESSED_SOURCE = ROOT / "data" / "processed" / "deneme_single_csv.csv"
    processed_csv, n_rows = build_single_csv_from_deneme(RAW_DENEME, PROCESSED_SOURCE)
    print(f"[prep] built processed source: {processed_csv} | rows={n_rows}")

    CFG = TrainConfig(
        project_root=ROOT,
        # --- Kendi Veri Setin İçin Ayarlar ---
        dataset="csv",
        csv_layout="har_processor",
        csv_path=processed_csv,
        
        # har_processor beklenen etiket/zaman/grup sütunları
        csv_target_column="Activity",
        csv_group_column="Recording",
        csv_time_column="Time",
        
        # --- 5 Sensör (Node) Birleştirme Ayarları ---
        csv_har_node_mode="single_csv", # Tek CSV'de Node sütunu kullanılır
        csv_har_node_values=("1", "2", "3", "5"),
        csv_har_node_files=(),
        csv_har_sensor_columns=("Ax", "Ay", "Az", "Gx", "Gy", "Gz"),
        
        # --- Sinyal İşleme Parametreleri ---
        csv_fs_hz=50.0,              # Örnekleme hızı (Hz)
        csv_har_lpf_cutoff_hz=20.0,  # Butterworth filtre kesim frekansı
        csv_har_use_body_acc=False,  # Yerçekimi ivmesini ayırmayı şimdilik kapattık
        
        # --- Model ve Hiperparametreler (XGBoost) ---
        model="xgboost",
        random_state=42,
        n_estimators=300,
        
        # XGBoost max_depth otomatik iyileştirme (Optuna)
        xgb_tune_max_depth=False,
        xgb_tuning_method="optuna",
        xgb_depth_candidates=(3, 4, 5, 6, 8, 10),
        xgb_cv_folds=3,
        xgb_scoring="accuracy",
        xgb_optuna_trials=20,

        # Derin Öğrenme kullanmayacağımız için bu kısımlar XGBoost'u etkilemez
        epochs=30,
        batch_size=64,
        learning_rate=1e-3,
        
        # Kayıt Ayarları
        save_model=True,
        models_dir=ROOT / "models",
    )

    # Eğitimi başlat
    raise SystemExit(run(CFG))