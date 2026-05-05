# HAR in Dance

Human Activity Recognition (HAR) egitimi icin hazirlanan bu proje, UCI HAR veya CSV tabanli verilerle model egitimi yapar ve tum deney ciktilarini duzenli bir sekilde kaydeder.

## Ozellikler

- Tek giris noktasi: `train.py`
- Veri akislari:
  - `dataset="uci_har"`
  - `dataset="csv"` + `csv_layout`:
    - `tabular`
    - `grouped_windows`
    - `har_processor`
- Model secenekleri:
  - `rf`
  - `xgboost`
  - `cnn1d`
  - `lstm`
- Otomatik metrik/rapor ciktilari:
  - OVR ROC + AUC (`roc_auc.png`)
  - Confusion Matrix heatmap (`confusion_matrix.png`)
  - Sinif isimleriyle classification report (`metrics.txt`)
- Logging:
  - Terminal + dosya log (`train.log`)
- Deney takibi:
  - Tum kosular `models/summary_results.csv` dosyasina satir olarak eklenir
- XGBoost hiperparametre tuning:
  - Opsiyonel `GridSearchCV` ile `max_depth` taramasi

## Proje Yapisi

```text
project/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── training/
│   ├── preprocessing/
│   └── utils/
├── training/
│   ├── config.py
│   ├── data_uci.py
│   ├── models.py
│   ├── pipeline.py
│   └── data_prep/
├── train.py
└── requirements.txt
```

> Not: Kodun aktif calisan bolumu su anda `training/` altindadir. `src/` dizini hedeflenen mimari icin ayrilmistir.

## Kurulum

```bash
pip install -r requirements.txt
```

## Egitim Calistirma

`train.py` icindeki `CFG = TrainConfig(...)` ayarlarini duzenleyip:

```bash
python train.py
```

## Cikti Yapisi

Her calismada `models/` altinda yeni bir klasor olusur:

```text
<dataset_name>_<model_type>_<YYYYMMDD_HHMMSS>/
├── train.log
├── <model>.joblib | <model>.keras
└── scores/
    ├── roc_auc.png
    ├── confusion_matrix.png
    └── metrics.txt
```

Ek olarak:

- `models/summary_results.csv`: tum kosularin ozet tablosu

## XGBoost Tuning Ayarlari

`TrainConfig` icinde:

- `xgb_tune_max_depth`: `True/False`
- `xgb_depth_candidates`: denenek derinlikler
- `xgb_cv_folds`: CV fold sayisi
- `xgb_scoring`: skor metriği (ornek: `accuracy`)

Ornek:

```python
xgb_tune_max_depth=True,
xgb_depth_candidates=(3, 4, 5, 6, 8, 10),
xgb_cv_folds=3,
xgb_scoring="accuracy",
```

## Dataset Notu

Yerel dataset ve model artifact dosyalari repoya dahil edilmez:

- `UCI HAR Dataset/`
- `data/*.csv`
- `models/`

Bu sayede repository temiz kalir, agir dosyalar lokal ortamda tutulur.
