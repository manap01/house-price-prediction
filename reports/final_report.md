# House Price Prediction - Final Report

## Project Overview
Proyek machine learning untuk memprediksi harga rumah berdasarkan berbagai fitur seperti luas tanah, jumlah kamar, lokasi, dan fasilitas.

## Dataset Summary
- **Total Samples**: 1000
- **Features**: 9 (excluding target)
- **Target Variable**: Harga rumah (dalam jutaan IDR)

### Dataset Statistics
- **luas_tanah**:
  - Count: 1000.00
  - Mean: 121.35
  - Std: 37.98
  - Min: 60.00
  - 25%: 91.40
  - 50%: 119.35
  - 75%: 149.03
  - Max: 234.60

- **kamar_tidur**:
  - Count: 1000.00
  - Mean: 3.61
  - Std: 1.01
  - Min: 2.00
  - 25%: 3.00
  - 50%: 3.00
  - 75%: 4.00
  - Max: 6.00

- **kamar_mandi**:
  - Count: 1000.00
  - Mean: 2.11
  - Std: 0.80
  - Min: 1.00
  - 25%: 2.00
  - 50%: 2.00
  - 75%: 3.00
  - Max: 4.00

- **lantai**:
  - Count: 1000.00
  - Mean: 1.42
  - Std: 0.58
  - Min: 1.00
  - 25%: 1.00
  - 50%: 1.00
  - 75%: 2.00
  - Max: 3.00

- **umur_bangunan**:
  - Count: 1000.00
  - Mean: 7.82
  - Std: 7.29
  - Min: 0.00
  - 25%: 2.50
  - 50%: 5.50
  - 75%: 11.20
  - Max: 50.00

- **jarak_pusat_kota**:
  - Count: 1000.00
  - Mean: 16.18
  - Std: 10.05
  - Min: 1.00
  - 25%: 7.50
  - 50%: 14.35
  - 75%: 23.50
  - Max: 46.30

- **harga**:
  - Count: 1000.00
  - Mean: 1086.18
  - Std: 420.94
  - Min: 324.71
  - 25%: 758.33
  - 50%: 991.37
  - 75%: 1351.72
  - Max: 2476.81



## Model Performance Comparison
| Model | CV R² Mean | CV R² Std |
|-------|------------|-----------|
| Gradient Boosting | 0.8905 | 0.0141 |
| LightGBM | 0.8836 | 0.0198 |
| XGBoost | 0.8742 | 0.0227 |
| Random Forest | 0.8661 | 0.0178 |
| Lasso Regression | 0.7296 | 0.0314 |
| Ridge Regression | 0.7289 | 0.0321 |
| Decision Tree | 0.7256 | 0.0498 |
| Linear Regression | 0.7244 | 0.0391 |


## Best Model Results

**Model**: Gradient Boosting_Tuned

| Metric | Value |
|--------|-------|
| **R² Score** | 0.8799 |
| **RMSE** | 134.07 juta IDR |
| **MAE** | 100.31 juta IDR |
| **MAPE** | 9.78% |


## Key Findings
1. Model terbaik menunjukkan performa yang baik dengan R² > 0.8
2. Features yang paling berpengaruh: luas tanah, lokasi, dan jumlah kamar
3. Model dapat memprediksi harga dengan akurasi tinggi

## Recommendations
1. Gunakan model terbaik untuk prediksi harga rumah
2. Pertimbangkan feature engineering lebih lanjut
3. Kumpulkan data lebih banyak untuk meningkatkan akurasi

## Technical Details
- **Preprocessing**: StandardScaler, Label Encoding
- **Feature Engineering**: Interaction features, binning
- **Model Selection**: Cross-validation dengan 5 folds
- **Evaluation Metrics**: R², RMSE, MAE, MAPE

Generated on: 2025-06-25 06:06:47
