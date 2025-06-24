# House Price Prediction - Final Report

## Project Overview
Proyek machine learning untuk memprediksi harga rumah berdasarkan berbagai fitur seperti luas tanah, jumlah kamar, lokasi, dan fasilitas.

## Dataset Summary
- **Total Samples**: 1000
- **Features**: 9 (excluding target)
- **Target Variable**: Harga rumah (dalam jutaan IDR)

### Dataset Statistics
        luas_tanah  kamar_tidur  kamar_mandi       lantai  umur_bangunan  jarak_pusat_kota        harga
count  1000.000000  1000.000000   1000.00000  1000.000000    1000.000000       1000.000000  1000.000000
mean    121.354600     3.609000      2.11200     1.423000       7.816400         16.184600  1086.179400
std      37.984456     1.012494      0.79504     0.578278       7.287168         10.047732   420.942674
min      60.000000     2.000000      1.00000     1.000000       0.000000          1.000000   324.710000
25%      91.400000     3.000000      2.00000     1.000000       2.500000          7.500000   758.330000
50%     119.350000     3.000000      2.00000     1.000000       5.500000         14.350000   991.365000
75%     149.025000     4.000000      3.00000     2.000000      11.200000         23.500000  1351.722500
max     234.600000     6.000000      4.00000     3.000000      50.000000         46.300000  2476.810000

## Model Performance Comparison
                  CV R² Mean CV R² Std                                                                                             CV Scores
Gradient Boosting   0.890528  0.014128    [0.8842311411728232, 0.9131076508675812, 0.899699681836142, 0.882296675847469, 0.8733056251166373]
LightGBM            0.883591  0.019806   [0.8779235266625209, 0.9076141754936404, 0.900649650794104, 0.8806186618383145, 0.8511504227541826]
XGBoost             0.874162  0.022712  [0.8735689870058518, 0.9100574251187972, 0.8824589817387952, 0.8640006996662082, 0.8407233301982885]
Random Forest       0.866116  0.017787  [0.8706306885807318, 0.8869187749759153, 0.8744373265825882, 0.8650067055633324, 0.8335848008520312]
Lasso Regression     0.72957  0.031417  [0.7057716009610646, 0.7617841138295367, 0.7544855658280236, 0.7457282109279795, 0.6800816426355067]
Ridge Regression    0.728935  0.032119   [0.7026757932611963, 0.7614749208402103, 0.7544991722647832, 0.746547118747809, 0.6794779528844803]
Decision Tree       0.725645  0.049843  [0.7119574817901579, 0.7524034170108996, 0.7786186920557749, 0.7497672902492818, 0.6354784652874597]
Linear Regression   0.724403  0.039076  [0.7022719570082543, 0.7613193293656111, 0.7537689790440883, 0.7467411834633086, 0.6579154663073598]

## Best Model Results

**Model**: Gradient Boosting_Tuned
- **R² Score**: 0.8799
- **RMSE**: 134.07 juta IDR
- **MAE**: 100.31 juta IDR
- **MAPE**: 9.78%


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

Generated on: 2025-06-25 05:59:11
