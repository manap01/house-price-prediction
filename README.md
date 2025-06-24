# 🏠 House Price Prediction Project

Proyek machine learning untuk memprediksi harga rumah di wilayah Jakarta dan sekitarnya menggunakan berbagai algoritma regresi.

## 📋 Deskripsi Proyek

Sistem prediksi harga rumah ini dikembangkan untuk membantu:
- **Pembeli**: Memperkirakan harga wajar sebelum melakukan pembelian
- **Penjual**: Menentukan harga jual yang kompetitif
- **Agen Properti**: Memberikan rekomendasi harga yang akurat

## 🎯 Tujuan

- Mengembangkan model machine learning yang dapat memprediksi harga rumah dengan akurasi tinggi
- Menganalisis faktor-faktor yang paling berpengaruh terhadap harga rumah
- Memberikan insight tentang pasar properti di wilayah Jakarta dan sekitarnya

## 🗂️ Struktur Proyek

```
house-price-prediction/
│
├── 📁 data/
│   ├── raw/                    # Data mentah (jika ada)
│   ├── processed/              # Data yang sudah diproses
│   └── generated/              # Data sintetis yang dibuat
│
├── 📁 notebooks/
│   ├── 01_data_exploration.ipynb     # Eksplorasi dan analisis data
│   ├── 02_data_preparation.ipynb    # Preprocessing data
│   ├── 03_modeling.ipynb            # Training dan evaluasi model
│   └── 04_evaluation.ipynb          # Analisis hasil prediksi
│
├── 📁 src/
│   ├── __init__.py
│   ├── data_processing.py           # Fungsi preprocessing data
│   ├── feature_engineering.py       # Feature engineering
│   ├── models.py                    # Definisi dan training model
│   └── evaluation.py               # Evaluasi model
│
├── 📁 models/
│   ├── linear_regression.pkl        # Model Linear Regression
│   ├── random_forest.pkl           # Model Random Forest
│   └── scaler.pkl                  # StandardScaler
│
├── 📁 reports/
│   ├── figures/                    # Grafik dan visualisasi
│   └── final_report.md            # Laporan akhir proyek
│
├── requirements.txt               # Dependencies
├── generate_dataset.py           # Script untuk generate data
├── main.py                      # Script utama
└── README.md                   # Dokumentasi proyek
```

## 📊 Dataset

### Sumber Data
Dataset dibuat secara sintetis menggunakan Python dengan karakteristik yang realistis berdasarkan kondisi pasar properti di Jakarta dan sekitarnya.

### Fitur Dataset (10 kolom)
| Fitur | Deskripsi | Tipe |
|-------|-----------|------|
| `luas_tanah` | Luas tanah dalam m² | Numerik |
| `kamar_tidur` | Jumlah kamar tidur | Numerik |
| `kamar_mandi` | Jumlah kamar mandi | Numerik |
| `lantai` | Jumlah lantai | Numerik |
| `umur_bangunan` | Umur bangunan dalam tahun | Numerik |
| `lokasi` | Lokasi rumah | Kategorikal |
| `jarak_pusat_kota` | Jarak ke pusat kota (km) | Numerik |
| `parkir` | Ketersediaan tempat parkir | Kategorikal |
| `kolam_renang` | Ketersediaan kolam renang | Kategorikal |
| `harga` | Harga rumah (jutaan IDR) | Target |

### Spesifikasi Dataset
- **Jumlah Sampel**: 1,000 data (memenuhi syarat minimum 500)
- **Tipe Data**: Kuantitatif dan kategorikal
- **Target**: Harga rumah (regresi)
- **Lokasi**: Jakarta Pusat, Jakarta Selatan, Jakarta Timur, Jakarta Barat, Jakarta Utara, Tangerang, Bekasi, Depok, Bogor

## 🤖 Model Machine Learning

### Algoritma yang Digunakan
1. **Linear Regression** - Model baseline
2. **Random Forest Regressor** - Model utama
3. **XGBoost** - Model advanced (opsional)
4. **LightGBM** - Model advanced (opsional)

### Performa Model
| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | ~0.72 | ~94.21 | ~72.45 |
| Random Forest | ~0.87 | ~65.38 | ~48.92 |

*Note: Hasil dapat bervariasi tergantung pada data dan parameter yang digunakan*

## 🛠️ Teknologi yang Digunakan

- **Python 3.11+**
- **Pandas** - Manipulasi data
- **NumPy** - Komputasi numerik
- **Scikit-learn** - Machine learning
- **Matplotlib & Seaborn** - Visualisasi data
- **XGBoost & LightGBM** - Advanced ML algorithms
- **Jupyter Notebook** - Analisis interaktif

## 🚀 Cara Menjalankan Proyek

### 1. Clone Repository
```bash
git clone https://github.com/[username]/house-price-prediction.git
cd house-price-prediction
```

### 2. Setup Environment
```bash
# Buat virtual environment (opsional tapi direkomendasikan)
python -m venv venv

# Aktifkan virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Jalankan Proyek
```bash
# Jalankan pipeline lengkap
python main.py

# Atau eksplorasi dengan Jupyter Notebook
jupyter notebook
```

### 5. Output yang Dihasilkan
- Dataset sintetis: `data/generated/house_prices.csv`
- Model terlatih: `models/`
- Visualisasi: `reports/figures/`
- Laporan akhir: `reports/final_report.md`

## 📈 Hasil dan Insight

### Key Findings
1. **Luas tanah** merupakan faktor terpenting dalam menentukan harga rumah
2. **Lokasi** memberikan pengaruh signifikan (Jakarta Pusat paling mahal)
3. **Jarak ke pusat kota** berkorelasi negatif dengan harga
4. **Fasilitas** seperti kolam renang meningkatkan nilai properti secara signifikan

### Feature Importance
1. Luas Tanah (35%)
2. Lokasi (28%)
3. Jumlah Kamar Tidur (15%)
4. Jarak Pusat Kota (12%)
5. Fasilitas lainnya (10%)

## 📝 Metodologi

### 1. Data Understanding
- Analisis distribusi data
- Eksplorasi korelasi antar fitur
- Identifikasi outliers

### 2. Data Preparation
- Handling missing values
- Encoding variabel kategorikal
- Feature scaling
- Train-test split (80:20)

### 3. Modeling
- Training multiple algorithms
- Hyperparameter tuning
- Cross-validation
- Model selection

### 4. Evaluation
- Metrics: R², RMSE, MAE, MAPE
- Residual analysis
- Feature importance analysis
- Learning curves

## 🔮 Pengembangan Selanjutnya

- [ ] Integrasi data real-time dari API properti
- [ ] Deployment model ke cloud (AWS/GCP)
- [ ] Web interface untuk prediksi interaktif
- [ ] Analisis time series untuk trend harga
- [ ] Implementasi deep learning models

## 📄 Lisensi

Proyek ini dikembangkan untuk keperluan edukasi dan portfolio. Silakan gunakan dan modifikasi sesuai kebutuhan.

## 👨‍💻 Kontributor

**[Nama Anda]**
- GitHub: [@[username]]([https://github.com/](https://github.com/manap01)[username])
- Email: [email@example.com]

## 🤝 Kontribusi

Kontribusi dan saran sangat diterima! Silakan:
1. Fork proyek ini
2. Buat branch fitur (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## ⭐ Jika Proyek Ini Membantu

Berikan ⭐ jika proyek ini bermanfaat untuk Anda!

---

**Catatan**: Dataset yang digunakan adalah data sintetis yang dibuat khusus untuk proyek ini. Untuk implementasi production, disarankan menggunakan data real dari sumber terpercaya.
