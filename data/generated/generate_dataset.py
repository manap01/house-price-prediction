import pandas as pd
import numpy as np
import random
from datetime import datetime

# Set seed untuk menghasilkan data yang konsisten (reproducibility)
np.random.seed(42)
random.seed(42)

def generate_house_data(n_samples=1000):
    """
    Fungsi untuk membuat dataset sintetis harga rumah
    """
    data = []
    
    # Lokasi-lokasi yang tersedia
    locations = ['Jakarta Pusat', 'Jakarta Selatan', 'Jakarta Timur', 'Jakarta Barat', 
                 'Jakarta Utara', 'Tangerang', 'Bekasi', 'Depok', 'Bogor']
    
    for i in range(n_samples):
        # Fitur dasar
        luas_tanah = np.random.normal(120, 40)  # rata-rata 120 m², standar deviasi 40
        luas_tanah = max(60, luas_tanah)        # minimal 60 m²
        
        kamar_tidur = np.random.choice([2, 3, 4, 5, 6], p=[0.1, 0.4, 0.3, 0.15, 0.05])
        kamar_mandi = np.random.choice([1, 2, 3, 4], p=[0.2, 0.5, 0.25, 0.05])
        
        lantai = np.random.choice([1, 2, 3], p=[0.6, 0.35, 0.05])
        
        # Umur bangunan (tahun)
        umur_bangunan = np.random.exponential(scale=8)
        umur_bangunan = min(50, umur_bangunan)  # maksimal 50 tahun
        
        # Lokasi rumah
        lokasi = np.random.choice(locations)
        
        # Jarak ke pusat kota (km)
        if lokasi in ['Jakarta Pusat', 'Jakarta Selatan']:
            jarak_pusat_kota = np.random.normal(5, 2)
        elif lokasi in ['Jakarta Timur', 'Jakarta Barat', 'Jakarta Utara']:
            jarak_pusat_kota = np.random.normal(12, 4)
        else:
            jarak_pusat_kota = np.random.normal(25, 8)
        
        jarak_pusat_kota = max(1, jarak_pusat_kota)  # minimal 1 km
        
        # Fasilitas
        parkir = np.random.choice(['Ya', 'Tidak'], p=[0.8, 0.2])
        kolam_renang = np.random.choice(['Ya', 'Tidak'], p=[0.3, 0.7])
        
        # Perhitungan harga (dalam juta IDR)
        base_price = 500  # harga dasar 500 juta
        price = base_price
        price += luas_tanah * 3  # 3 juta per meter
        price += kamar_tidur * 50
        price += kamar_mandi * 30
        price += (lantai - 1) * 100
        price -= umur_bangunan * 5
        price -= jarak_pusat_kota * 8
        
        # Penyesuaian harga berdasarkan lokasi
        location_multiplier = {
            'Jakarta Pusat': 1.5,
            'Jakarta Selatan': 1.4,
            'Jakarta Timur': 1.0,
            'Jakarta Barat': 1.1,
            'Jakarta Utara': 1.0,
            'Tangerang': 0.8,
            'Bekasi': 0.7,
            'Depok': 0.75,
            'Bogor': 0.6
        }
        price *= location_multiplier[lokasi]
        
        # Tambahan harga jika memiliki fasilitas
        if parkir == 'Ya':
            price += 50
        if kolam_renang == 'Ya':
            price += 200
        
        # Tambahkan noise acak ±10%
        price *= np.random.normal(1, 0.1)
        price = max(200, price)  # harga minimum 200 juta
        
        data.append({
            'luas_tanah': round(luas_tanah, 1),
            'kamar_tidur': kamar_tidur,
            'kamar_mandi': kamar_mandi,
            'lantai': lantai,
            'umur_bangunan': round(umur_bangunan, 1),
            'lokasi': lokasi,
            'jarak_pusat_kota': round(jarak_pusat_kota, 1),
            'parkir': parkir,
            'kolam_renang': kolam_renang,
            'harga': round(price, 2)
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Buat dataset
    df = generate_house_data(1000)
    
    # Buat folder jika belum ada
    import os
    os.makedirs('data/generated', exist_ok=True)
    
    # Simpan ke CSV
    df.to_csv('data/generated/house_prices.csv', index=False)
    
    print(f"✅ Dataset berhasil dibuat dengan {len(df)} sampel data.")
    print("\nℹ️ Informasi Dataset:")
    print(df.info())
    print("\n📊 Statistik Deskriptif:")
    print(df.describe())
