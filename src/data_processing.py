import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataProcessor:
    """
    Class untuk preprocessing data rumah
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, filepath):
        """
        Load data dari file CSV
        """
        try:
            data = pd.read_csv(filepath)
            print(f"Data berhasil dimuat. Shape: {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_data(self, data):
        """
        Membersihkan data dari missing values dan outliers
        """
        print("Membersihkan data...")
        
        # Check missing values
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            print("Missing values ditemukan:")
            print(missing_values[missing_values > 0])
            
            # Handle missing values
            # Untuk numerical: isi dengan median
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())
            
            # Untuk categorical: isi dengan mode
            categorical_cols = data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                data[col] = data[col].fillna(data[col].mode()[0])
        
        # Remove outliers menggunakan IQR method untuk harga
        Q1 = data['harga'].quantile(0.25)
        Q3 = data['harga'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        before_outlier_removal = len(data)
        data = data[(data['harga'] >= lower_bound) & (data['harga'] <= upper_bound)]
        after_outlier_removal = len(data)
        
        print(f"Outliers removed: {before_outlier_removal - after_outlier_removal} samples")
        
        return data
    
    def encode_categorical_features(self, data):
        """
        Encode categorical features
        """
        print("Encoding categorical features...")
        
        categorical_cols = ['lokasi', 'parkir', 'kolam_renang']
        
        for col in categorical_cols:
            if col in data.columns:
                le = LabelEncoder()
                data[col + '_encoded'] = le.fit_transform(data[col])
                self.label_encoders[col] = le
                print(f"Encoded {col}: {list(le.classes_)}")
        
        return data
    
    def create_features(self, data):
        """
        Create additional features
        """
        print("Creating additional features...")
        
        # Rasio kamar mandi terhadap kamar tidur
        data['rasio_kamar_mandi'] = data['kamar_mandi'] / data['kamar_tidur']
        
        # Luas per kamar
        data['luas_per_kamar'] = data['luas_tanah'] / data['kamar_tidur']
        
        # Kategori umur bangunan
        data['kategori_umur'] = pd.cut(data['umur_bangunan'], 
                                     bins=[0, 5, 15, 30, 100], 
                                     labels=['Baru', 'Muda', 'Sedang', 'Tua'])
        data['kategori_umur_encoded'] = LabelEncoder().fit_transform(data['kategori_umur'])
        
        # Kategori jarak
        data['kategori_jarak'] = pd.cut(data['jarak_pusat_kota'], 
                                      bins=[0, 10, 20, 100], 
                                      labels=['Dekat', 'Sedang', 'Jauh'])
        data['kategori_jarak_encoded'] = LabelEncoder().fit_transform(data['kategori_jarak'])
        
        # Score fasilitas
        data['score_fasilitas'] = 0
        data.loc[data['parkir'] == 'Ya', 'score_fasilitas'] += 1
        data.loc[data['kolam_renang'] == 'Ya', 'score_fasilitas'] += 1
        
        return data
    
    def prepare_features(self, data):
        """
        Prepare final feature set
        """
        feature_cols = [
            'luas_tanah', 'kamar_tidur', 'kamar_mandi', 'lantai', 
            'umur_bangunan', 'jarak_pusat_kota',
            'lokasi_encoded', 'parkir_encoded', 'kolam_renang_encoded',
            'rasio_kamar_mandi', 'luas_per_kamar', 
            'kategori_umur_encoded', 'kategori_jarak_encoded', 'score_fasilitas'
        ]
        
        # Filter features yang ada
        available_features = [col for col in feature_cols if col in data.columns]
        
        X = data[available_features]
        y = data['harga']
        
        self.feature_names = available_features
        print(f"Final features: {self.feature_names}")
        
        return X, y
    
    def scale_features(self, X_train, X_test):
        """
        Scale features menggunakan StandardScaler
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def process_pipeline(self, data):
        """
        Complete preprocessing pipeline
        """
        print("Starting data preprocessing pipeline...")
        
        # Clean data
        data = self.clean_data(data)
        
        # Encode categorical features
        data = self.encode_categorical_features(data)
        
        # Create features
        data = self.create_features(data)
        
        # Prepare features
        X, y = self.prepare_features(data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print("Data preprocessing completed!")
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test