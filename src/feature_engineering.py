import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureEngineer:
    """
    Class untuk feature engineering dan feature selection
    """
    
    def __init__(self):
        self.selected_features = None
        self.feature_scores = None
        self.poly_features = None
        
    def correlation_analysis(self, data, target_col='harga'):
        """
        Analisis korelasi antar features
        """
        print("Melakukan analisis korelasi...")
        
        # Hitung korelasi
        corr_matrix = data.corr()
        
        # Korelasi dengan target
        target_corr = corr_matrix[target_col].abs().sort_values(ascending=False)
        print(f"\nKorelasi dengan {target_col}:")
        print(target_corr.head(10))
        
        # Visualisasi correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('reports/figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return corr_matrix, target_corr
    
    def feature_importance_analysis(self, X, y, feature_names):
        """
        Analisis feature importance menggunakan F-score
        """
        print("Melakukan analisis feature importance...")
        
        # Hitung F-score
        selector = SelectKBest(score_func=f_regression, k='all')
        selector.fit(X, y)
        
        # Get scores
        scores = selector.scores_
        
        # Create DataFrame untuk hasil
        feature_scores = pd.DataFrame({
            'feature': feature_names,
            'f_score': scores
        }).sort_values('f_score', ascending=False)
        
        self.feature_scores = feature_scores
        
        print("\nTop 10 Important Features:")
        print(feature_scores.head(10))
        
        # Visualisasi
        plt.figure(figsize=(10, 8))
        top_features = feature_scores.head(10)
        plt.barh(range(len(top_features)), top_features['f_score'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('F-Score')
        plt.title('Top 10 Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('reports/figures/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_scores
    
    def select_best_features(self, X, y, feature_names, k=10):
        """
        Select k best features
        """
        print(f"Selecting {k} best features...")
        
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]
        
        self.selected_features = selected_features
        
        print(f"Selected features: {selected_features}")
        
        return X_selected, selected_features
    
    def create_polynomial_features(self, X, degree=2):
        """
        Create polynomial features
        """
        print(f"Creating polynomial features (degree={degree})...")
        
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
        X_poly = poly.fit_transform(X)
        
        self.poly_features = poly
        
        print(f"Original features: {X.shape[1]}")
        print(f"Polynomial features: {X_poly.shape[1]}")
        
        return X_poly
    
    def create_interaction_features(self, data):
        """
        Create interaction features manually
        """
        print("Creating interaction features...")
        
        # Interaction antara luas tanah dan jumlah kamar
        data['luas_x_kamar_tidur'] = data['luas_tanah'] * data['kamar_tidur']
        data['luas_x_kamar_mandi'] = data['luas_tanah'] * data['kamar_mandi']
        
        # Interaction antara lokasi dan fasilitas
        if 'lokasi_encoded' in data.columns:
            data['lokasi_x_parkir'] = data['lokasi_encoded'] * data['parkir_encoded']
            data['lokasi_x_kolam'] = data['lokasi_encoded'] * data['kolam_renang_encoded']
        
        # Interaction antara umur dan jarak
        data['umur_x_jarak'] = data['umur_bangunan'] * data['jarak_pusat_kota']
        
        # Interaction antara lantai dan kamar
        data['lantai_x_kamar'] = data['lantai'] * data['kamar_tidur']
        
        return data
    
    def create_binned_features(self, data):
        """
        Create binned/categorical features dari numerical features
        """
        print("Creating binned features...")
        
        # Bin luas tanah
        data['luas_kategori'] = pd.cut(data['luas_tanah'], 
                                     bins=[0, 80, 120, 200, np.inf], 
                                     labels=['Kecil', 'Sedang', 'Besar', 'Sangat Besar'])
        
        # Bin harga untuk analisis (jangan digunakan sebagai feature)
        data['harga_kategori'] = pd.cut(data['harga'], 
                                      bins=[0, 500, 1000, 2000, np.inf], 
                                      labels=['Murah', 'Menengah', 'Mahal', 'Sangat Mahal'])
        
        return data
    
    def feature_engineering_pipeline(self, data):
        """
        Complete feature engineering pipeline
        """
        print("Starting feature engineering pipeline...")
        
        # Create interaction features
        data = self.create_interaction_features(data)
        
        # Create binned features
        data = self.create_binned_features(data)
        
        # Encode new categorical features
        from sklearn.preprocessing import LabelEncoder
        
        categorical_features = ['luas_kategori']
        for col in categorical_features:
            if col in data.columns:
                le = LabelEncoder()
                data[col + '_encoded'] = le.fit_transform(data[col])
        
        print("Feature engineering completed!")
        
        return data
    
    def plot_feature_distributions(self, data, features):
        """
        Plot distribusi dari features yang dipilih
        """
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, feature in enumerate(features):
            if i < len(axes):
                if feature in data.columns:
                    axes[i].hist(data[feature], bins=30, alpha=0.7)
                    axes[i].set_title(f'Distribution of {feature}')
                    axes[i].set_xlabel(feature)
                    axes[i].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('reports/figures/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()