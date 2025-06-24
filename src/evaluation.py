import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve, validation_curve
import scipy.stats as stats

class ModelEvaluator:
    """
    Kelas untuk melakukan evaluasi model machine learning regresi
    """

    def __init__(self):
        self.metrics = {}

    def calculate_metrics(self, y_true, y_pred, model_name="Model"):
        """
        Menghitung metrik evaluasi regresi untuk model
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # Median Absolute Error
        median_ae = np.median(np.abs(y_true - y_pred))

        metrics = {
            'Model': model_name,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'MAPE (%)': mape,
            'Median AE': median_ae
        }

        self.metrics[model_name] = metrics

        print(f"\nMetrik untuk {model_name}:")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Median AE: {median_ae:.2f}")

        return metrics

    def plot_predictions(self, y_true, y_pred, model_name="Model"):
        """
        Menampilkan plot perbandingan antara nilai aktual dan prediksi
        """
        plt.figure(figsize=(12, 5))

        # Plot scatter nilai aktual vs prediksi
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Harga Aktual')
        plt.ylabel('Harga Prediksi')
        plt.title(f'{model_name}: Prediksi vs Aktual')

        # Hitung dan tampilkan R²
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Plot residual
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Harga Prediksi')
        plt.ylabel('Residual')
        plt.title(f'{model_name}: Plot Residual')

        plt.tight_layout()
        plt.savefig(f'reports/figures/{model_name.lower().replace(" ", "_")}_predictions.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_residuals_distribution(self, y_true, y_pred, model_name="Model"):
        """
        Menampilkan distribusi residual model
        """
        residuals = y_true - y_pred

        plt.figure(figsize=(12, 4))

        # Histogram residual
        plt.subplot(1, 3, 1)
        plt.hist(residuals, bins=30, alpha=0.7, density=True)
        plt.xlabel('Residual')
        plt.ylabel('Kerapatan')
        plt.title(f'{model_name}: Distribusi Residual')

        # Q-Q plot
        plt.subplot(1, 3, 2)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title(f'{model_name}: Q-Q Plot')

        # Box plot residual
        plt.subplot(1, 3, 3)
        plt.boxplot(residuals)
        plt.ylabel('Residual')
        plt.title(f'{model_name}: Boxplot Residual')

        plt.tight_layout()
        plt.savefig(f'reports/figures/{model_name.lower().replace(" ", "_")}_residuals.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        # Uji normalitas dengan Shapiro-Wilk
        shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
        print(f"\nUji Normalitas Shapiro-Wilk:")
        print(f"Statistik: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}")

        if shapiro_p > 0.05:
            print("Residual terdistribusi normal (p > 0.05)")
        else:
            print("Residual kemungkinan tidak terdistribusi normal (p ≤ 0.05)")

    def plot_learning_curves(self, model, X, y, model_name="Model"):
        """
        Menampilkan kurva pembelajaran (learning curves)
        """
        print(f"Menghasilkan learning curve untuk {model_name}...")

        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='r2'
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Skor Pelatihan')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Skor Validasi')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

        plt.xlabel('Jumlah Data Pelatihan')
        plt.ylabel('Skor R²')
        plt.title(f'Kurva Pembelajaran - {model_name}')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'reports/figures/{model_name.lower().replace(" ", "_")}_learning_curves.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_validation_curves(self, model, X, y, param_name, param_range, model_name="Model"):
        """
        Menampilkan kurva validasi (validation curves) untuk tuning hyperparameter
        """
        print(f"Menghasilkan validation curve untuk {model_name} - {param_name}...")

        train_scores, val_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range,
            cv=5, scoring='r2', n_jobs=-1
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(param_range, train_mean, 'o-', color='blue', label='Skor Pelatihan')
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

        plt.plot(param_range, val_mean, 'o-', color='red', label='Skor Validasi')
        plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

        plt.xlabel(param_name)
        plt.ylabel('Skor R²')
        plt.title(f'Kurva Validasi - {model_name} ({param_name})')
        plt.legend()
        plt.grid(True)

        if isinstance(param_range[0], str):
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(f'reports/figures/{model_name.lower().replace(" ", "_")}_validation_curves.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def compare_models(self, results_dict):
        """
        Membandingkan performa dari beberapa model
        """
        if not results_dict:
            print("Tidak ada hasil yang bisa dibandingkan")
            return

        comparison_df = pd.DataFrame(results_dict).T
        comparison_df = comparison_df.sort_values('R²', ascending=False)

        print("\nPerbandingan Model:")
        print(comparison_df.round(4))

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].barh(comparison_df.index, comparison_df['R²'])
        axes[0, 0].set_xlabel('Skor R²')
        axes[0, 0].set_title('Perbandingan R²')

        axes[0, 1].barh(comparison_df.index, comparison_df['RMSE'])
        axes[0, 1].set_xlabel('RMSE')
        axes[0, 1].set_title('Perbandingan RMSE')

        axes[1, 0].barh(comparison_df.index, comparison_df['MAE'])
        axes[1, 0].set_xlabel('MAE')
        axes[1, 0].set_title('Perbandingan MAE')

        axes[1, 1].barh(comparison_df.index, comparison_df['MAPE (%)'])
        axes[1, 1].set_xlabel('MAPE (%)')
        axes[1, 1].set_title('Perbandingan MAPE')

        plt.tight_layout()
        plt.savefig('reports/figures/model_comparison_detailed.png', dpi=300, bbox_inches='tight')
        plt.show()

        return comparison_df

    def error_analysis(self, y_true, y_pred, X_test=None, feature_names=None):
        """
        Analisis mendalam terhadap error prediksi
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        errors = np.abs(y_true - y_pred)
        relative_errors = errors / y_true * 100

        print("\nAnalisis Error:")
        print(f"Mean Absolute Error: {np.mean(errors):.2f}")
        print(f"Median Absolute Error: {np.median(errors):.2f}")
        print(f"Max Absolute Error: {np.max(errors):.2f}")
        print(f"95th Percentile Error: {np.percentile(errors, 95):.2f}")

        print(f"\nAnalisis Error Relatif:")
        print(f"Mean Relative Error: {np.mean(relative_errors):.2f}%")
        print(f"Median Relative Error: {np.median(relative_errors):.2f}%")
        print(f"Sampel dengan error >20%: {np.sum(relative_errors > 20)} ({np.sum(relative_errors > 20)/len(relative_errors)*100:.1f}%)")

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.hist(errors, bins=30, alpha=0.7)
        plt.xlabel('Absolute Error')
        plt.ylabel('Frekuensi')
        plt.title('Distribusi Absolute Error')

        plt.subplot(1, 3, 2)
        plt.hist(relative_errors, bins=30, alpha=0.7)
        plt.xlabel('Relative Error (%)')
        plt.ylabel('Frekuensi')
        plt.title('Distribusi Relative Error')

        plt.subplot(1, 3, 3)
        plt.scatter(y_true, errors, alpha=0.6)
        plt.xlabel('Harga Sebenarnya')
        plt.ylabel('Absolute Error')
        plt.title('Error vs Harga Sebenarnya')

        plt.tight_layout()
        plt.savefig('reports/figures/error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        worst_indices = np.argsort(relative_errors)[-10:]
        print(f"\n10 Prediksi dengan Error Terbesar:")
        for i, idx in enumerate(worst_indices):
            print(f"{i+1}. Aktual: {y_true[idx]:.2f}, Prediksi: {y_pred[idx]:.2f}, Error: {relative_errors[idx]:.1f}%")

    def generate_evaluation_report(self, model_name, y_true, y_pred, model, X_test, feature_names):
        """
        Menyusun laporan evaluasi lengkap untuk sebuah model
        """
        print(f"\n{'='*50}")
        print(f"LAPORAN EVALUASI: {model_name}")
        print(f"{'='*50}")

        metrics = self.calculate_metrics(y_true, y_pred, model_name)
        self.plot_predictions(y_true, y_pred, model_name)
        self.plot_residuals_distribution(y_true, y_pred, model_name)
        self.error_analysis(y_true, y_pred, X_test, feature_names)

        return metrics
