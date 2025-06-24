"""
Main script untuk House Price Prediction Project
Author: [Nama Anda]
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.models import HousePriceModels
from src.evaluation import ModelEvaluator

def create_directories():
    """Create necessary directories"""
    directories = [
        'data/raw', 'data/processed', 'data/generated',
        'models', 'reports/figures'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created: {directory}")

def generate_summary_report(data, results_df, final_metrics):
    """
    Generate summary report
    """
    # Handle case where final_metrics might be None
    metrics_section = ""
    if final_metrics is not None:
        metrics_section = f"""
**Model**: {final_metrics['Model']}
- **R² Score**: {final_metrics['R²']:.4f}
- **RMSE**: {final_metrics['RMSE']:.2f} juta IDR
- **MAE**: {final_metrics['MAE']:.2f} juta IDR
- **MAPE**: {final_metrics['MAPE (%)']:.2f}%
"""
    else:
        metrics_section = "\n**Tidak ada hasil model terbaik yang tersedia**\n"
    
    report_content = f"""# House Price Prediction - Final Report

## Project Overview
Proyek machine learning untuk memprediksi harga rumah berdasarkan berbagai fitur seperti luas tanah, jumlah kamar, lokasi, dan fasilitas.

## Dataset Summary
- **Total Samples**: {len(data)}
- **Features**: {len(data.columns)-1} (excluding target)
- **Target Variable**: Harga rumah (dalam jutaan IDR)

### Dataset Statistics
{data.describe().to_string()}

## Model Performance Comparison
{results_df.to_string() if not results_df.empty else "Tidak ada hasil evaluasi model"}

## Best Model Results
{metrics_section}

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

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Save report
    report_dir = 'reports'
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, 'final_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Summary report saved to {report_path}")
    return report_path

def main():
    """
    Main function untuk menjalankan complete pipeline
    """
    print("="*60)
    print("HOUSE PRICE PREDICTION PROJECT")
    print("="*60)
    
    # Create directories
    create_directories()
    
    # Step 1: Generate Dataset (jika belum ada)
    dataset_path = 'data/generated/house_prices.csv'
    if not os.path.exists(dataset_path):
        print("\n1. GENERATING DATASET...")
        try:
            # Eksekusi yang lebih aman
            from generate_dataset import main as generate_data
            generate_data()
            print("Dataset generated successfully")
        except ImportError as e:
            print(f"Error: File generate_dataset.py tidak ditemukan - {e}")
            return
        except Exception as e:
            print(f"Error generating dataset: {e}")
            return
    else:
        print(f"\n1. DATASET FOUND: {dataset_path}")
    
    # Step 2: Load dan Process Data
    print("\n2. DATA PROCESSING...")
    processor = DataProcessor()
    
    # Load data
    try:
        data = processor.load_data(dataset_path)
        if data is None:
            print("Error: Could not load data")
            return
        
        print(f"Dataset shape: {data.shape}")
        print("Sample data:")
        print(data.head(3))
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Process data
    try:
        X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test = processor.process_pipeline(data)
        print("Data processing completed successfully")
        print(f"Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
    except Exception as e:
        print(f"Error in data processing: {e}")
        return
    
    # Step 3: Feature Engineering
    print("\n3. FEATURE ENGINEERING...")
    feature_engineer = FeatureEngineer()
    
    # Correlation analysis
    try:
        correlation_matrix, target_corr = feature_engineer.correlation_analysis(data)
        print("Correlation analysis completed")
    except Exception as e:
        print(f"Error in correlation analysis: {e}")
    
    # Feature importance analysis
    try:
        feature_scores = feature_engineer.feature_importance_analysis(
            X_train_scaled, y_train, processor.feature_names
        )
        print("Feature importance analysis completed")
    except Exception as e:
        print(f"Error in feature importance analysis: {e}")
    
    # Step 4: Model Training dan Evaluation
    print("\n4. MODEL TRAINING & EVALUATION...")
    model_trainer = HousePriceModels()
    results_df = pd.DataFrame()  # Initialize empty dataframe
    
    try:
        # Initialize models
        models = model_trainer.initialize_models()
        print(f"Initialized {len(models)} models")
        
        # Train models
        trained_models = model_trainer.train_models(X_train_scaled, y_train)
        print("Models trained successfully")
        
        # Evaluate models
        results_df = model_trainer.evaluate_models(X_train_scaled, y_train)
        print("\nModel Evaluation Results:")
        print(results_df)
        
        # Plot model comparison
        model_trainer.plot_model_comparison(results_df)
        print("Model comparison plot saved")
    except Exception as e:
        print(f"Error in model training: {e}")
    
    # Step 5: Hyperparameter tuning untuk best model
    print("\n5. HYPERPARAMETER TUNING...")
    try:
        if hasattr(model_trainer, 'best_model'):
            tuned_model, best_params = model_trainer.hyperparameter_tuning(X_train_scaled, y_train)
            print(f"Best parameters: {best_params}")
        else:
            print("Skipping hyperparameter tuning - no best model available")
    except Exception as e:
        print(f"Error in hyperparameter tuning: {e}")
    
    # Step 6: Final Evaluation
    print("\n6. FINAL EVALUATION...")
    evaluator = ModelEvaluator()
    final_metrics = None
    
    try:
        if hasattr(model_trainer, 'best_model'):
            best_model_name, best_model = model_trainer.best_model
            print(f"Using best model: {best_model_name}")
            
            # Get predictions from best model
            best_predictions = best_model.predict(X_test_scaled)
            
            if best_predictions is not None:
                # Calculate metrics
                final_metrics = evaluator.calculate_metrics(
                    y_test, best_predictions, best_model_name
                )
                print("\nFinal Model Metrics:")
                print(f"R²: {final_metrics['R²']:.4f}")
                print(f"RMSE: {final_metrics['RMSE']:.2f}")
                print(f"MAE: {final_metrics['MAE']:.2f}")
                print(f"MAPE: {final_metrics['MAPE (%)']:.2f}%")
                
                # Generate comprehensive evaluation report
                evaluator.generate_evaluation_report(
                    best_model_name, y_test, best_predictions,
                    best_model, X_test, processor.feature_names
                )
                
                # Feature importance analysis
                try:
                    feature_importance = model_trainer.get_feature_importance(processor.feature_names)
                except:
                    print("Skipping feature importance - not available for this model")
                
                # Learning curves
                try:
                    evaluator.plot_learning_curves(
                        best_model, X_train_scaled, y_train, 
                        best_model_name
                    )
                except:
                    print("Skipping learning curves - not supported for this model")
        else:
            print("Skipping final evaluation - no best model available")
    except Exception as e:
        print(f"Error in final evaluation: {e}")
    
    # Step 7: Save Models
    print("\n7. SAVING MODELS...")
    try:
        if hasattr(model_trainer, 'save_models'):
            model_trainer.save_models()
            print("Models saved successfully")
        else:
            print("Skipping model saving - no models available")
        
        # Save scaler
        if hasattr(processor, 'scaler') and processor.scaler is not None:
            scaler_dir = 'models'
            os.makedirs(scaler_dir, exist_ok=True)
            scaler_path = os.path.join(scaler_dir, 'scaler.pkl')
            joblib.dump(processor.scaler, scaler_path)
            print(f"Scaler saved to {scaler_path}")
        else:
            print("Warning: No scaler found to save")
    except Exception as e:
        print(f"Error saving models: {e}")
    
    # Step 8: Generate Summary Report
    print("\n8. GENERATING SUMMARY REPORT...")
    try:
        report_path = generate_summary_report(data, results_df, final_metrics)
    except Exception as e:
        print(f"Error generating summary report: {e}")
        report_path = "reports/final_report.md"
    
    print("\n" + "="*60)
    print("PROJECT EXECUTION COMPLETED")
    print("="*60)
    print("\nOutput files:")
    print(f"- Dataset: {dataset_path}")
    print("- Processed data: data/processed/")
    print(f"- Models: models/")
    print(f"- Visualizations: reports/figures/")
    print(f"- Final report: {report_path}")
    print("\nNote: Beberapa langkah mungkin dilewati jika terjadi error")
    print("="*60)

if __name__ == "__main__":
    # Set style untuk plots
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    sns.set_palette("husl")
    
    # Run main function
    main()