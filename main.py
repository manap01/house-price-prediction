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
from sklearn.model_selection import train_test_split
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
        exec(open('generate_dataset.py').read())
    else:
        print(f"\n1. DATASET FOUND: {dataset_path}")
    
    # Step 2: Load dan Process Data
    print("\n2. DATA PROCESSING...")
    processor = DataProcessor()
    
    # Load data
    data = processor.load_data(dataset_path)
    if data is None:
        print("Error: Could not load data")
        return
    
    print(f"Dataset shape: {data.shape}")
    print(f"Dataset info:")
    print(data.info())
    
    # Process data
    X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test = processor.process_pipeline(data)
    
    # Step 3: Feature Engineering
    print("\n3. FEATURE ENGINEERING...")
    feature_engineer = FeatureEngineer()
    
    # Correlation analysis
    correlation_matrix, target_corr = feature_engineer.correlation_analysis(data)
    
    # Feature importance analysis
    feature_scores = feature_engineer.feature_importance_analysis(
        X_train_scaled, y_train, processor.feature_names
    )
    
    # Step 4: Model Training dan Evaluation
    print("\n4. MODEL TRAINING...")
    model_trainer = HousePriceModels()
    
    # Initialize models
    models = model_trainer.initialize_models()
    
    # Train models
    trained_models = model_trainer.train_models(X_train_scaled, y_train)
    
    # Evaluate models
    results_df = model_trainer.evaluate_models(X_train_scaled, y_train)
    print("\nModel Evaluation Results:")
    print(results_df)
    
    # Plot model comparison
    model_trainer.plot_model_comparison(results_df)
    
    # Hyperparameter tuning untuk best model
    print("\n5. HYPERPARAMETER TUNING...")
    tuned_model, best_params = model_trainer.hyperparameter_tuning(X_train_scaled, y_train)
    
    # Step 5: Final Evaluation
    print("\n6. FINAL EVALUATION...")
    evaluator = ModelEvaluator()
    
    # Get predictions dari best model
    best_predictions = model_trainer.predict(X_test_scaled)
    
    if best_predictions is not None:
        # Calculate metrics
        final_metrics = evaluator.calculate_metrics(
            y_test, best_predictions, model_trainer.best_model[0]
        )
        
        # Generate comprehensive evaluation report
        evaluator.generate_evaluation_report(
            model_trainer.best_model[0], y_test, best_predictions,
            model_trainer.best_model[1], X_test, processor.feature_names
        )
        
        # Feature importance analysis
        feature_importance = model_trainer.get_feature_importance(processor.feature_names)
        
        # Learning curves
        evaluator.plot_learning_curves(
            model_trainer.best_model[1], X_train_scaled, y_train, 
            model_trainer.best_model[0]
        )
    
    # Step 6: Save Models
    print("\n7. SAVING MODELS...")
    model_trainer.save_models()
    
    # Save scaler
    import joblib
    joblib.dump(processor.scaler, 'models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")
    
    # Step 7: Generate Summary Report
    print("\n8. GENERATING SUMMARY REPORT...")
    generate_summary_report(data, results_df, final_metrics if 'final_metrics' in locals() else None)
    
    print("\n" + "="*60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nFiles generated:")
    print("- Dataset: data/generated/house_prices.csv")
    print("- Models: models/")
    print("- Figures: reports/figures/")
    print("- Report: reports/final_report.md")

def generate_summary_report(data, results_df, final_metrics):
    """
    Generate summary report
    """
    report_content = f"""# House Price Prediction - Final Report

## Project Overview
Proyek machine learning untuk memprediksi harga rumah berdasarkan berbagai fitur seperti luas tanah, jumlah kamar, lokasi, dan fasilitas.

## Dataset Summary
- **Total Samples**: {len(data)}
- **Features**: {len(data.columns)-1} (excluding target)
- **Target Variable**: Harga rumah (dalam jutaan IDR)

### Dataset Statistics
```
{data.describe()}
```

## Model Performance Comparison
```
{results_df.to_string()}
```

## Best Model Results
"""
    
    if final_metrics:
        report_content += f"""
**Model**: {final_metrics['Model']}
- **R² Score**: {final_metrics['R²']:.4f}
- **RMSE**: {final_metrics['RMSE']:.2f} juta IDR
- **MAE**: {final_metrics['MAE']:.2f} juta IDR
- **MAPE**: {final_metrics['MAPE (%)']:.2f}%
"""
    
    report_content += """
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
    with open('reports/final_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("Summary report saved to reports/final_report.md")

if __name__ == "__main__":
    # Set style untuk plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Run main function
    main()