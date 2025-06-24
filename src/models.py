import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt

class HousePriceModels:
    """
    Class untuk berbagai model prediksi harga rumah
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = -np.inf
        
    def initialize_models(self):
        """
        Initialize berbagai model untuk comparison
        """
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42),
            'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1)
        }
        
        print(f"Initialized {len(self.models)} models")
        return self.models
    
    def train_models(self, X_train, y_train):
        """
        Train semua model
        """
        print("Training all models...")
        trained_models = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            try:
                model.fit(X_train, y_train)
                trained_models[name] = model
                print(f"✓ {name} trained successfully")
            except Exception as e:
                print(f"✗ Error training {name}: {e}")
        
        self.models = trained_models
        return trained_models
    
    def evaluate_models(self, X_train, y_train, cv=5):
        """
        Evaluate semua model menggunakan cross-validation
        """
        print(f"Evaluating models with {cv}-fold cross-validation...")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(model, X_train, y_train, 
                                          cv=cv, scoring='r2')
                
                results[name] = {
                    'CV R² Mean': cv_scores.mean(),
                    'CV R² Std': cv_scores.std(),
                    'CV Scores': cv_scores
                }
                
                print(f"  R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                
                # Track best model
                if cv_scores.mean() > self.best_score:
                    self.best_score = cv_scores.mean()
                    self.best_model = (name, model)
                    
            except Exception as e:
                print(f"  Error evaluating {name}: {e}")
                results[name] = {'Error': str(e)}
        
        # Convert to DataFrame untuk display yang lebih baik
        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values('CV R² Mean', ascending=False)
        
        print(f"\nBest model: {self.best_model[0]} with R² = {self.best_score:.4f}")
        
        return results_df
    
    def hyperparameter_tuning(self, X_train, y_train):
        """
        Hyperparameter tuning untuk model terbaik
        """
        print("Performing hyperparameter tuning...")
        
        # Define parameter grids untuk setiap model
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'LightGBM': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        best_model_name = self.best_model[0]
        
        if best_model_name in param_grids:
            print(f"Tuning hyperparameters for {best_model_name}...")
            
            model = self.models[best_model_name]
            param_grid = param_grids[best_model_name]
            
            # GridSearchCV
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring='r2', 
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            # Update best model
            self.models[best_model_name + '_Tuned'] = grid_search.best_estimator_
            self.best_model = (best_model_name + '_Tuned', grid_search.best_estimator_)
            
            return grid_search.best_estimator_, grid_search.best_params_
        else:
            print(f"No hyperparameter tuning defined for {best_model_name}")
            return None, None
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance dari model terbaik
        """
        if self.best_model is None:
            print("No best model found. Train models first.")
            return None
        
        model_name, model = self.best_model
        
        # Check if model has feature_importance_ attribute
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\nFeature Importance ({model_name}):")
            print(feature_importance.head(10))
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            top_features = feature_importance.head(10)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title(f'Top 10 Feature Importance - {model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('reports/figures/model_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return feature_importance
        else:
            print(f"Model {model_name} doesn't have feature importance")
            return None
    
    def save_models(self, save_dir='models/'):
        """
        Save trained models
        """
        print(f"Saving models to {save_dir}...")
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.models.items():
            filename = name.lower().replace(' ', '_') + '.pkl'
            filepath = os.path.join(save_dir, filename)
            joblib.dump(model, filepath)
            print(f"Saved {name} to {filepath}")
        
        # Save best model separately
        if self.best_model:
            best_name, best_model = self.best_model
            best_filepath = os.path.join(save_dir, 'best_model.pkl')
            joblib.dump(best_model, best_filepath)
            print(f"Saved best model ({best_name}) to {best_filepath}")
    
    def load_model(self, filepath):
        """
        Load saved model
        """
        try:
            model = joblib.load(filepath)
            print(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def predict(self, X_test):
        """
        Make predictions using best model
        """
        if self.best_model is None:
            print("No trained model found.")
            return None
        
        model_name, model = self.best_model
        predictions = model.predict(X_test)
        
        print(f"Predictions made using {model_name}")
        return predictions
    
    def plot_model_comparison(self, results_df):
        """
        Plot perbandingan performa model
        """
        plt.figure(figsize=(12, 6))
        
        # Filter hasil yang valid (tidak ada error)
        valid_results = results_df[results_df['CV R² Mean'].notna()]
        
        plt.barh(range(len(valid_results)), valid_results['CV R² Mean'])
        plt.yticks(range(len(valid_results)), valid_results.index)
        plt.xlabel('R² Score')
        plt.title('Model Performance Comparison')
        plt.gca().invert_yaxis()
        
        # Add error bars
        plt.errorbar(valid_results['CV R² Mean'], range(len(valid_results)), 
                    xerr=valid_results['CV R² Std'], fmt='none', color='red', capsize=5)
        
        plt.tight_layout()
        plt.savefig('reports/figures/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def ensemble_prediction(self, X_test, top_n=3):
        """
        Ensemble prediction menggunakan top N models
        """
        print(f"Creating ensemble prediction with top {top_n} models...")
        
        # Get top models berdasarkan performance
        model_scores = []
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                # Assuming we have validation scores stored somewhere
                # For now, we'll use a simple average
                model_scores.append((name, model))
        
        # Take top N models
        top_models = model_scores[:top_n]
        
        predictions = []
        weights = []
        
        for name, model in top_models:
            pred = model.predict(X_test)
            predictions.append(pred)
            weights.append(1.0)  # Equal weights, bisa di-adjust
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        print(f"Ensemble prediction completed using {len(top_models)} models")
        return ensemble_pred