"""
LightGBM training script for TOI classification
This module trains and evaluates LightGBM models on the preprocessed TOI data
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

class LightGBMTOIClassifier:
    def __init__(self, random_state=42):
        """
        Initialize the LightGBM classifier for TOI data
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.feature_importance = None
        
    def load_preprocessed_data(self, data_dir="/Users/lilianahotsko/Desktop/nasa-challenge/ml_tabular"):
        """Load preprocessed training and test data"""
        print("Loading preprocessed data...")
        
        self.X_train = pd.read_csv(f"{data_dir}/X_train.csv")
        self.X_test = pd.read_csv(f"{data_dir}/X_test.csv")
        self.y_train = pd.read_csv(f"{data_dir}/y_train.csv")['target'].values
        self.y_test = pd.read_csv(f"{data_dir}/y_test.csv")['target'].values
        
        print(f"Training data: {self.X_train.shape}")
        print(f"Test data: {self.X_test.shape}")
        print(f"Training target distribution: {np.bincount(self.y_train)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_baseline_model(self):
        """Train a baseline LightGBM model with default parameters"""
        print("\n=== TRAINING BASELINE LIGHTGBM MODEL ===")
        
        # Create baseline LightGBM classifier
        self.baseline_model = lgb.LGBMClassifier(
            random_state=self.random_state,
            verbose=-1,  # Suppress warnings
            force_col_wise=True  # Avoid data copy warning
        )
        
        # Train the model
        self.baseline_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred_baseline = self.baseline_model.predict(self.X_test)
        baseline_accuracy = accuracy_score(self.y_test, y_pred_baseline)
        
        print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
        print("\nBaseline Classification Report:")
        print(classification_report(self.y_test, y_pred_baseline))
        
        return self.baseline_model, baseline_accuracy
    
    def train_optimized_model(self):
        """Train LightGBM model with optimized hyperparameters"""
        print("\n=== TRAINING OPTIMIZED LIGHTGBM MODEL ===")
        
        # Create LightGBM classifier with optimized hyperparameters
        # Using similar parameters to XGBoost but adapted for LightGBM
        self.model = lgb.LGBMClassifier(
            colsample_bytree=0.8,
            learning_rate=0.1,
            max_depth=6,
            n_estimators=300,
            subsample=0.8,
            num_leaves=31,  # LightGBM specific parameter
            min_child_samples=20,  # LightGBM specific parameter
            random_state=self.random_state,
            verbose=-1,
            force_col_wise=True
        )
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"Optimized Model Accuracy: {accuracy:.4f}")
        print("\nOptimized Model Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        return self.model, accuracy
    
    def hyperparameter_tuning(self, cv_folds=3):
        """Perform hyperparameter tuning using GridSearchCV"""
        print("\n=== HYPERPARAMETER TUNING ===")
        
        # Define parameter grid for tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'num_leaves': [15, 31, 63]
        }
        
        # Create LightGBM classifier
        lgb_classifier = lgb.LGBMClassifier(
            random_state=self.random_state,
            verbose=-1,
            force_col_wise=True
        )
        
        # Perform grid search
        print(f"Performing grid search with {cv_folds}-fold cross-validation...")
        grid_search = GridSearchCV(
            estimator=lgb_classifier,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return self.model, self.best_params
    
    def evaluate_model(self, model_to_evaluate=None):
        """Evaluate the trained model"""
        print("\n=== MODEL EVALUATION ===")
        
        # Use specified model or the main model
        if model_to_evaluate is None:
            if self.model is not None:
                model_to_evaluate = self.model
            elif self.baseline_model is not None:
                model_to_evaluate = self.baseline_model
            else:
                raise ValueError("No model has been trained yet.")
        
        # Make predictions
        y_pred = model_to_evaluate.predict(self.X_test)
        y_pred_proba = model_to_evaluate.predict_proba(self.X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('LightGBM Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('/Users/lilianahotsko/Desktop/nasa-challenge/ml_tabular/lightgbm_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': model_to_evaluate.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(20)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title('LightGBM: Top 20 Most Important Features')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.savefig('/Users/lilianahotsko/Desktop/nasa-challenge/ml_tabular/lightgbm_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy, y_pred, y_pred_proba
    
    def cross_validate(self, model_to_validate=None, cv_folds=5):
        """Perform cross-validation on the model"""
        print("\n=== CROSS VALIDATION ===")
        
        # Use specified model or the main model
        if model_to_validate is None:
            if self.model is not None:
                model_to_validate = self.model
            elif self.baseline_model is not None:
                model_to_validate = self.baseline_model
            else:
                raise ValueError("No model has been trained yet.")
        
        cv_scores = cross_val_score(
            model_to_validate, self.X_train, self.y_train, 
            cv=cv_folds, scoring='accuracy'
        )
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def save_model(self, filepath="/Users/lilianahotsko/Desktop/nasa-challenge/ml_tabular/lightgbm_toi_model.pkl"):
        """Save the trained model"""
        model_to_save = self.model if self.model is not None else self.baseline_model
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': model_to_save,
                'feature_columns': self.X_train.columns.tolist(),
                'best_params': self.best_params,
                'feature_importance': self.feature_importance
            }, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath="/Users/lilianahotsko/Desktop/nasa-challenge/ml_tabular/lightgbm_toi_model.pkl"):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data['model']
            self.best_params = model_data.get('best_params')
            self.feature_importance = model_data.get('feature_importance')
        print(f"Model loaded from {filepath}")

def main():
    """Main training pipeline"""
    print("=== LightGBM TOI Classification Training ===\n")
    
    # Initialize classifier
    classifier = LightGBMTOIClassifier()
    
    # Load preprocessed data
    X_train, X_test, y_train, y_test = classifier.load_preprocessed_data()
    
    # Train baseline model
    baseline_model, baseline_accuracy = classifier.train_baseline_model()
    
    # Train optimized model
    optimized_model, optimized_accuracy = classifier.train_optimized_model()
    
    # Evaluate the optimized model
    accuracy, y_pred, y_pred_proba = classifier.evaluate_model()
    
    # Cross-validation
    cv_scores = classifier.cross_validate()
    
    # Optional: Perform hyperparameter tuning (comment out if you want to skip for faster training)
    # print("\nStarting hyperparameter tuning (this may take a while)...")
    # tuned_model, best_params = classifier.hyperparameter_tuning(cv_folds=3)
    # tuned_accuracy, _, _ = classifier.evaluate_model(tuned_model)
    # tuned_cv_scores = classifier.cross_validate(tuned_model)
    
    # Save model
    classifier.save_model()
    
    print(f"\n=== LIGHTGBM TRAINING COMPLETE ===")
    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
    print(f"Optimized Accuracy: {optimized_accuracy:.4f}")
    print(f"Cross-validation Mean: {cv_scores.mean():.4f}")
    print(f"Model saved successfully!")
    
    return classifier

if __name__ == "__main__":
    classifier = main()
