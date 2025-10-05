"""
XGBoost training script for TOI classification
This module trains and evaluates XGBoost models on the preprocessed TOI data
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

class XGBoostTOIClassifier:
    def __init__(self, random_state=42):
        """
        Initialize the XGBoost classifier for TOI data
        
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
        """Train XGBoost model with optimized hyperparameters"""
        print("\n=== TRAINING OPTIMIZED XGBOOST MODEL ===")
        
        # Create XGBoost classifier with optimized hyperparameters
        self.baseline_model = xgb.XGBClassifier(
            colsample_bytree=0.8,
            learning_rate=0.1,
            max_depth=6,
            n_estimators=300,
            subsample=0.8,
            random_state=self.random_state,
            eval_metric='mlogloss'
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
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        print("\n=== MODEL EVALUATION ===")
        
        # Use the baseline model (which is now our optimized model)
        if self.baseline_model is None:
            raise ValueError("No model has been trained yet. Call train_baseline_model() first.")
        
        # Make predictions
        y_pred = self.baseline_model.predict(self.X_test)
        y_pred_proba = self.baseline_model.predict_proba(self.X_test)
        
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
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('/Users/lilianahotsko/Desktop/nasa-challenge/ml_tabular/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.baseline_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(20)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title('Top 20 Most Important Features')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.savefig('/Users/lilianahotsko/Desktop/nasa-challenge/ml_tabular/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy, y_pred, y_pred_proba
    
    def cross_validate(self, cv_folds=5):
        """Perform cross-validation on the model"""
        print("\n=== CROSS VALIDATION ===")
        
        cv_scores = cross_val_score(
            self.baseline_model, self.X_train, self.y_train, 
            cv=cv_folds, scoring='accuracy'
        )
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def save_model(self, filepath="/Users/lilianahotsko/Desktop/nasa-challenge/ml_tabular/xgboost_toi_model.pkl"):
        """Save the trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.baseline_model,
                'feature_columns': self.X_train.columns.tolist(),
                'hyperparameters': {
                    'colsample_bytree': 0.8,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'n_estimators': 300,
                    'subsample': 0.8
                },
                'feature_importance': self.feature_importance
            }, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath="/Users/lilianahotsko/Desktop/nasa-challenge/ml_tabular/xgboost_toi_model.pkl"):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            self.baseline_model = model_data['model']
            self.hyperparameters = model_data.get('hyperparameters')
            self.feature_importance = model_data.get('feature_importance')
        print(f"Model loaded from {filepath}")

def main():
    """Main training pipeline"""
    print("=== XGBoost TOI Classification Training ===\n")
    
    # Initialize classifier
    classifier = XGBoostTOIClassifier()
    
    # Load preprocessed data
    X_train, X_test, y_train, y_test = classifier.load_preprocessed_data()
    
    # Train baseline model
    baseline_model, baseline_accuracy = classifier.train_baseline_model()    
    # Evaluate model
    accuracy, y_pred, y_pred_proba = classifier.evaluate_model()
    
    # Cross-validation
    cv_scores = classifier.cross_validate()
    
    # Save model
    classifier.save_model()
    
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(f"Cross-validation Mean: {cv_scores.mean():.4f}")
    print(f"Model saved successfully!")
    
    return classifier

if __name__ == "__main__":
    classifier = main()
