"""
Complete ML pipeline for TOI classification
This script runs the entire preprocessing and training pipeline
"""

import os
import sys
sys.path.append('/Users/lilianahotsko/Desktop/nasa-challenge/ml_tabular')

from data_preprocessing import main as preprocess_data
from xgboost_trainer import main as train_xgboost

def main():
    """Run the complete ML pipeline"""
    print("="*60)
    print("TOI CLASSIFICATION ML PIPELINE")
    print("="*60)
    
    # Step 1: Data Preprocessing
    print("\nSTEP 1: Data Preprocessing")
    print("-" * 30)
    try:
        preprocessor, X_train, X_test, y_train, y_test = preprocess_data()
        print("✓ Data preprocessing completed successfully!")
    except Exception as e:
        print(f"✗ Error in preprocessing: {e}")
        return
    
    # Step 2: XGBoost Training
    print("\nSTEP 2: XGBoost Training")
    print("-" * 30)
    try:
        classifier = train_xgboost()
        print("✓ XGBoost training completed successfully!")
    except Exception as e:
        print(f"✗ Error in training: {e}")
        return
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Files generated:")
    print("- Preprocessed data: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
    print("- Trained model: xgboost_toi_model.pkl")
    print("- Visualizations: confusion_matrix.png, feature_importance.png")

if __name__ == "__main__":
    main()
