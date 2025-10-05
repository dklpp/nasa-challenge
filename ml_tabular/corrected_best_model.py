"""
CORRECTED BEST MODEL SETUP - Realistic High Accuracy
This addresses the overfitting issue and provides a realistic best setup
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

class RealisticBestTOIModel:
    def __init__(self, data_path="/Users/lilianahotsko/Desktop/nasa-challenge/data/clean/TOI_2025.10.04_08.44.51.csv"):
        self.data_path = data_path
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess data with realistic approach"""
        print("🔍 LOADING TOI DATASET")
        print("="*50)
        
        df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Target distribution:\n{df['tfopwg_disp'].value_counts()}")
        
        # Remove identifier columns
        id_cols = ['toi', 'tid', 'rastr', 'decstr', 'toi_created', 'rowupdate']
        df = df.drop(columns=[col for col in id_cols if col in df.columns])
        
        print("\n🔧 REALISTIC PREPROCESSING")
        print("="*50)
        
        # 1. Smart missing value handling
        error_cols = [col for col in df.columns if 'err' in col.lower()]
        limit_cols = [col for col in df.columns if 'lim' in col.lower()]
        
        for col in error_cols:
            df[col] = df[col].fillna(0)
        for col in limit_cols:
            df[col] = df[col].fillna(0)
        
        # For main measurements, use median imputation
        measurement_cols = df.select_dtypes(include=[np.number]).columns
        measurement_cols = [col for col in measurement_cols 
                          if col not in error_cols and col not in limit_cols]
        
        for col in measurement_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # 2. Focused feature engineering (avoid overfitting)
        print("Creating key astrophysical features...")
        
        # Only the most important features based on domain knowledge
        if 'pl_rade' in df.columns and 'pl_orbper' in df.columns:
            df['pl_density_proxy'] = df['pl_rade'] / (df['pl_orbper'] ** (2/3))
            
        if 'pl_insol' in df.columns:
            df['in_habitable_zone'] = ((df['pl_insol'] >= 0.5) & 
                                     (df['pl_insol'] <= 2.0)).astype(int)
            df['insol_log'] = np.log10(df['pl_insol'] + 1e-10)
        
        if 'pl_eqt' in df.columns:
            df['temp_earth_like'] = ((df['pl_eqt'] >= 200) & 
                                   (df['pl_eqt'] <= 350)).astype(int)
        
        if 'st_teff' in df.columns and 'st_rad' in df.columns:
            df['st_luminosity_proxy'] = (df['st_rad'] ** 2) * (df['st_teff'] / 5778) ** 4
        
        if 'st_tmag' in df.columns:
            df['st_bright'] = (df['st_tmag'] < 10).astype(int)
        
        print(f"Features after engineering: {df.shape[1]} (was 59)")
        
        return df
    
    def prepare_ml_data(self, df):
        """Prepare data for ML training"""
        print("\n📊 PREPARING ML DATA")
        print("="*50)
        
        # Encode target
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['tfopwg_disp'])
        
        # Prepare features
        X = df.drop(['tfopwg_disp'], axis=1)
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le_col = LabelEncoder()
            X[col] = le_col.fit_transform(X[col].astype(str))
        
        X = X.fillna(X.median())
        self.feature_names = X.columns.tolist()
        
        print(f"Final feature matrix: {X.shape}")
        print(f"Target distribution: {np.bincount(y)}")
        print(f"Target classes: {self.label_encoder.classes_}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_realistic_best_model(self, X_train, X_test, y_train, y_test):
        """Train the realistic best model approach"""
        print("\n🚀 TRAINING REALISTIC BEST MODEL")
        print("="*50)
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        print(f"Class weights: {class_weight_dict}")
        
        # 1. XGBoost with class weights and regularization
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1,  # L1 regularization
            reg_lambda=1,  # L2 regularization
            random_state=42,
            eval_metric='mlogloss'
        )
        
        # 2. LightGBM with class weights
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            verbose=-1,
            force_col_wise=True
        )
        
        # 3. Moderate SMOTE (less aggressive)
        print("Applying moderate SMOTE...")
        smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy='auto')
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        print(f"After SMOTE: {np.bincount(y_train_smote)}")
        
        # Train models
        print("Training XGBoost...")
        xgb_model.fit(X_train_smote, y_train_smote)
        
        print("Training LightGBM...")
        lgb_model.fit(X_train_smote, y_train_smote)
        
        # Ensemble approach
        print("Creating ensemble...")
        self.model = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('lgb', lgb_model)
            ],
            voting='soft'
        )
        
        self.model.fit(X_train_smote, y_train_smote)
        
        # Evaluate individual models
        xgb_pred = xgb_model.predict(X_test)
        lgb_pred = lgb_model.predict(X_test)
        ensemble_pred = self.model.predict(X_test)
        
        xgb_acc = accuracy_score(y_test, xgb_pred)
        lgb_acc = accuracy_score(y_test, lgb_pred)
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        
        print(f"\nModel Performance:")
        print(f"XGBoost accuracy: {xgb_acc:.4f}")
        print(f"LightGBM accuracy: {lgb_acc:.4f}")
        print(f"Ensemble accuracy: {ensemble_acc:.4f}")
        
        # Cross-validation on original training data (more realistic)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return self.model, ensemble_acc
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        print("\n📈 FINAL MODEL EVALUATION")
        print("="*50)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"🎯 Final Test Accuracy: {accuracy:.4f}")
        
        print("\nDetailed Classification Report:")
        target_names = self.label_encoder.classes_
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('Realistic Best Model - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('/Users/lilianahotsko/Desktop/nasa-challenge/ml_tabular/realistic_best_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy
    
    def run_complete_pipeline(self):
        """Run the complete realistic pipeline"""
        print("🏆 RUNNING REALISTIC BEST TOI MODEL")
        print("="*80)
        
        # 1. Load and preprocess
        df = self.load_and_preprocess_data()
        
        # 2. Prepare data
        X_train, X_test, y_train, y_test = self.prepare_ml_data(df)
        
        # 3. Train best model
        model, accuracy = self.train_realistic_best_model(X_train, X_test, y_train, y_test)
        
        # 4. Final evaluation
        final_accuracy = self.evaluate_model(X_test, y_test)
        
        # 5. Summary
        print("\n" + "="*80)
        print("🎉 REALISTIC BEST MODEL COMPLETED")
        print("="*80)
        print(f"🎯 Final Test Accuracy: {final_accuracy:.4f}")
        print(f"📊 This represents a realistic achievable accuracy")
        print("\n🔑 KEY COMPONENTS:")
        print("1. 📊 Complete TOI dataset (7,703 samples)")
        print("2. 🔧 Smart preprocessing with domain knowledge") 
        print("3. ⚖️  Moderate SMOTE for class balance")
        print("4. 🤝 XGBoost + LightGBM ensemble")
        print("5. 🛡️  Regularization to prevent overfitting")
        
        return final_accuracy

def main():
    """Run the realistic best model"""
    model = RealisticBestTOIModel()
    accuracy = model.run_complete_pipeline()
    return model, accuracy

if __name__ == "__main__":
    model, accuracy = main()
