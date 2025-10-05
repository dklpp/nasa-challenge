"""
FINAL OPTIMIZED MODEL - Targeting 80%+ Realistic Accuracy
Combines best practices to achieve maximum realistic performance
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class FinalOptimizedTOIModel:
    def __init__(self, data_path="/Users/lilianahotsko/Desktop/nasa-challenge/data/clean/TOI_2025.10.04_08.44.51.csv"):
        self.data_path = data_path
        self.scaler = RobustScaler()  # More robust to outliers
        self.label_encoder = None
        self.feature_selector = None
        self.best_model = None
        
    def load_and_engineer_features(self):
        """Optimized feature engineering"""
        print("🔍 FINAL OPTIMIZED FEATURE ENGINEERING")
        print("="*60)
        
        df = pd.read_csv(self.data_path)
        
        # Remove identifiers
        id_cols = ['toi', 'tid', 'rastr', 'decstr', 'toi_created', 'rowupdate']
        df = df.drop(columns=[col for col in id_cols if col in df.columns])
        
        # Smart missing value handling
        error_cols = [col for col in df.columns if 'err' in col.lower()]
        limit_cols = [col for col in df.columns if 'lim' in col.lower()]
        
        for col in error_cols + limit_cols:
            df[col] = df[col].fillna(0)
        
        # Group-based imputation
        measurement_cols = df.select_dtypes(include=[np.number]).columns
        measurement_cols = [col for col in measurement_cols 
                          if col not in error_cols and col not in limit_cols]
        
        for col in measurement_cols:
            if df[col].isnull().sum() > 0:
                group_medians = df.groupby('tfopwg_disp')[col].median()
                df[col] = df[col].fillna(df['tfopwg_disp'].map(group_medians))
                df[col] = df[col].fillna(df[col].median())
        
        # Key astrophysical features (most predictive)
        print("Creating key astrophysical features...")
        
        # Planet physics
        if 'pl_rade' in df.columns and 'pl_orbper' in df.columns:
            df['pl_density_proxy'] = df['pl_rade'] / (df['pl_orbper'] ** (2/3))
            df['pl_rade_cubed'] = df['pl_rade'] ** 3  # Volume proxy
            
        if 'pl_insol' in df.columns:
            df['pl_insol_log'] = np.log10(df['pl_insol'] + 1e-8)
            df['habitable_zone'] = ((df['pl_insol'] >= 0.5) & (df['pl_insol'] <= 2.0)).astype(int)
            df['hot_jupiter'] = ((df['pl_insol'] > 100) & (df['pl_rade'] > 8)).astype(int)
            
        if 'pl_eqt' in df.columns:
            df['pl_eqt_log'] = np.log10(df['pl_eqt'] + 1e-8)
            df['earth_like_temp'] = ((df['pl_eqt'] >= 200) & (df['pl_eqt'] <= 350)).astype(int)
            
        # Stellar physics
        if 'st_teff' in df.columns and 'st_rad' in df.columns:
            df['st_luminosity'] = (df['st_rad'] ** 2) * ((df['st_teff'] / 5778) ** 4)
            df['st_luminosity_log'] = np.log10(df['st_luminosity'] + 1e-8)
            
        if 'st_tmag' in df.columns:
            df['st_observable'] = (df['st_tmag'] < 12).astype(int)  # Observable threshold
            
        # Transit characteristics
        if 'pl_trandurh' in df.columns and 'pl_orbper' in df.columns:
            df['transit_duty_cycle'] = df['pl_trandurh'] / (df['pl_orbper'] * 24)
            
        # Detection difficulty indicators
        if 'pl_rade' in df.columns and 'st_rad' in df.columns:
            df['transit_depth_proxy'] = (df['pl_rade'] / df['st_rad']) ** 2
            
        print(f"Enhanced features: {df.shape[1]}")
        return df
    
    def optimize_hyperparameters(self, X_train, y_train):
        """Optimize hyperparameters using Optuna"""
        print("\n🎛️ HYPERPARAMETER OPTIMIZATION")
        print("="*60)
        
        def objective(trial):
            # XGBoost parameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 800),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': 42,
                'eval_metric': 'mlogloss'
            }
            
            model = xgb.XGBClassifier(**params)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            return cv_scores.mean()
        
        print("Running Optuna optimization...")
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=30)  # Reduced for faster execution
        
        print(f"Best parameters: {study.best_params}")
        print(f"Best CV score: {study.best_value:.4f}")
        
        return study.best_params
    
    def create_final_model(self, best_params):
        """Create the final optimized model"""
        print("\n🚀 CREATING FINAL MODEL")
        print("="*60)
        
        # Optimized XGBoost
        xgb_model = xgb.XGBClassifier(**best_params)
        
        # Optimized LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1,
            reg_lambda=1,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
            force_col_wise=True
        )
        
        # Random Forest for diversity
        rf_model = RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # Create ensemble
        self.best_model = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('lgb', lgb_model),
                ('rf', rf_model)
            ],
            voting='soft'
        )
        
        return self.best_model
    
    def run_final_pipeline(self):
        """Run the complete final pipeline"""
        print("🏆 FINAL OPTIMIZED TOI MODEL PIPELINE")
        print("="*80)
        
        # 1. Feature engineering
        df = self.load_and_engineer_features()
        
        # 2. Prepare data
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['tfopwg_disp'])
        X = df.drop(['tfopwg_disp'], axis=1)
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le_col = LabelEncoder()
            X[col] = le_col.fit_transform(X[col].astype(str))
        
        X = X.fillna(X.median())
        
        print(f"Data prepared: {X.shape}")
        print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # 3. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y, test_size=0.25, random_state=42, stratify=y  # Larger test set
        )
        
        # 4. Feature selection
        print("\n🎯 FEATURE SELECTION")
        print("="*50)
        
        self.feature_selector = SelectKBest(f_classif, k=min(50, X_train.shape[1]))  # Top 50 features
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        print(f"Selected {X_train_selected.shape[1]} best features")
        
        # 5. Scaling
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # 6. Smart resampling
        print("\n⚖️ SMART RESAMPLING")
        print("="*50)
        
        # Use SMOTETomek for balanced and clean data
        sampler = SMOTETomek(random_state=42)
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_scaled, y_train)
        
        print(f"Resampled distribution: {dict(zip(*np.unique(y_train_resampled, return_counts=True)))}")
        
        # 7. Hyperparameter optimization
        best_params = self.optimize_hyperparameters(X_train_resampled, y_train_resampled)
        
        # 8. Train final model
        final_model = self.create_final_model(best_params)
        final_model.fit(X_train_resampled, y_train_resampled)
        
        # 9. Evaluation
        print("\n📊 FINAL EVALUATION")
        print("="*50)
        
        y_pred = final_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"🎯 Final Test Accuracy: {accuracy:.4f}")
        print(f"🎯 Final F1 Score: {f1:.4f}")
        
        # Cross-validation on original data for realistic estimate
        cv_scores = cross_val_score(final_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        print(f"🎯 Realistic CV Estimate: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Detailed report
        target_names = self.label_encoder.classes_
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Final Optimized Model - Confusion Matrix\nAccuracy: {accuracy:.4f}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('/Users/lilianahotsko/Desktop/nasa-challenge/ml_tabular/final_optimized_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Final summary
        print("\n" + "="*80)
        print("🎉 FINAL OPTIMIZED MODEL RESULTS")
        print("="*80)
        print(f"🎯 Test Accuracy: {accuracy:.4f}")
        print(f"🎯 F1 Score: {f1:.4f}")
        print(f"🎯 Realistic CV: {cv_scores.mean():.4f}")
        
        if accuracy >= 0.80:
            print("🏆 EXCELLENT: Achieved 80%+ accuracy!")
        elif accuracy >= 0.75:
            print("✅ VERY GOOD: Achieved 75%+ accuracy!")
        elif accuracy >= 0.70:
            print("✅ GOOD: Achieved 70%+ accuracy!")
        else:
            print("📈 BASELINE: Room for improvement")
        
        print(f"\n🔑 FINAL SETUP SUMMARY:")
        print(f"1. 📊 Enhanced feature engineering")
        print(f"2. 🎯 Feature selection (top 50)")
        print(f"3. 🛡️  Robust scaling")
        print(f"4. ⚖️  SMOTETomek resampling")
        print(f"5. 🎛️  Hyperparameter optimization")
        print(f"6. 🤝 Multi-algorithm ensemble")
        
        return accuracy, final_model, {
            'test_accuracy': accuracy,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }

def main():
    """Run the final optimized pipeline"""
    model = FinalOptimizedTOIModel()
    accuracy, trained_model, metrics = model.run_final_pipeline()
    return model, accuracy, trained_model, metrics

if __name__ == "__main__":
    model, accuracy, trained_model, metrics = main()
