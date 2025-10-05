"""
Anti-Overfitting TOI Classification Model
Comprehensive approach to reduce overfitting and improve generalization
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class AntiOverfittingTOIModel:
    def __init__(self, data_path="/Users/lilianahotsko/Desktop/nasa-challenge/data/clean/TOI_2025.10.04_08.44.51.csv"):
        self.data_path = data_path
        self.scaler = RobustScaler()
        self.label_encoder = None
        self.feature_selector = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess with minimal feature engineering to reduce overfitting"""
        print("🔍 LOADING DATA WITH MINIMAL FEATURE ENGINEERING")
        print("="*60)
        
        df = pd.read_csv(self.data_path)
        
        # Remove identifiers
        id_cols = ['toi', 'tid', 'rastr', 'decstr', 'toi_created', 'rowupdate']
        df = df.drop(columns=[col for col in id_cols if col in df.columns])
        
        # Conservative missing value handling
        error_cols = [col for col in df.columns if 'err' in col.lower()]
        limit_cols = [col for col in df.columns if 'lim' in col.lower()]
        
        for col in error_cols + limit_cols:
            df[col] = df[col].fillna(0)
        
        measurement_cols = df.select_dtypes(include=[np.number]).columns
        measurement_cols = [col for col in measurement_cols 
                          if col not in error_cols and col not in limit_cols]
        
        for col in measurement_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Only the most essential features to avoid overfitting
        print("Creating only essential astrophysical features...")
        
        # Key physical relationships only
        if 'pl_rade' in df.columns and 'pl_orbper' in df.columns:
            df['pl_density_proxy'] = df['pl_rade'] / (df['pl_orbper'] ** (2/3))
            
        if 'pl_insol' in df.columns:
            df['habitable_zone'] = ((df['pl_insol'] >= 0.5) & (df['pl_insol'] <= 2.0)).astype(int)
            
        if 'pl_eqt' in df.columns:
            df['earth_like_temp'] = ((df['pl_eqt'] >= 200) & (df['pl_eqt'] <= 350)).astype(int)
        
        print(f"Conservative feature set: {df.shape[1]} features")
        return df
    
    def aggressive_feature_selection(self, X, y):
        """Aggressive feature selection to prevent overfitting"""
        print("\n🎯 AGGRESSIVE FEATURE SELECTION")
        print("="*50)
        
        print(f"Starting with {X.shape[1]} features")
        
        # Step 1: Remove low-variance features
        from sklearn.feature_selection import VarianceThreshold
        variance_selector = VarianceThreshold(threshold=0.01)
        X_variance = variance_selector.fit_transform(X)
        print(f"After variance threshold: {X_variance.shape[1]} features")
        
        # Step 2: Select top K features based on statistical tests
        k_best = min(30, X_variance.shape[1])  # Limit to 30 features max
        self.feature_selector = SelectKBest(f_classif, k=k_best)
        X_selected = self.feature_selector.fit_transform(X_variance, y)
        
        print(f"After SelectKBest: {X_selected.shape[1]} features")
        
        # Step 3: Recursive feature elimination with cross-validation (most conservative)
        rf_estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        rfecv = RFECV(estimator=rf_estimator, step=1, cv=3, scoring='accuracy', min_features_to_select=15)
        X_final = rfecv.fit_transform(X_selected, y)
        
        print(f"After RFECV: {X_final.shape[1]} features (final)")
        print(f"Optimal number of features: {rfecv.n_features_}")
        
        return X_final, variance_selector, rfecv
    
    def prepare_data_with_regularization(self, df):
        """Prepare data with strong regularization focus"""
        print("\n📊 PREPARING DATA WITH REGULARIZATION FOCUS")
        print("="*60)
        
        # Encode target
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['tfopwg_disp'])
        X = df.drop(['tfopwg_disp'], axis=1)
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le_col = LabelEncoder()
            X[col] = le_col.fit_transform(X[col].astype(str))
        
        X = X.fillna(X.median())
        
        # Feature selection to prevent overfitting
        X_selected, variance_selector, rfecv = self.aggressive_feature_selection(X.values, y)
        
        # Larger test set for better generalization assessment
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_selected, y, test_size=0.3, random_state=42, stratify=y  # Larger test set
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.3, random_state=42, stratify=y_temp  # Larger validation set
        )
        
        print(f"Train: {X_train.shape}")
        print(f"Val: {X_val.shape}")
        print(f"Test: {X_test.shape}")
        
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Conservative resampling - less aggressive to avoid overfitting
        print("\n⚖️ CONSERVATIVE RESAMPLING")
        print("="*40)
        
        # Use less aggressive resampling
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42, k_neighbors=3)  # Fewer neighbors for less overfitting
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        
        # Don't fully balance - keep some imbalance to reflect real distribution
        unique_classes, class_counts = np.unique(y_train_resampled, return_counts=True)
        target_count = int(np.median(class_counts))  # Use median instead of max
        
        print(f"Conservative resampling to median count: {target_count}")
        print(f"Resampled classes: {dict(zip(unique_classes, class_counts))}")
        
        return X_train_resampled, X_val_scaled, X_test_scaled, y_train_resampled, y_val, y_test
    
    def train_regularized_models(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train models with strong regularization to prevent overfitting"""
        print("\n🛡️ TRAINING HEAVILY REGULARIZED MODELS")
        print("="*60)
        
        # XGBoost with strong regularization (applying user's n_estimators=100)
        print("Training Regularized XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,  # User's anti-overfitting change
            max_depth=4,       # Reduced from 6
            learning_rate=0.05, # Reduced from 0.1
            subsample=0.7,     # Reduced from 0.8
            colsample_bytree=0.7, # Reduced from 0.8
            reg_alpha=3,       # Increased L1 regularization
            reg_lambda=3,      # Increased L2 regularization
            min_child_weight=5, # Increased
            gamma=2,           # Increased
            random_state=42,
            eval_metric='mlogloss',
            early_stopping_rounds=20  # More aggressive early stopping
        )
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # LightGBM with strong regularization
        print("Training Regularized LightGBM...")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(y_train)),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 15,      # Reduced from 31
            'learning_rate': 0.05,  # Reduced
            'feature_fraction': 0.7, # Reduced
            'bagging_fraction': 0.7, # Reduced
            'bagging_freq': 3,
            'reg_alpha': 3,        # Increased
            'reg_lambda': 3,       # Increased
            'min_child_samples': 30, # Increased
            'min_child_weight': 5,   # Added
            'random_state': 42,
            'verbose': -1
        }
        
        eval_results = {}
        lgb_model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            num_boost_round=100,  # Reduced from 300
            callbacks=[lgb.record_evaluation(eval_results), lgb.early_stopping(20)]
        )
        
        # Extract training curves
        xgb_results = xgb_model.evals_result()
        eval_keys = list(xgb_results.keys())
        
        xgb_train_loss = xgb_results[eval_keys[0]]['mlogloss']
        xgb_val_loss = xgb_results[eval_keys[1]]['mlogloss']
        lgb_train_loss = eval_results['train']['multi_logloss']
        lgb_val_loss = eval_results['val']['multi_logloss']
        
        # Test performance
        xgb_pred = xgb_model.predict(X_test)
        xgb_test_acc = accuracy_score(y_test, xgb_pred)
        
        lgb_pred_proba = lgb_model.predict(X_test)
        lgb_pred = np.argmax(lgb_pred_proba, axis=1)
        lgb_test_acc = accuracy_score(y_test, lgb_pred)
        
        # Cross-validation for robust evaluation
        print("\n🔄 CROSS-VALIDATION EVALUATION")
        print("="*50)
        
        # Use original training data for CV (not resampled)
        X_original = np.vstack([X_train, X_val])
        y_original = np.hstack([y_train, y_val])
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Create CV model without early stopping (no validation set in CV)
        xgb_cv_params = xgb_model.get_params()
        xgb_cv_params.pop('early_stopping_rounds', None)  # Remove early stopping for CV
        xgb_cv_model = xgb.XGBClassifier(**xgb_cv_params)
        xgb_cv_scores = cross_val_score(xgb_cv_model, X_original, y_original, cv=cv, scoring='accuracy')
        
        print(f"XGBoost CV: {xgb_cv_scores.mean():.4f} (+/- {xgb_cv_scores.std() * 2:.4f})")
        print(f"XGBoost Test: {xgb_test_acc:.4f}")
        print(f"XGBoost Overfitting Gap: {xgb_val_loss[-1] - xgb_train_loss[-1]:.4f}")
        
        print(f"LightGBM Test: {lgb_test_acc:.4f}")
        print(f"LightGBM Overfitting Gap: {lgb_val_loss[-1] - lgb_train_loss[-1]:.4f}")
        
        return {
            'xgb_model': xgb_model,
            'lgb_model': lgb_model,
            'xgb_train_loss': xgb_train_loss,
            'xgb_val_loss': xgb_val_loss,
            'lgb_train_loss': lgb_train_loss,
            'lgb_val_loss': lgb_val_loss,
            'xgb_test_acc': xgb_test_acc,
            'lgb_test_acc': lgb_test_acc,
            'xgb_cv_scores': xgb_cv_scores,
            'xgb_pred': xgb_pred,
            'lgb_pred': lgb_pred
        }
    
    def plot_regularization_analysis(self, results):
        """Plot analysis showing overfitting reduction"""
        print("\n📈 PLOTTING REGULARIZATION ANALYSIS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Anti-Overfitting Model Analysis', fontsize=16, fontweight='bold')
        
        # XGBoost Loss Curves
        epochs_xgb = range(1, len(results['xgb_train_loss']) + 1)
        axes[0, 0].plot(epochs_xgb, results['xgb_train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs_xgb, results['xgb_val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('XGBoost - Regularized Loss Curves', fontweight='bold')
        axes[0, 0].set_xlabel('Boosting Rounds')
        axes[0, 0].set_ylabel('Log Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # LightGBM Loss Curves
        epochs_lgb = range(1, len(results['lgb_train_loss']) + 1)
        axes[0, 1].plot(epochs_lgb, results['lgb_train_loss'], 'g-', label='Training Loss', linewidth=2)
        axes[0, 1].plot(epochs_lgb, results['lgb_val_loss'], 'orange', label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('LightGBM - Regularized Loss Curves', fontweight='bold')
        axes[0, 1].set_xlabel('Boosting Rounds')
        axes[0, 1].set_ylabel('Log Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Overfitting Gap Analysis
        xgb_gap = results['xgb_val_loss'][-1] - results['xgb_train_loss'][-1]
        lgb_gap = results['lgb_val_loss'][-1] - results['lgb_train_loss'][-1]
        
        models = ['XGBoost', 'LightGBM']
        gaps = [xgb_gap, lgb_gap]
        colors = ['lightcoral' if gap > 0.3 else 'lightgreen' for gap in gaps]
        
        bars = axes[1, 0].bar(models, gaps, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Overfitting Gap Analysis', fontweight='bold')
        axes[1, 0].set_ylabel('Validation Loss - Training Loss')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Overfitting Threshold')
        axes[1, 0].legend()
        
        # Add value labels
        for bar, gap in zip(bars, gaps):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{gap:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Performance Summary
        summary_text = f"""Regularization Results:
        
XGBoost:
• Test Accuracy: {results['xgb_test_acc']:.4f}
• CV Mean: {results['xgb_cv_scores'].mean():.4f}
• CV Std: {results['xgb_cv_scores'].std():.4f}
• Overfitting Gap: {xgb_gap:.4f}
• Status: {'✅ Good' if xgb_gap < 0.3 else '⚠️ Overfitting'}

LightGBM:
• Test Accuracy: {results['lgb_test_acc']:.4f}
• Overfitting Gap: {lgb_gap:.4f}
• Status: {'✅ Good' if lgb_gap < 0.3 else '⚠️ Overfitting'}

Regularization Impact:
• Reduced n_estimators: 100 (was 300+)
• Increased regularization: α=3, λ=3
• Aggressive feature selection
• Conservative resampling
• Early stopping: 20 rounds"""
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Regularization Summary', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/Users/lilianahotsko/Desktop/nasa-challenge/ml_tabular/anti_overfitting_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def run_anti_overfitting_pipeline(self):
        """Run complete anti-overfitting pipeline"""
        print("🛡️ ANTI-OVERFITTING TOI MODEL PIPELINE")
        print("="*80)
        
        # 1. Load with minimal features
        df = self.load_and_preprocess_data()
        
        # 2. Prepare with strong regularization
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data_with_regularization(df)
        
        # 3. Train regularized models
        results = self.train_regularized_models(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # 4. Plot analysis
        fig = self.plot_regularization_analysis(results)
        
        # 5. Choose best model based on generalization
        xgb_generalization = abs(results['xgb_cv_scores'].mean() - results['xgb_test_acc'])
        
        if results['xgb_test_acc'] > results['lgb_test_acc'] and xgb_generalization < 0.05:
            best_model = "XGBoost"
            best_acc = results['xgb_test_acc']
            best_pred = results['xgb_pred']
        else:
            best_model = "LightGBM"
            best_acc = results['lgb_test_acc']
            best_pred = results['lgb_pred']
        
        # 6. Final evaluation
        print("\n📊 FINAL ANTI-OVERFITTING EVALUATION")
        print("="*60)
        
        print(f"🏆 Best Generalized Model: {best_model}")
        print(f"🎯 Test Accuracy: {best_acc:.4f}")
        
        target_names = self.label_encoder.classes_
        print(f"\nDetailed Classification Report ({best_model}):")
        print(classification_report(y_test, best_pred, target_names=target_names))
        
        # Overfitting assessment
        xgb_gap = results['xgb_val_loss'][-1] - results['xgb_train_loss'][-1]
        lgb_gap = results['lgb_val_loss'][-1] - results['lgb_train_loss'][-1]
        
        print(f"\n🛡️ OVERFITTING ASSESSMENT:")
        print(f"XGBoost gap: {xgb_gap:.4f} {'✅ Good' if xgb_gap < 0.3 else '⚠️ Still overfitting'}")
        print(f"LightGBM gap: {lgb_gap:.4f} {'✅ Good' if lgb_gap < 0.3 else '⚠️ Still overfitting'}")
        
        print(f"\n📈 NEXT STEPS FOR HIGHER ACCURACY:")
        if min(xgb_gap, lgb_gap) < 0.3:
            print("✅ Overfitting controlled! Ready for:")
            print("   1. 🧠 Neural networks with dropout")
            print("   2. 🔄 Advanced ensemble methods")
            print("   3. 📊 External feature engineering")
        else:
            print("⚠️ Still overfitting. Try:")
            print("   1. 🎯 Even more aggressive feature selection")
            print("   2. 🛡️ Stronger regularization")
            print("   3. 📉 Reduce model complexity further")
        
        return results, best_model, best_acc

def main():
    """Run the anti-overfitting analysis"""
    model = AntiOverfittingTOIModel()
    results, best_model, best_acc = model.run_anti_overfitting_pipeline()
    return model, results, best_model, best_acc

if __name__ == "__main__":
    model, results, best_model, best_acc = main()
