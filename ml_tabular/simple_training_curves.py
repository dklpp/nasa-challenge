"""
Simplified Training Curves for TOI Classification
Focus on essential loss and accuracy tracking with clean visualization
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, log_loss
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SimpleTOITrainingCurves:
    def __init__(self, data_path="/Users/lilianahotsko/Desktop/nasa-challenge/data/clean/TOI_2025.10.04_08.44.51.csv"):
        self.data_path = data_path
        self.scaler = RobustScaler()
        self.label_encoder = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess data efficiently"""
        print("🔍 LOADING DATA FOR TRAINING CURVES")
        print("="*50)
        
        df = pd.read_csv(self.data_path)
        
        # Remove identifiers
        id_cols = ['toi', 'tid', 'rastr', 'decstr', 'toi_created', 'rowupdate']
        df = df.drop(columns=[col for col in id_cols if col in df.columns])
        
        # Handle missing values
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
        
        # Key features
        if 'pl_rade' in df.columns and 'pl_orbper' in df.columns:
            df['pl_density_proxy'] = df['pl_rade'] / (df['pl_orbper'] ** (2/3))
            
        if 'pl_insol' in df.columns:
            df['habitable_zone'] = ((df['pl_insol'] >= 0.5) & (df['pl_insol'] <= 2.0)).astype(int)
            
        print(f"Data shape: {df.shape}")
        return df
    
    def prepare_data(self, df):
        """Prepare data with train/val/test splits"""
        print("\n📊 PREPARING DATA SPLITS")
        print("="*50)
        
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
        
        # Three-way split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X.values, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply resampling
        sampler = SMOTETomek(random_state=42)
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_scaled, y_train)
        
        print(f"Train: {X_train_resampled.shape}")
        print(f"Val: {X_val_scaled.shape}")
        print(f"Test: {X_test_scaled.shape}")
        print(f"Resampled classes: {dict(zip(*np.unique(y_train_resampled, return_counts=True)))}")
        
        return X_train_resampled, X_val_scaled, X_test_scaled, y_train_resampled, y_val, y_test
    
    def train_with_curves(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train models and capture training curves"""
        print("\n🚀 TRAINING WITH CURVE TRACKING")
        print("="*50)
        
        # XGBoost with evaluation tracking
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1,
            reg_lambda=1,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # Extract XGBoost curves
        xgb_results = xgb_model.evals_result()
        eval_keys = list(xgb_results.keys())
        
        xgb_train_loss = xgb_results[eval_keys[0]]['mlogloss']
        xgb_val_loss = xgb_results[eval_keys[1]]['mlogloss']
        
        # LightGBM with evaluation tracking
        print("Training LightGBM...")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(y_train)),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'reg_alpha': 1,
            'reg_lambda': 1,
            'random_state': 42,
            'verbose': -1
        }
        
        eval_results = {}
        lgb_model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            num_boost_round=300,
            callbacks=[lgb.record_evaluation(eval_results), lgb.early_stopping(50)]
        )
        
        lgb_train_loss = eval_results['train']['multi_logloss']
        lgb_val_loss = eval_results['val']['multi_logloss']
        
        print(f"XGBoost final: Train={xgb_train_loss[-1]:.4f}, Val={xgb_val_loss[-1]:.4f}")
        print(f"LightGBM final: Train={lgb_train_loss[-1]:.4f}, Val={lgb_val_loss[-1]:.4f}")
        
        # Test performance
        xgb_pred = xgb_model.predict(X_test)
        xgb_test_acc = accuracy_score(y_test, xgb_pred)
        
        lgb_pred_proba = lgb_model.predict(X_test)
        lgb_pred = np.argmax(lgb_pred_proba, axis=1)
        lgb_test_acc = accuracy_score(y_test, lgb_pred)
        
        print(f"Test Accuracy - XGBoost: {xgb_test_acc:.4f}, LightGBM: {lgb_test_acc:.4f}")
        
        return {
            'xgb_model': xgb_model,
            'lgb_model': lgb_model,
            'xgb_train_loss': xgb_train_loss,
            'xgb_val_loss': xgb_val_loss,
            'lgb_train_loss': lgb_train_loss,
            'lgb_val_loss': lgb_val_loss,
            'xgb_test_acc': xgb_test_acc,
            'lgb_test_acc': lgb_test_acc,
            'xgb_pred': xgb_pred,
            'lgb_pred': lgb_pred
        }
    
    def plot_training_curves(self, results):
        """Plot clean training curves"""
        print("\n📈 PLOTTING TRAINING CURVES")
        print("="*50)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('TOI Classification - Training Curves Analysis', fontsize=16, fontweight='bold')
        
        # XGBoost Loss Curves
        epochs_xgb = range(1, len(results['xgb_train_loss']) + 1)
        axes[0, 0].plot(epochs_xgb, results['xgb_train_loss'], 'b-', label='Training Loss', linewidth=2, alpha=0.8)
        axes[0, 0].plot(epochs_xgb, results['xgb_val_loss'], 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
        axes[0, 0].set_title('XGBoost - Loss Curves', fontweight='bold')
        axes[0, 0].set_xlabel('Boosting Rounds')
        axes[0, 0].set_ylabel('Log Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(bottom=0)
        
        # LightGBM Loss Curves
        epochs_lgb = range(1, len(results['lgb_train_loss']) + 1)
        axes[0, 1].plot(epochs_lgb, results['lgb_train_loss'], 'g-', label='Training Loss', linewidth=2, alpha=0.8)
        axes[0, 1].plot(epochs_lgb, results['lgb_val_loss'], 'orange', label='Validation Loss', linewidth=2, alpha=0.8)
        axes[0, 1].set_title('LightGBM - Loss Curves', fontweight='bold')
        axes[0, 1].set_xlabel('Boosting Rounds')
        axes[0, 1].set_ylabel('Log Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(bottom=0)
        
        # Model Comparison
        models = ['XGBoost', 'LightGBM']
        test_accuracies = [results['xgb_test_acc'], results['lgb_test_acc']]
        colors = ['skyblue', 'lightgreen']
        
        bars = axes[1, 0].bar(models, test_accuracies, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Test Accuracy Comparison', fontweight='bold')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, acc in zip(bars, test_accuracies):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Training Analysis
        xgb_overfitting = results['xgb_val_loss'][-1] - results['xgb_train_loss'][-1]
        lgb_overfitting = results['lgb_val_loss'][-1] - results['lgb_train_loss'][-1]
        
        analysis_text = f"""Training Analysis:
        
XGBoost:
• Final Train Loss: {results['xgb_train_loss'][-1]:.4f}
• Final Val Loss: {results['xgb_val_loss'][-1]:.4f}
• Overfitting Gap: {xgb_overfitting:.4f}
• Test Accuracy: {results['xgb_test_acc']:.4f}

LightGBM:
• Final Train Loss: {results['lgb_train_loss'][-1]:.4f}
• Final Val Loss: {results['lgb_val_loss'][-1]:.4f}
• Overfitting Gap: {lgb_overfitting:.4f}
• Test Accuracy: {results['lgb_test_acc']:.4f}

Recommendations:
{'• XGBoost shows better generalization' if xgb_overfitting < lgb_overfitting else '• LightGBM shows better generalization'}
{'• Consider more regularization' if max(xgb_overfitting, lgb_overfitting) > 0.2 else '• Good generalization achieved'}"""
        
        axes[1, 1].text(0.05, 0.95, analysis_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Training Analysis Summary', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/Users/lilianahotsko/Desktop/nasa-challenge/ml_tabular/toi_training_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def detailed_evaluation(self, results, y_test):
        """Detailed evaluation of the best model"""
        print("\n📊 DETAILED EVALUATION")
        print("="*50)
        
        # Choose best model
        if results['xgb_test_acc'] > results['lgb_test_acc']:
            best_pred = results['xgb_pred']
            best_name = "XGBoost"
            best_acc = results['xgb_test_acc']
        else:
            best_pred = results['lgb_pred']
            best_name = "LightGBM"
            best_acc = results['lgb_test_acc']
        
        print(f"🏆 Best Model: {best_name} with {best_acc:.4f} accuracy")
        
        # Classification report
        target_names = self.label_encoder.classes_
        print(f"\nDetailed Classification Report ({best_name}):")
        print(classification_report(y_test, best_pred, target_names=target_names))
        
        return best_name, best_acc
    
    def run_complete_analysis(self):
        """Run complete training curves analysis"""
        print("🏆 TOI TRAINING CURVES ANALYSIS")
        print("="*80)
        
        # 1. Load and preprocess
        df = self.load_and_preprocess_data()
        
        # 2. Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(df)
        
        # 3. Train with curves
        results = self.train_with_curves(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # 4. Plot curves
        fig = self.plot_training_curves(results)
        
        # 5. Detailed evaluation
        best_name, best_acc = self.detailed_evaluation(results, y_test)
        
        print("\n" + "="*80)
        print("🎉 TRAINING CURVES ANALYSIS COMPLETE")
        print("="*80)
        print(f"🎯 Best Model: {best_name}")
        print(f"🎯 Best Accuracy: {best_acc:.4f}")
        print(f"📊 Training curves saved to: toi_training_curves.png")
        
        return results, best_name, best_acc

def main():
    """Run the training curves analysis"""
    analyzer = SimpleTOITrainingCurves()
    results, best_name, best_acc = analyzer.run_complete_analysis()
    return analyzer, results, best_name, best_acc

if __name__ == "__main__":
    analyzer, results, best_name, best_acc = main()
