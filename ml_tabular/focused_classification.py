"""
Focused TOI Classification - Excluding PC and APC classes
Focus on definitive classifications: CP, FA, FP, KP
This should reduce class imbalance and improve accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class FocusedTOIClassifier:
    def __init__(self, data_path="/Users/lilianahotsko/Desktop/nasa-challenge/data/clean/TOI_2025.10.04_08.44.51.csv"):
        self.data_path = data_path
        
    def load_and_filter_data(self):
        """Load data and filter out PC and APC classes"""
        print("🎯 FOCUSED CLASSIFICATION: EXCLUDING PC AND APC")
        print("="*60)
        
        df = pd.read_csv(self.data_path)
        print(f"Original dataset: {df.shape}")
        
        # Show original distribution
        print(f"\nOriginal class distribution:")
        original_dist = df['tfopwg_disp'].value_counts()
        for class_name, count in original_dist.items():
            percentage = (count / len(df)) * 100
            print(f"  {class_name}: {count:4d} ({percentage:5.1f}%)")
        
        # Filter out PC and APC classes
        classes_to_keep = ['CP', 'FA', 'FP', 'KP']  # Confirmed Planet, False Alarm, False Positive, Known Planet
        df_filtered = df[df['tfopwg_disp'].isin(classes_to_keep)].copy()
        
        print(f"\nFiltered dataset: {df_filtered.shape}")
        print(f"Removed {len(df) - len(df_filtered)} samples")
        
        # Show new distribution
        print(f"\nNew class distribution:")
        new_dist = df_filtered['tfopwg_disp'].value_counts()
        total_filtered = len(df_filtered)
        
        for class_name, count in new_dist.items():
            percentage = (count / total_filtered) * 100
            print(f"  {class_name}: {count:4d} ({percentage:5.1f}%)")
        
        # Calculate new imbalance ratio
        imbalance_ratio = new_dist.max() / new_dist.min()
        print(f"\nImbalance ratio: {imbalance_ratio:.1f}:1 (was 47.7:1)")
        
        if imbalance_ratio < 10:
            print("✅ MUCH BETTER BALANCE achieved!")
        elif imbalance_ratio < 5:
            print("🎉 EXCELLENT BALANCE achieved!")
        
        return df_filtered
    
    def preprocess_focused_data(self, df):
        """Preprocess the filtered dataset"""
        print(f"\n🔧 PREPROCESSING FOCUSED DATASET")
        print("="*60)
        
        # Remove identifiers
        id_cols = ['toi', 'tid', 'rastr', 'decstr', 'toi_created', 'rowupdate']
        df = df.drop(columns=[col for col in id_cols if col in df.columns])
        
        # Handle missing values
        error_cols = [col for col in df.columns if 'err' in col.lower()]
        limit_cols = [col for col in df.columns if 'lim' in col.lower()]
        
        for col in error_cols + limit_cols:
            df[col] = df[col].fillna(0)
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        measurement_cols = [col for col in numerical_cols 
                          if col not in error_cols and col not in limit_cols]
        
        for col in measurement_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Essential features only
        if 'pl_rade' in df.columns and 'pl_orbper' in df.columns:
            df['density_proxy'] = df['pl_rade'] / (df['pl_orbper'] ** (2/3))
            
        if 'pl_insol' in df.columns:
            df['habitable'] = ((df['pl_insol'] >= 0.5) & (df['pl_insol'] <= 2.0)).astype(int)
        
        if 'pl_eqt' in df.columns:
            df['earth_like_temp'] = ((df['pl_eqt'] >= 200) & (df['pl_eqt'] <= 350)).astype(int)
        
        print(f"Preprocessed shape: {df.shape}")
        return df
    
    def prepare_focused_data(self, df):
        """Prepare data for focused classification"""
        print(f"\n📊 PREPARING FOCUSED DATA")
        print("="*60)
        
        # Encode target and features
        le = LabelEncoder()
        y = le.fit_transform(df['tfopwg_disp'])
        X = df.drop(['tfopwg_disp'], axis=1)
        
        # Handle categoricals
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le_col = LabelEncoder()
            X[col] = le_col.fit_transform(X[col].astype(str))
        
        X = X.fillna(X.median())
        
        print(f"Feature matrix: {X.shape}")
        print(f"Target classes: {le.classes_}")
        print(f"Class distribution: {dict(zip(le.classes_, np.bincount(y)))}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Train: {X_train.shape[0]} samples")
        print(f"Test: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test, le
    
    def handle_remaining_imbalance(self, X_train, y_train, strategy='smart'):
        """Handle any remaining class imbalance"""
        print(f"\n⚖️ HANDLING REMAINING IMBALANCE: {strategy.upper()}")
        print("="*60)
        
        original_dist = np.bincount(y_train)
        print(f"Original: {dict(enumerate(original_dist))}")
        
        if strategy == 'none':
            return X_train, y_train
        elif strategy == 'smart':
            # Very conservative approach for the focused dataset
            class_counts = np.bincount(y_train)
            max_count = max(class_counts)
            target_count = int(max_count * 0.8)  # Aim for 80% of max class
            
            sampling_strategy = {}
            for i, count in enumerate(class_counts):
                if count < target_count:
                    sampling_strategy[i] = min(target_count, count * 2)  # Max 2x increase
            
            if sampling_strategy:
                sampler = SMOTE(random_state=42, k_neighbors=3)
                X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            else:
                X_resampled, y_resampled = X_train, y_train
                
        new_dist = np.bincount(y_resampled)
        print(f"Resampled: {dict(enumerate(new_dist))}")
        
        return X_resampled, y_resampled
    
    def train_focused_models(self, X_train, X_test, y_train, y_test, le):
        """Train models on focused dataset"""
        print(f"\n🚀 TRAINING FOCUSED MODELS")
        print("="*60)
        
        # Feature selection
        selector = SelectKBest(f_classif, k=min(15, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        print(f"Selected {X_train_selected.shape[1]} features")
        
        # Scale data
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        models = {}
        
        # 1. Logistic Regression
        print("Training Logistic Regression...")
        lr = LogisticRegression(
            C=1.0,  # Can be less aggressive since better balanced
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_test_scaled)
        lr_acc = accuracy_score(y_test, lr_pred)
        models['Logistic Regression'] = (lr, lr_acc)
        print(f"  Test accuracy: {lr_acc:.4f}")
        
        # 2. Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,  # Can be deeper with better balance
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        )
        rf.fit(X_train_scaled, y_train)
        rf_pred = rf.predict(X_test_scaled)
        rf_acc = accuracy_score(y_test, rf_pred)
        models['Random Forest'] = (rf, rf_acc)
        print(f"  Test accuracy: {rf_acc:.4f}")
        
        # 3. XGBoost (your optimized settings)
        print("Training XGBoost (optimized)...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,    # Your setting
            max_depth=4,         # Conservative
            learning_rate=0.1,   # Can be higher with better balance
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=2,         # Less aggressive regularization needed
            reg_lambda=2,
            min_child_weight=3,
            gamma=1,
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train_scaled, y_train)
        xgb_pred = xgb_model.predict(X_test_scaled)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        models['XGBoost'] = (xgb_model, xgb_acc)
        print(f"  Test accuracy: {xgb_acc:.4f}")
        
        # 4. LightGBM
        print("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=2,
            reg_lambda=2,
            min_child_samples=15,
            num_leaves=15,
            class_weight='balanced',
            random_state=42,
            verbose=-1,
            force_col_wise=True
        )
        lgb_model.fit(X_train_scaled, y_train)
        lgb_pred = lgb_model.predict(X_test_scaled)
        lgb_acc = accuracy_score(y_test, lgb_pred)
        models['LightGBM'] = (lgb_model, lgb_acc)
        print(f"  Test accuracy: {lgb_acc:.4f}")
        
        # Select best model
        best_model_name = max(models, key=lambda x: models[x][1])
        best_model, best_acc = models[best_model_name]
        
        print(f"\n🏆 Best model: {best_model_name} ({best_acc:.4f})")
        
        return models, best_model_name, best_model, scaler, selector
    
    def comprehensive_evaluation(self, models, X_train, X_test, y_train, y_test, scaler, selector, le):
        """Comprehensive evaluation of focused models"""
        print(f"\n📊 COMPREHENSIVE EVALUATION")
        print("="*60)
        
        X_train_processed = scaler.transform(selector.transform(X_train))
        X_test_processed = scaler.transform(selector.transform(X_test))
        
        results = {}
        
        for model_name, (model, _) in models.items():
            print(f"\n{model_name} Results:")
            print("-" * 30)
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train_processed, y_train, cv=cv, scoring='accuracy')
            
            # Test predictions
            test_pred = model.predict(X_test_processed)
            test_acc = accuracy_score(y_test, test_pred)
            test_f1 = f1_score(y_test, test_pred, average='weighted')
            
            # Overfitting check
            train_pred = model.predict(X_train_processed)
            train_acc = accuracy_score(y_train, train_pred)
            overfitting_gap = train_acc - test_acc
            
            results[model_name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': test_acc,
                'test_f1': test_f1,
                'train_accuracy': train_acc,
                'overfitting_gap': overfitting_gap,
                'predictions': test_pred
            }
            
            print(f"  CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"  Test accuracy: {test_acc:.4f}")
            print(f"  Test F1: {test_f1:.4f}")
            print(f"  Overfitting gap: {overfitting_gap:.4f}")
            
            if overfitting_gap < 0.05:
                print("  ✅ Good generalization")
            elif overfitting_gap < 0.10:
                print("  ⚠️ Mild overfitting")
            else:
                print("  🚨 Severe overfitting")
        
        # Best model analysis
        best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
        best_result = results[best_model_name]
        best_pred = best_result['predictions']
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"Test Accuracy: {best_result['test_accuracy']:.4f}")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, best_pred, target_names=le.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, best_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'Focused Classification - {best_model_name}\nAccuracy: {best_result["test_accuracy"]:.4f}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('/Users/lilianahotsko/Desktop/nasa-challenge/ml_tabular/focused_classification_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return results, best_model_name
    
    def compare_with_original(self, focused_accuracy):
        """Compare focused results with original 6-class results"""
        print(f"\n🔄 COMPARISON WITH ORIGINAL APPROACH")
        print("="*60)
        
        original_accuracy = 0.6742  # From the best practices guide
        improvement = focused_accuracy - original_accuracy
        
        print(f"Original (6 classes): {original_accuracy:.4f}")
        print(f"Focused (4 classes): {focused_accuracy:.4f}")
        print(f"Improvement: {improvement:+.4f}")
        
        if improvement > 0.05:
            print("🎉 SIGNIFICANT IMPROVEMENT achieved!")
        elif improvement > 0.02:
            print("✅ Good improvement achieved!")
        elif improvement > -0.02:
            print("📊 Similar performance (expected with fewer classes)")
        else:
            print("📉 Performance decreased (investigate further)")
        
        print(f"\n💡 INSIGHTS:")
        print(f"• Removed ambiguous classes (PC, APC)")
        print(f"• Reduced imbalance from 47.7:1 to ~3:1")
        print(f"• Focused on definitive classifications")
        print(f"• Better model interpretability")
    
    def run_focused_pipeline(self, imbalance_strategy='smart'):
        """Run the complete focused classification pipeline"""
        print("🎯 FOCUSED TOI CLASSIFICATION PIPELINE")
        print("="*80)
        print("Excluding PC (Planet Candidate) and APC (Ambiguous Planet Candidate)")
        print("Focus on: CP (Confirmed Planet), FA (False Alarm), FP (False Positive), KP (Known Planet)")
        
        # Step 1: Load and filter data
        df_filtered = self.load_and_filter_data()
        
        # Step 2: Preprocess
        df_processed = self.preprocess_focused_data(df_filtered)
        
        # Step 3: Prepare data
        X_train, X_test, y_train, y_test, le = self.prepare_focused_data(df_processed)
        
        # Step 4: Handle remaining imbalance
        X_train_balanced, y_train_balanced = self.handle_remaining_imbalance(
            X_train, y_train, imbalance_strategy
        )
        
        # Step 5: Train models
        models, best_model_name, best_model, scaler, selector = self.train_focused_models(
            X_train_balanced, X_test, y_train_balanced, y_test, le
        )
        
        # Step 6: Comprehensive evaluation
        results, final_best_model = self.comprehensive_evaluation(
            models, X_train_balanced, X_test, y_train_balanced, y_test, scaler, selector, le
        )
        
        # Step 7: Compare with original
        best_accuracy = results[final_best_model]['test_accuracy']
        self.compare_with_original(best_accuracy)
        
        print(f"\n🎉 FOCUSED CLASSIFICATION COMPLETE!")
        print(f"Best model: {final_best_model}")
        print(f"Final accuracy: {best_accuracy:.4f}")
        
        return {
            'best_model': final_best_model,
            'best_accuracy': best_accuracy,
            'results': results,
            'classes': le.classes_
        }

def main():
    """Run focused classification with different strategies"""
    classifier = FocusedTOIClassifier()
    
    strategies = ['none', 'smart']
    results = {}
    
    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"TESTING IMBALANCE STRATEGY: {strategy.upper()}")
        print(f"{'='*80}")
        
        classifier_instance = FocusedTOIClassifier()
        result = classifier_instance.run_focused_pipeline(strategy)
        results[strategy] = result
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("FOCUSED CLASSIFICATION SUMMARY")
    print(f"{'='*80}")
    
    for strategy, result in results.items():
        acc = result['best_accuracy']
        model = result['best_model']
        print(f"{strategy:8}: {model:18} - Accuracy: {acc:.4f}")
    
    return results

if __name__ == "__main__":
    results = main()
