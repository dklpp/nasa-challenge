"""
Best Practices Guide for TOI Classification
Systematic approach to handle overfitting, undersampling, and low accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

class BestPracticesTOIClassifier:
    def __init__(self, data_path="/Users/lilianahotsko/Desktop/nasa-challenge/data/clean/TOI_2025.10.04_08.44.51.csv"):
        self.data_path = data_path
        
    def analyze_problem(self):
        """Analyze the core problems: overfitting, imbalance, low accuracy"""
        print("🔍 PROBLEM ANALYSIS")
        print("="*60)
        
        df = pd.read_csv(self.data_path)
        target_counts = df['tfopwg_disp'].value_counts()
        
        print(f"Dataset: {df.shape}")
        print(f"Classes: {len(target_counts)}")
        print(f"Imbalance ratio: {target_counts.max()/target_counts.min():.1f}:1")
        
        # The three main problems:
        problems = {
            'Class Imbalance': target_counts.max()/target_counts.min() > 10,
            'High Dimensionality': df.shape[1] > 50,
            'Small Dataset': df.shape[0] < 10000
        }
        
        print(f"\n🚨 IDENTIFIED PROBLEMS:")
        for problem, exists in problems.items():
            status = "YES" if exists else "NO"
            print(f"  {problem}: {status}")
        
        return problems
    
    def best_practice_preprocessing(self):
        """Apply best practices for preprocessing"""
        print(f"\n🔧 BEST PRACTICE PREPROCESSING")
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
        
        # Median imputation for measurements
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        measurement_cols = [col for col in numerical_cols 
                          if col not in error_cols and col not in limit_cols]
        
        for col in measurement_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Only essential features to prevent overfitting
        if 'pl_rade' in df.columns and 'pl_orbper' in df.columns:
            df['density_proxy'] = df['pl_rade'] / (df['pl_orbper'] ** (2/3))
            
        if 'pl_insol' in df.columns:
            df['habitable'] = ((df['pl_insol'] >= 0.5) & (df['pl_insol'] <= 2.0)).astype(int)
        
        print(f"Preprocessed shape: {df.shape}")
        return df
    
    def best_practice_train_test_split(self, df):
        """Optimal train/test splitting strategy"""
        print(f"\n📊 BEST PRACTICE DATA SPLITTING")
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
        
        # Stratified split with larger test set for reliable evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Train: {X_train.shape[0]} samples")
        print(f"Test: {X_test.shape[0]} samples")
        print(f"Classes: {le.classes_}")
        
        return X_train, X_test, y_train, y_test, le
    
    def best_practice_imbalance_handling(self, X_train, y_train, strategy='conservative'):
        """Best practices for handling class imbalance"""
        print(f"\n⚖️ BEST PRACTICE IMBALANCE HANDLING: {strategy.upper()}")
        print("="*60)
        
        original_dist = np.bincount(y_train)
        print(f"Original: {dict(enumerate(original_dist))}")
        
        if strategy == 'none':
            return X_train, y_train
        elif strategy == 'conservative':
            # Conservative SMOTE - don't fully balance
            class_counts = np.bincount(y_train)
            target_count = int(np.median(class_counts) * 1.5)  # 1.5x median
            
            sampling_strategy = {}
            for i, count in enumerate(class_counts):
                if count < target_count:
                    sampling_strategy[i] = min(target_count, count * 3)  # Max 3x increase
            
            if sampling_strategy:
                sampler = SMOTE(random_state=42, k_neighbors=3)
                X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            else:
                X_resampled, y_resampled = X_train, y_train
        elif strategy == 'balanced':
            sampler = SMOTE(random_state=42, k_neighbors=5)
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        
        new_dist = np.bincount(y_resampled)
        print(f"Resampled: {dict(enumerate(new_dist))}")
        
        return X_resampled, y_resampled
    
    def best_practice_feature_selection(self, X_train, y_train, n_features=20):
        """Best practices for feature selection"""
        print(f"\n🎯 BEST PRACTICE FEATURE SELECTION")
        print("="*60)
        
        print(f"Selecting top {n_features} features from {X_train.shape[1]}")
        
        selector = SelectKBest(f_classif, k=min(n_features, X_train.shape[1]))
        X_selected = selector.fit_transform(X_train, y_train)
        
        print(f"Selected: {X_selected.shape[1]} features")
        return X_selected, selector
    
    def best_practice_model_training(self, X_train, X_test, y_train, y_test):
        """Best practices for model training with overfitting prevention"""
        print(f"\n🚀 BEST PRACTICE MODEL TRAINING")
        print("="*60)
        
        # Scale data
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        
        # 1. Regularized Logistic Regression
        print("Training Regularized Logistic Regression...")
        lr = LogisticRegression(
            C=0.1,  # Strong regularization
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_test_scaled)
        lr_acc = accuracy_score(y_test, lr_pred)
        models['Logistic Regression'] = (lr, lr_acc)
        print(f"  Test accuracy: {lr_acc:.4f}")
        
        # 2. Conservative Random Forest
        print("Training Conservative Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,  # Shallow trees
            min_samples_split=20,  # Conservative splits
            min_samples_leaf=10,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        )
        rf.fit(X_train_scaled, y_train)
        rf_pred = rf.predict(X_test_scaled)
        rf_acc = accuracy_score(y_test, rf_pred)
        models['Random Forest'] = (rf, rf_acc)
        print(f"  Test accuracy: {rf_acc:.4f}")
        
        # 3. Heavily Regularized XGBoost
        print("Training Heavily Regularized XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=50,     # Very conservative
            max_depth=3,         # Very shallow
            learning_rate=0.05,  # Very slow
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=5,         # Very strong L1
            reg_lambda=5,        # Very strong L2
            min_child_weight=10,
            gamma=2,
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train_scaled, y_train)
        xgb_pred = xgb_model.predict(X_test_scaled)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        models['XGBoost'] = (xgb_model, xgb_acc)
        print(f"  Test accuracy: {xgb_acc:.4f}")
        
        # 4. Conservative LightGBM
        print("Training Conservative LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=5,
            reg_lambda=5,
            min_child_samples=30,
            num_leaves=8,  # Very few leaves
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
        
        return models, best_model_name, best_model, scaler
    
    def best_practice_evaluation(self, model, X_train, X_test, y_train, y_test, scaler, le):
        """Best practices for model evaluation"""
        print(f"\n📊 BEST PRACTICE EVALUATION")
        print("="*60)
        
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Cross-validation for robustness
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        
        print(f"Cross-validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Test set evaluation
        test_pred = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average='weighted')
        
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Test F1-score: {test_f1:.4f}")
        
        # Overfitting check
        train_pred = model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        overfitting_gap = train_acc - test_acc
        
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Overfitting gap: {overfitting_gap:.4f}")
        
        if overfitting_gap < 0.05:
            print("✅ Good generalization")
        elif overfitting_gap < 0.10:
            print("⚠️ Mild overfitting")
        else:
            print("🚨 Severe overfitting")
        
        # Detailed report
        print(f"\nClassification Report:")
        print(classification_report(y_test, test_pred, target_names=le.classes_))
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'overfitting_gap': overfitting_gap
        }
    
    def run_best_practices_pipeline(self, imbalance_strategy='conservative'):
        """Run the complete best practices pipeline"""
        print("🏗️ BEST PRACTICES TOI CLASSIFICATION PIPELINE")
        print("="*80)
        
        # Step 1: Analyze the problem
        problems = self.analyze_problem()
        
        # Step 2: Best practice preprocessing
        df = self.best_practice_preprocessing()
        
        # Step 3: Optimal data splitting
        X_train, X_test, y_train, y_test, le = self.best_practice_train_test_split(df)
        
        # Step 4: Handle class imbalance
        X_train_balanced, y_train_balanced = self.best_practice_imbalance_handling(
            X_train, y_train, imbalance_strategy
        )
        
        # Step 5: Feature selection
        X_train_selected, selector = self.best_practice_feature_selection(
            X_train_balanced, y_train_balanced, n_features=15
        )
        X_test_selected = selector.transform(X_test)
        
        # Step 6: Model training
        models, best_model_name, best_model, scaler = self.best_practice_model_training(
            X_train_selected, X_test_selected, y_train_balanced, y_test
        )
        
        # Step 7: Comprehensive evaluation
        evaluation = self.best_practice_evaluation(
            best_model, X_train_selected, X_test_selected, y_train_balanced, y_test, scaler, le
        )
        
        # Step 8: Recommendations
        self.provide_recommendations(evaluation, problems)
        
        return {
            'best_model': best_model_name,
            'evaluation': evaluation,
            'models': models
        }
    
    def provide_recommendations(self, evaluation, problems):
        """Provide specific recommendations based on results"""
        print(f"\n💡 SPECIFIC RECOMMENDATIONS")
        print("="*60)
        
        test_acc = evaluation['test_accuracy']
        overfitting_gap = evaluation['overfitting_gap']
        
        print(f"Current performance: {test_acc:.4f}")
        print(f"Overfitting gap: {overfitting_gap:.4f}")
        
        if test_acc < 0.65:
            print(f"\n🚨 LOW ACCURACY - Priority actions:")
            print(f"1. Try ensemble methods (voting, stacking)")
            print(f"2. More sophisticated feature engineering")
            print(f"3. Collect more training data")
            print(f"4. Try neural networks with dropout")
        
        elif test_acc < 0.75:
            print(f"\n📈 MODERATE ACCURACY - Improvement strategies:")
            print(f"1. Hyperparameter optimization with Optuna")
            print(f"2. Advanced ensemble methods")
            print(f"3. Class-specific models for difficult classes")
        
        else:
            print(f"\n✅ GOOD ACCURACY - Fine-tuning:")
            print(f"1. Model interpretation and analysis")
            print(f"2. Threshold optimization")
            print(f"3. Ensemble refinement")
        
        if overfitting_gap > 0.10:
            print(f"\n🛡️ OVERFITTING - Immediate actions:")
            print(f"1. Increase regularization further")
            print(f"2. Reduce model complexity")
            print(f"3. More aggressive feature selection")
            print(f"4. Increase training data")
        
        if problems.get('Class Imbalance', False):
            print(f"\n⚖️ IMBALANCE - Advanced techniques:")
            print(f"1. Cost-sensitive learning")
            print(f"2. Focal loss for neural networks")
            print(f"3. Ensemble of balanced classifiers")

def main():
    """Run best practices comparison"""
    classifier = BestPracticesTOIClassifier()
    
    strategies = ['none', 'conservative', 'balanced']
    results = {}
    
    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"TESTING IMBALANCE STRATEGY: {strategy.upper()}")
        print(f"{'='*80}")
        
        classifier_instance = BestPracticesTOIClassifier()
        result = classifier_instance.run_best_practices_pipeline(strategy)
        results[strategy] = result
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    for strategy, result in results.items():
        acc = result['evaluation']['test_accuracy']
        gap = result['evaluation']['overfitting_gap']
        print(f"{strategy:12}: Accuracy={acc:.4f}, Overfitting={gap:.4f}")
    
    return results

if __name__ == "__main__":
    results = main()
