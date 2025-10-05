"""
Systematic ML Approach for TOI Classification
Comprehensive solution for overfitting, undersampling, and low accuracy
Starting from first principles with best practices
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
import warnings
warnings.filterwarnings('ignore')

class SystematicTOIClassifier:
    def __init__(self, data_path="/Users/lilianahotsko/Desktop/nasa-challenge/data/clean/TOI_2025.10.04_08.44.51.csv"):
        self.data_path = data_path
        self.df = None
        self.scaler = None
        self.label_encoder = None
        self.feature_selector = None
        
    def step1_data_exploration(self):
        """Step 1: Thorough data exploration to understand the problem"""
        print("🔍 STEP 1: COMPREHENSIVE DATA EXPLORATION")
        print("="*60)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        
        # Target analysis
        print(f"\n📊 TARGET DISTRIBUTION ANALYSIS:")
        target_counts = self.df['tfopwg_disp'].value_counts()
        total_samples = len(self.df)
        
        print(f"Total samples: {total_samples}")
        print(f"Number of classes: {len(target_counts)}")
        
        for class_name, count in target_counts.items():
            percentage = (count / total_samples) * 100
            print(f"{class_name}: {count:4d} samples ({percentage:5.1f}%)")
        
        # Calculate imbalance metrics
        majority_class = target_counts.max()
        minority_class = target_counts.min()
        imbalance_ratio = majority_class / minority_class
        
        print(f"\n⚖️ IMBALANCE ANALYSIS:")
        print(f"Majority class: {majority_class} samples")
        print(f"Minority class: {minority_class} samples")
        print(f"Imbalance ratio: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio > 10:
            print("🚨 SEVERE IMBALANCE detected - requires special handling")
        elif imbalance_ratio > 5:
            print("⚠️ MODERATE IMBALANCE - needs attention")
        else:
            print("✅ BALANCED dataset")
        
        # Missing data analysis
        print(f"\n🕳️ MISSING DATA ANALYSIS:")
        missing_data = self.df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            print(f"Columns with missing data: {len(missing_data)}")
            for col, missing_count in missing_data.head(10).items():
                missing_pct = (missing_count / total_samples) * 100
                print(f"  {col}: {missing_count} ({missing_pct:.1f}%)")
        else:
            print("No missing data found")
        
        # Feature types analysis
        print(f"\n🔢 FEATURE TYPES:")
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        print(f"Numerical features: {len(numerical_cols)}")
        print(f"Categorical features: {len(categorical_cols)}")
        
        return {
            'imbalance_ratio': imbalance_ratio,
            'target_counts': target_counts,
            'missing_data': missing_data,
            'numerical_cols': numerical_cols,
            'categorical_cols': categorical_cols
        }
    
    def step2_data_preprocessing(self, exploration_results):
        """Step 2: Systematic data preprocessing"""
        print("\n🔧 STEP 2: SYSTEMATIC DATA PREPROCESSING")
        print("="*60)
        
        # Remove obvious identifier columns
        id_cols = ['toi', 'tid', 'rastr', 'decstr', 'toi_created', 'rowupdate']
        self.df = self.df.drop(columns=[col for col in id_cols if col in self.df.columns])
        print(f"Removed identifier columns: {[col for col in id_cols if col in self.df.columns]}")
        
        # Smart missing value handling
        print(f"\n💊 MISSING VALUE HANDLING:")
        
        # Strategy 1: Domain knowledge for error columns
        error_cols = [col for col in self.df.columns if 'err' in col.lower()]
        limit_cols = [col for col in self.df.columns if 'lim' in col.lower()]
        
        for col in error_cols:
            self.df[col] = self.df[col].fillna(0)
        for col in limit_cols:
            self.df[col] = self.df[col].fillna(0)
        
        print(f"Filled {len(error_cols)} error columns with 0")
        print(f"Filled {len(limit_cols)} limit columns with 0")
        
        # Strategy 2: Median imputation for measurements
        measurement_cols = self.df.select_dtypes(include=[np.number]).columns
        measurement_cols = [col for col in measurement_cols 
                          if col not in error_cols and col not in limit_cols]
        
        for col in measurement_cols:
            if self.df[col].isnull().sum() > 0:
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
        
        print(f"Applied median imputation to {len(measurement_cols)} measurement columns")
        
        # Strategy 3: Mode for categorical
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'tfopwg_disp' and self.df[col].isnull().sum() > 0:
                mode_val = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown'
                self.df[col] = self.df[col].fillna(mode_val)
        
        print(f"Applied mode imputation to categorical columns")
        
        print(f"Dataset shape after preprocessing: {self.df.shape}")
        return self.df
    
    def step3_feature_engineering(self):
        """Step 3: Conservative feature engineering to avoid overfitting"""
        print("\n⚙️ STEP 3: CONSERVATIVE FEATURE ENGINEERING")
        print("="*60)
        
        original_features = self.df.shape[1]
        
        # Only create the most essential domain-knowledge features
        print("Creating essential astrophysical features...")
        
        # Planet physical properties (most important)
        if 'pl_rade' in self.df.columns and 'pl_orbper' in self.df.columns:
            self.df['pl_density_proxy'] = self.df['pl_rade'] / (self.df['pl_orbper'] ** (2/3))
            
        # Habitability indicators (highly relevant)
        if 'pl_insol' in self.df.columns:
            self.df['habitable_zone'] = ((self.df['pl_insol'] >= 0.5) & 
                                       (self.df['pl_insol'] <= 2.0)).astype(int)
            
        if 'pl_eqt' in self.df.columns:
            self.df['earth_like_temp'] = ((self.df['pl_eqt'] >= 200) & 
                                        (self.df['pl_eqt'] <= 350)).astype(int)
        
        # Observational bias indicators
        if 'st_tmag' in self.df.columns:
            self.df['observable_star'] = (self.df['st_tmag'] < 12).astype(int)
        
        new_features = self.df.shape[1] - original_features
        print(f"Added {new_features} essential features")
        print(f"Total features: {self.df.shape[1]}")
        
        return self.df
    
    def step4_smart_train_test_split(self):
        """Step 4: Smart train/validation/test splitting"""
        print("\n📊 STEP 4: SMART DATA SPLITTING")
        print("="*60)
        
        # Encode target
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(self.df['tfopwg_disp'])
        X = self.df.drop(['tfopwg_disp'], axis=1)
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le_col = LabelEncoder()
            X[col] = le_col.fit_transform(X[col].astype(str))
        
        X = X.fillna(X.median())
        
        print(f"Feature matrix: {X.shape}")
        print(f"Target classes: {self.label_encoder.classes_}")
        
        # Strategic splitting: larger test set for reliable evaluation
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y  # 25% test
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp  # 15% validation
        )
        
        print(f"Training set: {X_train.shape} ({len(y_train)/len(y)*100:.1f}%)")
        print(f"Validation set: {X_val.shape} ({len(y_val)/len(y)*100:.1f}%)")
        print(f"Test set: {X_test.shape} ({len(y_test)/len(y)*100:.1f}%)")
        
        # Check class distribution in splits
        print(f"\nClass distribution in splits:")
        train_dist = np.bincount(y_train)
        val_dist = np.bincount(y_val)
        test_dist = np.bincount(y_test)
        
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"{class_name}: Train={train_dist[i]}, Val={val_dist[i]}, Test={test_dist[i]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def step5_handle_class_imbalance(self, X_train, y_train, strategy='smart'):
        """Step 5: Systematic approach to class imbalance"""
        print(f"\n⚖️ STEP 5: HANDLING CLASS IMBALANCE - {strategy.upper()} STRATEGY")
        print("="*60)
        
        original_dist = np.bincount(y_train)
        print(f"Original distribution: {dict(enumerate(original_dist))}")
        
        if strategy == 'none':
            return X_train, y_train
        
        elif strategy == 'undersample':
            # Conservative undersampling
            undersampler = RandomUnderSampler(random_state=42, sampling_strategy='auto')
            X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
            
        elif strategy == 'oversample':
            # Conservative oversampling
            oversampler = SMOTE(random_state=42, k_neighbors=5)
            X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)
            
        elif strategy == 'combine':
            # Combined approach
            combiner = SMOTETomek(random_state=42)
            X_resampled, y_resampled = combiner.fit_resample(X_train, y_train)
            
        elif strategy == 'smart':
            # Smart strategy: partial balancing to avoid overfitting
            # Don't fully balance - aim for 70% of majority class
            class_counts = np.bincount(y_train)
            max_samples = int(max(class_counts) * 0.7)
            
            sampling_strategy = {}
            for i, count in enumerate(class_counts):
                if count < max_samples:
                    sampling_strategy[i] = max_samples
            
            if sampling_strategy:
                smart_sampler = SMOTE(random_state=42, sampling_strategy=sampling_strategy, k_neighbors=3)
                X_resampled, y_resampled = smart_sampler.fit_resample(X_train, y_train)
            else:
                X_resampled, y_resampled = X_train, y_train
        
        new_dist = np.bincount(y_resampled)
        print(f"Resampled distribution: {dict(enumerate(new_dist))}")
        print(f"Samples: {len(y_train)} → {len(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def step6_feature_selection(self, X_train, y_train, method='smart'):
        """Step 6: Intelligent feature selection to prevent overfitting"""
        print(f"\n🎯 STEP 6: FEATURE SELECTION - {method.upper()} METHOD")
        print("="*60)
        
        print(f"Starting with {X_train.shape[1]} features")
        
        if method == 'none':
            return X_train, None
        
        elif method == 'variance':
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
            X_selected = selector.fit_transform(X_train)
            
        elif method == 'univariate':
            # Select top K features
            k = min(20, X_train.shape[1])
            selector = SelectKBest(f_classif, k=k)
            X_selected = selector.fit_transform(X_train, y_train)
            
        elif method == 'smart':
            # Multi-step selection
            # Step 1: Remove low variance
            variance_selector = VarianceThreshold(threshold=0.01)
            X_variance = variance_selector.fit_transform(X_train)
            
            # Step 2: Select top features
            k = min(25, X_variance.shape[1])
            univariate_selector = SelectKBest(f_classif, k=k)
            X_selected = univariate_selector.fit_transform(X_variance, y_train)
            
            # Combine selectors
            selector = (variance_selector, univariate_selector)
        
        print(f"Selected {X_selected.shape[1]} features")
        return X_selected, selector
    
    def step7_model_selection_and_training(self, X_train, X_val, y_train, y_val):
        """Step 7: Systematic model selection with overfitting prevention"""
        print("\n🚀 STEP 7: SYSTEMATIC MODEL TRAINING")
        print("="*60)
        
        # Scale data
        self.scaler = RobustScaler()  # More robust to outliers
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        models = {}
        results = {}
        
        # 1. Logistic Regression (baseline)
        print("Training Logistic Regression (baseline)...")
        lr_model = LogisticRegression(
            random_state=42, 
            max_iter=1000,
            C=1.0,  # Regularization strength
            class_weight='balanced'
        )
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_val_scaled)
        lr_acc = accuracy_score(y_val, lr_pred)
        
        models['Logistic Regression'] = lr_model
        results['Logistic Regression'] = lr_acc
        print(f"  Validation accuracy: {lr_acc:.4f}")
        
        # 2. Random Forest (ensemble baseline)
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,  # Conservative
            max_depth=8,       # Prevent overfitting
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        )
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_val_scaled)
        rf_acc = accuracy_score(y_val, rf_pred)
        
        models['Random Forest'] = rf_model
        results['Random Forest'] = rf_acc
        print(f"  Validation accuracy: {rf_acc:.4f}")
        
        # 3. XGBoost (regularized)
        print("Training XGBoost (regularized)...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,    # Conservative (user's suggestion)
            max_depth=4,         # Shallow trees
            learning_rate=0.05,  # Slow learning
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=2,         # L1 regularization
            reg_lambda=2,        # L2 regularization
            min_child_weight=5,
            gamma=1,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        xgb_model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        xgb_pred = xgb_model.predict(X_val_scaled)
        xgb_acc = accuracy_score(y_val, xgb_pred)
        
        models['XGBoost'] = xgb_model
        results['XGBoost'] = xgb_acc
        print(f"  Validation accuracy: {xgb_acc:.4f}")
        
        # 4. LightGBM (regularized)
        print("Training LightGBM (regularized)...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=2,
            reg_lambda=2,
            min_child_samples=20,
            class_weight='balanced',
            random_state=42,
            verbose=-1,
            force_col_wise=True
        )
        
        lgb_model.fit(X_train_scaled, y_train)
        lgb_pred = lgb_model.predict(X_val_scaled)
        lgb_acc = accuracy_score(y_val, lgb_pred)
        
        models['LightGBM'] = lgb_model
        results['LightGBM'] = lgb_acc
        print(f"  Validation accuracy: {lgb_acc:.4f}")
        
        # Select best model
        best_model_name = max(results, key=results.get)
        best_model = models[best_model_name]
        best_acc = results[best_model_name]
        
        print(f"\n🏆 Best model: {best_model_name} ({best_acc:.4f})")
        
        return models, results, best_model_name, best_model
    
    def step8_cross_validation_and_evaluation(self, best_model, X_train, y_train):
        """Step 8: Robust cross-validation evaluation"""
        print("\n📊 STEP 8: CROSS-VALIDATION EVALUATION")
        print("="*60)
        
        # Stratified K-Fold cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy')
        
        print(f"Cross-validation results:")
        print(f"  Individual folds: {[f'{score:.4f}' for score in cv_scores]}")
        print(f"  Mean CV accuracy: {cv_scores.mean():.4f}")
        print(f"  Standard deviation: {cv_scores.std():.4f}")
        print(f"  95% Confidence interval: {cv_scores.mean():.4f} ± {cv_scores.std() * 2:.4f}")
        
        return cv_scores
    
    def step9_final_test_evaluation(self, models, X_test, y_test):
        """Step 9: Final test set evaluation"""
        print("\n🎯 STEP 9: FINAL TEST SET EVALUATION")
        print("="*60)
        
        # Scale test set
        X_test_scaled = self.scaler.transform(X_test)
        
        test_results = {}
        
        for model_name, model in models.items():
            pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, average='weighted')
            
            test_results[model_name] = {
                'accuracy': acc,
                'f1_score': f1,
                'predictions': pred
            }
            
            print(f"{model_name}:")
            print(f"  Test accuracy: {acc:.4f}")
            print(f"  F1-score: {f1:.4f}")
        
        # Best model on test set
        best_test_model = max(test_results, key=lambda x: test_results[x]['accuracy'])
        best_test_acc = test_results[best_test_model]['accuracy']
        best_predictions = test_results[best_test_model]['predictions']
        
        print(f"\n🏆 Best test performance: {best_test_model} ({best_test_acc:.4f})")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report ({best_test_model}):")
        target_names = self.label_encoder.classes_
        print(classification_report(y_test, best_predictions, target_names=target_names))
        
        return test_results, best_test_model, best_test_acc
    
    def step10_analysis_and_recommendations(self, cv_scores, test_results):
        """Step 10: Analysis and recommendations"""
        print("\n📋 STEP 10: ANALYSIS AND RECOMMENDATIONS")
        print("="*60)
        
        best_cv_acc = cv_scores.mean()
        best_test_acc = max([r['accuracy'] for r in test_results.values()])
        
        generalization_gap = abs(best_cv_acc - best_test_acc)
        
        print(f"🔍 PERFORMANCE ANALYSIS:")
        print(f"  Best CV accuracy: {best_cv_acc:.4f}")
        print(f"  Best test accuracy: {best_test_acc:.4f}")
        print(f"  Generalization gap: {generalization_gap:.4f}")
        
        # Diagnosis
        if generalization_gap < 0.02:
            print("✅ EXCELLENT generalization")
        elif generalization_gap < 0.05:
            print("✅ GOOD generalization")
        elif generalization_gap < 0.10:
            print("⚠️ MODERATE overfitting")
        else:
            print("🚨 SEVERE overfitting")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if best_test_acc < 0.70:
            print("📈 LOW ACCURACY - Try:")
            print("   1. More sophisticated feature engineering")
            print("   2. Ensemble methods")
            print("   3. Deep learning approaches")
            print("   4. External data sources")
        
        if generalization_gap > 0.05:
            print("🛡️ OVERFITTING - Try:")
            print("   1. More aggressive regularization")
            print("   2. Reduce model complexity")
            print("   3. More training data")
            print("   4. Feature selection")
        
        if best_test_acc >= 0.75 and generalization_gap < 0.05:
            print("🎉 GOOD PERFORMANCE - Next steps:")
            print("   1. Hyperparameter optimization")
            print("   2. Advanced ensemble methods")
            print("   3. Model interpretation")
        
        return {
            'cv_accuracy': best_cv_acc,
            'test_accuracy': best_test_acc,
            'generalization_gap': generalization_gap
        }
    
    def run_complete_pipeline(self, imbalance_strategy='smart', feature_selection='smart'):
        """Run the complete systematic pipeline"""
        print("🏗️ SYSTEMATIC TOI CLASSIFICATION PIPELINE")
        print("="*80)
        print(f"Imbalance strategy: {imbalance_strategy}")
        print(f"Feature selection: {feature_selection}")
        
        # Execute all steps
        exploration_results = self.step1_data_exploration()
        self.step2_data_preprocessing(exploration_results)
        self.step3_feature_engineering()
        X_train, X_val, X_test, y_train, y_val, y_test = self.step4_smart_train_test_split()
        X_train_balanced, y_train_balanced = self.step5_handle_class_imbalance(X_train, y_train, imbalance_strategy)
        X_train_selected, feature_selector = self.step6_feature_selection(X_train_balanced, y_train_balanced, feature_selection)
        
        # Apply same transformations to validation and test sets
        if feature_selector is not None:
            if isinstance(feature_selector, tuple):
                X_val_selected = feature_selector[0].transform(X_val)
                X_val_selected = feature_selector[1].transform(X_val_selected)
                X_test_selected = feature_selector[0].transform(X_test)
                X_test_selected = feature_selector[1].transform(X_test_selected)
            else:
                X_val_selected = feature_selector.transform(X_val)
                X_test_selected = feature_selector.transform(X_test)
        else:
            X_val_selected = X_val
            X_test_selected = X_test
        
        models, val_results, best_model_name, best_model = self.step7_model_selection_and_training(
            X_train_selected, X_val_selected, y_train_balanced, y_val
        )
        
        cv_scores = self.step8_cross_validation_and_evaluation(best_model, X_train_selected, y_train_balanced)
        test_results, best_test_model, best_test_acc = self.step9_final_test_evaluation(models, X_test_selected, y_test)
        analysis = self.step10_analysis_and_recommendations(cv_scores, test_results)
        
        print(f"\n🎉 PIPELINE COMPLETE!")
        print(f"Best model: {best_test_model}")
        print(f"Final accuracy: {best_test_acc:.4f}")
        
        return {
            'models': models,
            'best_model': best_test_model,
            'test_accuracy': best_test_acc,
            'cv_scores': cv_scores,
            'analysis': analysis
        }

def main():
    """Run systematic comparison of different strategies"""
    classifier = SystematicTOIClassifier()
    
    # Test different strategies
    strategies = [
        ('none', 'none'),
        ('smart', 'smart'),
        ('combine', 'univariate'),
    ]
    
    results = {}
    
    for imbalance_strategy, feature_strategy in strategies:
        print(f"\n{'='*80}")
        print(f"TESTING: Imbalance={imbalance_strategy}, Features={feature_strategy}")
        print(f"{'='*80}")
        
        classifier_instance = SystematicTOIClassifier()
        result = classifier_instance.run_complete_pipeline(imbalance_strategy, feature_strategy)
        results[f"{imbalance_strategy}_{feature_strategy}"] = result
    
    # Compare results
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON")
    print(f"{'='*80}")
    
    for strategy_name, result in results.items():
        print(f"{strategy_name}: {result['test_accuracy']:.4f}")
    
    return results

if __name__ == "__main__":
    results = main()
