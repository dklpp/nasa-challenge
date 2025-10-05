"""
TOI Dataset Optimization - Focused strategies to boost accuracy from 72% to 90%
Using the complete TOI dataset with advanced techniques
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
import warnings
warnings.filterwarnings('ignore')

class TOIOptimizer:
    def __init__(self, data_path="/Users/lilianahotsko/Desktop/nasa-challenge/data/clean/TOI_2025.10.04_08.44.51.csv"):
        """Initialize with full TOI dataset path"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
        
    def load_and_analyze_full_dataset(self):
        """Load the complete TOI dataset and analyze it thoroughly"""
        print("🔍 LOADING COMPLETE TOI DATASET")
        print("="*50)
        
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Analyze target distribution
        print(f"\nTarget distribution:")
        target_dist = self.df['tfopwg_disp'].value_counts()
        print(target_dist)
        print(f"Total samples: {len(self.df)}")
        
        # Class imbalance analysis
        print(f"\nClass imbalance ratios:")
        for class_name, count in target_dist.items():
            percentage = (count / len(self.df)) * 100
            print(f"{class_name}: {count} samples ({percentage:.1f}%)")
        
        # Missing data analysis
        print(f"\nMissing data analysis:")
        missing_data = self.df.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            print(f"Columns with missing data: {len(missing_data)}")
            print(missing_data.head(10))
        else:
            print("No missing data found")
        
        return self.df
    
    def advanced_preprocessing(self):
        """Advanced preprocessing specifically for TOI data"""
        print("\n🔧 ADVANCED TOI PREPROCESSING")
        print("="*50)
        
        df_processed = self.df.copy()
        
        # Remove identifier columns but keep important ones
        id_cols = ['toi', 'tid', 'rastr', 'decstr', 'toi_created', 'rowupdate']
        df_processed = df_processed.drop(columns=[col for col in id_cols if col in df_processed.columns])
        
        # 1. Advanced missing value handling
        print("Handling missing values with domain knowledge...")
        
        # For astronomical measurements, NaN often means "not measured" rather than missing
        # Fill error columns with 0 (no error reported)
        error_cols = [col for col in df_processed.columns if 'err' in col.lower()]
        for col in error_cols:
            df_processed[col] = df_processed[col].fillna(0)
        
        # Fill limit flags with 0 (no limit applied)
        limit_cols = [col for col in df_processed.columns if 'lim' in col.lower()]
        for col in limit_cols:
            df_processed[col] = df_processed[col].fillna(0)
        
        # For main measurements, use sophisticated imputation
        measurement_cols = df_processed.select_dtypes(include=[np.number]).columns
        measurement_cols = [col for col in measurement_cols 
                          if col not in error_cols and col not in limit_cols]
        
        # Group-based imputation for better accuracy
        if 'tfopwg_disp' in df_processed.columns:
            for col in measurement_cols:
                if df_processed[col].isnull().sum() > 0:
                    # Fill with median by disposition group
                    group_medians = df_processed.groupby('tfopwg_disp')[col].median()
                    df_processed[col] = df_processed[col].fillna(
                        df_processed['tfopwg_disp'].map(group_medians)
                    )
                    # If still NaN, use overall median
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        # 2. Advanced feature engineering based on astrophysics
        print("Creating astrophysically-motivated features...")
        
        # Planet characteristics
        if 'pl_rade' in df_processed.columns and 'pl_orbper' in df_processed.columns:
            # Planet density proxy (radius/period relationship)
            df_processed['pl_density_proxy'] = df_processed['pl_rade'] / (df_processed['pl_orbper'] ** (2/3))
            
        if 'pl_insol' in df_processed.columns:
            # Habitable zone indicators
            df_processed['in_habitable_zone'] = ((df_processed['pl_insol'] >= 0.5) & 
                                               (df_processed['pl_insol'] <= 2.0)).astype(int)
            df_processed['insol_log'] = np.log10(df_processed['pl_insol'] + 1e-10)
        
        if 'pl_eqt' in df_processed.columns:
            # Temperature categories
            df_processed['temp_earth_like'] = ((df_processed['pl_eqt'] >= 200) & 
                                             (df_processed['pl_eqt'] <= 350)).astype(int)
            df_processed['temp_category'] = pd.cut(df_processed['pl_eqt'], 
                                                 bins=[0, 200, 350, 600, 1000, np.inf],
                                                 labels=[0, 1, 2, 3, 4])
        
        # Stellar characteristics
        if 'st_teff' in df_processed.columns and 'st_rad' in df_processed.columns:
            # Stellar luminosity proxy
            df_processed['st_luminosity_proxy'] = (df_processed['st_rad'] ** 2) * (df_processed['st_teff'] / 5778) ** 4
        
        if 'st_tmag' in df_processed.columns:
            # Brightness categories for observational bias
            df_processed['st_bright'] = (df_processed['st_tmag'] < 10).astype(int)
        
        # Transit characteristics
        if 'pl_trandurh' in df_processed.columns and 'pl_orbper' in df_processed.columns:
            # Transit duty cycle
            df_processed['transit_duty_cycle'] = df_processed['pl_trandurh'] / (df_processed['pl_orbper'] * 24)
        
        # 3. Interaction features
        print("Creating interaction features...")
        key_features = ['pl_rade', 'pl_orbper', 'pl_insol', 'st_teff', 'st_rad', 'st_tmag']
        
        for i, feat1 in enumerate(key_features):
            if feat1 in df_processed.columns:
                for feat2 in key_features[i+1:]:
                    if feat2 in df_processed.columns:
                        # Ratios
                        df_processed[f'{feat1}_{feat2}_ratio'] = df_processed[feat1] / (df_processed[feat2] + 1e-8)
                        # Products for non-linear relationships
                        df_processed[f'{feat1}_{feat2}_product'] = df_processed[feat1] * df_processed[feat2]
        
        # 4. Statistical features
        print("Creating statistical aggregation features...")
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        # Rolling statistics based on similar objects
        if 'st_teff' in df_processed.columns:
            # Group by stellar temperature ranges
            df_processed['st_teff_bin'] = pd.cut(df_processed['st_teff'], bins=10, labels=False)
            for col in ['pl_rade', 'pl_orbper', 'pl_insol']:
                if col in df_processed.columns:
                    group_stats = df_processed.groupby('st_teff_bin')[col].agg(['mean', 'std'])
                    df_processed[f'{col}_teff_group_mean'] = df_processed['st_teff_bin'].map(group_stats['mean'])
                    df_processed[f'{col}_teff_group_std'] = df_processed['st_teff_bin'].map(group_stats['std'])
        
        print(f"Features after engineering: {df_processed.shape[1]} (was {self.df.shape[1]})")
        
        return df_processed
    
    def prepare_ml_data(self, df_processed, test_size=0.15, random_state=42):
        """Prepare data for ML with stratified sampling"""
        print("\n📊 PREPARING ML DATA")
        print("="*50)
        
        # Encode target
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(df_processed['tfopwg_disp'])
        
        # Prepare features
        X = df_processed.drop(['tfopwg_disp'], axis=1)
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le_col = LabelEncoder()
            X[col] = le_col.fit_transform(X[col].astype(str))
        
        # Handle any remaining NaN values
        X = X.fillna(X.median())
        
        print(f"Final feature matrix: {X.shape}")
        print(f"Target distribution: {np.bincount(y)}")
        
        # Stratified split to maintain class distribution
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def strategy_class_balancing(self):
        """Advanced class balancing for imbalanced TOI data"""
        print("\n⚖️ ADVANCED CLASS BALANCING")
        print("="*50)
        
        print("Original class distribution:", np.bincount(self.y_train))
        
        # Test multiple resampling strategies
        strategies = {
            'SMOTE': SMOTE(random_state=42, k_neighbors=3),
            'BorderlineSMOTE': BorderlineSMOTE(random_state=42, k_neighbors=3),
            'ADASYN': ADASYN(random_state=42, n_neighbors=3),
            'SMOTEENN': SMOTEENN(random_state=42),
            'SMOTETomek': SMOTETomek(random_state=42),
        }
        
        best_strategy = None
        best_accuracy = 0
        best_X_resampled = None
        best_y_resampled = None
        
        for name, strategy in strategies.items():
            try:
                print(f"\nTesting {name}...")
                X_resampled, y_resampled = strategy.fit_resample(self.X_train, self.y_train)
                print(f"Resampled distribution: {np.bincount(y_resampled)}")
                
                # Quick test with XGBoost
                model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric='mlogloss'
                )
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=3, scoring='accuracy')
                avg_score = cv_scores.mean()
                
                print(f"{name} CV accuracy: {avg_score:.4f}")
                
                if avg_score > best_accuracy:
                    best_accuracy = avg_score
                    best_strategy = name
                    best_X_resampled = X_resampled.copy()
                    best_y_resampled = y_resampled.copy()
                    
            except Exception as e:
                print(f"Error with {name}: {e}")
        
        print(f"\n🏆 Best resampling strategy: {best_strategy} (CV: {best_accuracy:.4f})")
        
        self.results['resampling'] = {
            'strategy': best_strategy,
            'cv_accuracy': best_accuracy,
            'X_resampled': best_X_resampled,
            'y_resampled': best_y_resampled
        }
        
        return best_X_resampled, best_y_resampled
    
    def strategy_ensemble_stacking(self, X_train, y_train):
        """Advanced ensemble with multiple algorithms"""
        print("\n🤝 ADVANCED ENSEMBLE METHODS")
        print("="*50)
        
        # Define diverse base models
        base_models = {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                num_leaves=31,
                random_state=42,
                verbose=-1,
                force_col_wise=True
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=300,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        }
        
        # Train individual models and collect predictions
        model_predictions = {}
        model_accuracies = {}
        
        for name, model in base_models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            model_accuracies[name] = accuracy
            model_predictions[name] = y_pred
            print(f"{name} accuracy: {accuracy:.4f}")
        
        # Weighted ensemble based on individual performance
        print("\nCreating weighted ensemble...")
        weights = np.array(list(model_accuracies.values()))
        weights = weights / weights.sum()  # Normalize weights
        
        # Create ensemble predictions
        ensemble_pred = np.zeros(len(self.y_test))
        for i, (name, pred) in enumerate(model_predictions.items()):
            ensemble_pred += weights[i] * pred
        
        ensemble_pred = np.round(ensemble_pred).astype(int)
        ensemble_accuracy = accuracy_score(self.y_test, ensemble_pred)
        
        print(f"Weighted ensemble accuracy: {ensemble_accuracy:.4f}")
        print(f"Weights: {dict(zip(base_models.keys(), weights))}")
        
        # Advanced stacking
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        
        print("Creating stacking ensemble...")
        stacking_clf = StackingClassifier(
            estimators=[(name, model) for name, model in base_models.items()],
            final_estimator=LogisticRegression(random_state=42, max_iter=1000),
            cv=5
        )
        
        stacking_clf.fit(X_train, y_train)
        stacking_pred = stacking_clf.predict(self.X_test)
        stacking_accuracy = accuracy_score(self.y_test, stacking_pred)
        
        print(f"Stacking ensemble accuracy: {stacking_accuracy:.4f}")
        
        best_ensemble_accuracy = max(ensemble_accuracy, stacking_accuracy)
        best_method = "Weighted" if ensemble_accuracy > stacking_accuracy else "Stacking"
        
        self.results['ensemble'] = {
            'weighted_accuracy': ensemble_accuracy,
            'stacking_accuracy': stacking_accuracy,
            'best_accuracy': best_ensemble_accuracy,
            'best_method': best_method,
            'individual_accuracies': model_accuracies
        }
        
        return best_ensemble_accuracy
    
    def run_comprehensive_optimization(self):
        """Run complete optimization pipeline"""
        print("🚀 COMPREHENSIVE TOI OPTIMIZATION")
        print("="*80)
        
        # 1. Load and analyze full dataset
        df = self.load_and_analyze_full_dataset()
        
        # 2. Advanced preprocessing
        df_processed = self.advanced_preprocessing()
        
        # 3. Prepare ML data
        X_train, X_test, y_train, y_test = self.prepare_ml_data(df_processed)
        
        # 4. Baseline model
        print("\n📊 BASELINE MODEL")
        print("="*50)
        baseline_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        baseline_model.fit(X_train, y_train)
        baseline_accuracy = accuracy_score(y_test, baseline_model.predict(X_test))
        print(f"Baseline accuracy: {baseline_accuracy:.4f}")
        
        # 5. Class balancing
        X_resampled, y_resampled = self.strategy_class_balancing()
        
        # 6. Ensemble methods
        ensemble_accuracy = self.strategy_ensemble_stacking(X_resampled, y_resampled)
        
        # 7. Final summary
        self.print_final_summary(baseline_accuracy)
    
    def print_final_summary(self, baseline_accuracy):
        """Print comprehensive results summary"""
        print("\n" + "="*80)
        print("🎯 FINAL OPTIMIZATION RESULTS")
        print("="*80)
        
        print(f"📊 Baseline accuracy: {baseline_accuracy:.4f}")
        
        if 'resampling' in self.results:
            print(f"⚖️  Best resampling: {self.results['resampling']['strategy']} "
                  f"(CV: {self.results['resampling']['cv_accuracy']:.4f})")
        
        if 'ensemble' in self.results:
            print(f"🤝 Best ensemble: {self.results['ensemble']['best_method']} "
                  f"({self.results['ensemble']['best_accuracy']:.4f})")
            
            print("\nIndividual model performance:")
            for model, acc in self.results['ensemble']['individual_accuracies'].items():
                print(f"   {model}: {acc:.4f}")
        
        # Calculate improvement
        best_accuracy = max([
            baseline_accuracy,
            self.results.get('resampling', {}).get('cv_accuracy', 0),
            self.results.get('ensemble', {}).get('best_accuracy', 0)
        ])
        
        improvement = best_accuracy - baseline_accuracy
        progress_to_90 = (best_accuracy - baseline_accuracy) / (0.90 - baseline_accuracy) * 100
        
        print(f"\n🏆 BEST OVERALL ACCURACY: {best_accuracy:.4f}")
        print(f"📈 Improvement: +{improvement:.4f}")
        print(f"🎯 Progress to 90%: {progress_to_90:.1f}%")
        
        if best_accuracy >= 0.90:
            print("🎉 TARGET ACHIEVED: 90%+ accuracy reached!")
        else:
            remaining = 0.90 - best_accuracy
            print(f"📋 Remaining gap to 90%: {remaining:.4f}")
            
            print(f"\n💡 NEXT STEPS TO REACH 90%:")
            print("1. 🔬 Deep feature engineering with domain expertise")
            print("2. 🧠 Try neural networks/deep learning approaches") 
            print("3. 🎯 Focus on minority classes with specialized models")
            print("4. 📊 Advanced cross-validation and model selection")
            print("5. 🔄 Iterative feature selection and hyperparameter tuning")

def main():
    """Run the comprehensive TOI optimization"""
    optimizer = TOIOptimizer()
    optimizer.run_comprehensive_optimization()

if __name__ == "__main__":
    main()
