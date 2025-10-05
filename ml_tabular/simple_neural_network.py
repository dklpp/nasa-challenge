"""
Simple Neural Network for TOI Classification
Using scikit-learn's MLPClassifier to avoid TensorFlow issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class SimpleNeuralNetworkTOI:
    def __init__(self, data_path="/Users/lilianahotsko/Desktop/nasa-challenge/data/clean/TOI_2025.10.04_08.44.51.csv"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoder = None
        self.feature_selector = None
        
    def load_focused_data(self):
        """Load and prepare focused 4-class dataset"""
        print("🎯 LOADING FOCUSED DATA FOR NEURAL NETWORK")
        print("="*60)
        
        df = pd.read_csv(self.data_path)
        
        # Filter to focused classes
        classes_to_keep = ['CP', 'FA', 'FP', 'KP']
        df_filtered = df[df['tfopwg_disp'].isin(classes_to_keep)].copy()
        
        print(f"Filtered dataset: {df_filtered.shape}")
        print(f"Classes: {classes_to_keep}")
        
        # Show class distribution
        class_dist = df_filtered['tfopwg_disp'].value_counts()
        for class_name, count in class_dist.items():
            percentage = (count / len(df_filtered)) * 100
            print(f"  {class_name}: {count:4d} ({percentage:5.1f}%)")
        
        imbalance_ratio = class_dist.max() / class_dist.min()
        print(f"Imbalance ratio: {imbalance_ratio:.1f}:1")
        
        return df_filtered
    
    def preprocess_for_neural_network(self, df):
        """Preprocess data specifically for neural networks"""
        print(f"\n🔧 NEURAL NETWORK PREPROCESSING")
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
        
        # Neural network friendly feature engineering
        print("Creating neural network features...")
        
        # Log transforms for skewed features
        skewed_features = ['pl_rade', 'pl_orbper', 'pl_insol', 'st_dist']
        for feat in skewed_features:
            if feat in df.columns:
                df[f'{feat}_log'] = np.log10(df[feat] + 1e-8)
        
        # Normalized features
        if 'pl_rade' in df.columns and 'pl_orbper' in df.columns:
            df['density_proxy'] = df['pl_rade'] / (df['pl_orbper'] ** (2/3))
            
        if 'pl_insol' in df.columns:
            df['habitable_zone'] = ((df['pl_insol'] >= 0.5) & (df['pl_insol'] <= 2.0)).astype(int)
            
        if 'pl_eqt' in df.columns:
            df['earth_like_temp'] = ((df['pl_eqt'] >= 200) & (df['pl_eqt'] <= 350)).astype(int)
        
        if 'st_teff' in df.columns and 'st_rad' in df.columns:
            df['stellar_luminosity'] = (df['st_rad'] ** 2) * ((df['st_teff'] / 5778) ** 4)
        
        print(f"Features after engineering: {df.shape[1]}")
        return df
    
    def prepare_data(self, df):
        """Prepare data for neural network training"""
        print(f"\n📊 PREPARING DATA FOR NEURAL NETWORK")
        print("="*60)
        
        # Encode target and features
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['tfopwg_disp'])
        X = df.drop(['tfopwg_disp'], axis=1)
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le_col = LabelEncoder()
            X[col] = le_col.fit_transform(X[col].astype(str))
        
        X = X.fillna(X.median())
        
        print(f"Feature matrix: {X.shape}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        # Feature selection for neural network
        self.feature_selector = SelectKBest(f_classif, k=min(20, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        print(f"Selected {X_selected.shape[1]} features for neural network")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features (crucial for neural networks)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Train: {X_train_scaled.shape}")
        print(f"Test: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_neural_networks(self, X_train, X_test, y_train, y_test):
        """Train different neural network architectures"""
        print(f"\n🧠 TRAINING NEURAL NETWORKS")
        print("="*60)
        
        # Define different NN architectures to test
        architectures = {
            'Small NN': {
                'hidden_layer_sizes': (32, 16),
                'alpha': 0.01,
                'learning_rate_init': 0.01,
                'max_iter': 500
            },
            'Medium NN': {
                'hidden_layer_sizes': (64, 32, 16),
                'alpha': 0.001,
                'learning_rate_init': 0.001,
                'max_iter': 800
            },
            'Large NN': {
                'hidden_layer_sizes': (128, 64, 32, 16),
                'alpha': 0.0001,
                'learning_rate_init': 0.001,
                'max_iter': 1000
            },
            'Deep NN': {
                'hidden_layer_sizes': (100, 50, 25, 12, 6),
                'alpha': 0.001,
                'learning_rate_init': 0.0001,
                'max_iter': 1200
            }
        }
        
        models = {}
        results = {}
        
        for arch_name, params in architectures.items():
            print(f"\nTraining {arch_name}...")
            
            # Create model with regularization
            model = MLPClassifier(
                hidden_layer_sizes=params['hidden_layer_sizes'],
                activation='relu',
                solver='adam',
                alpha=params['alpha'],  # L2 regularization
                learning_rate='adaptive',
                learning_rate_init=params['learning_rate_init'],
                max_iter=params['max_iter'],
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            test_f1 = f1_score(y_test, test_pred, average='weighted')
            overfitting_gap = train_acc - test_acc
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            models[arch_name] = model
            results[arch_name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'test_f1': test_f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'overfitting_gap': overfitting_gap,
                'n_iter': model.n_iter_,
                'loss_curve': model.loss_curve_ if hasattr(model, 'loss_curve_') else None,
                'predictions': test_pred
            }
            
            print(f"  Architecture: {params['hidden_layer_sizes']}")
            print(f"  Training accuracy: {train_acc:.4f}")
            print(f"  Test accuracy: {test_acc:.4f}")
            print(f"  CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"  Overfitting gap: {overfitting_gap:.4f}")
            print(f"  Training iterations: {model.n_iter_}")
            
            if overfitting_gap < 0.05:
                print("  ✅ Good generalization")
            elif overfitting_gap < 0.10:
                print("  ⚠️ Mild overfitting")
            else:
                print("  🚨 Severe overfitting")
        
        # Select best model
        best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
        best_result = results[best_model_name]
        
        print(f"\n🏆 Best Neural Network: {best_model_name}")
        print(f"Test Accuracy: {best_result['test_accuracy']:.4f}")
        
        return models, results, best_model_name
    
    def plot_neural_network_analysis(self, results, y_test):
        """Plot comprehensive neural network analysis"""
        print(f"\n📈 PLOTTING NEURAL NETWORK ANALYSIS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Neural Network Analysis - Focused TOI Classification', fontsize=16, fontweight='bold')
        
        # 1. Architecture Comparison
        arch_names = list(results.keys())
        test_accuracies = [results[name]['test_accuracy'] for name in arch_names]
        cv_accuracies = [results[name]['cv_mean'] for name in arch_names]
        
        x = np.arange(len(arch_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, test_accuracies, width, label='Test Accuracy', alpha=0.8)
        axes[0, 0].bar(x + width/2, cv_accuracies, width, label='CV Accuracy', alpha=0.8)
        axes[0, 0].set_xlabel('Neural Network Architecture')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Architecture Performance Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(arch_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Overfitting Analysis
        overfitting_gaps = [results[name]['overfitting_gap'] for name in arch_names]
        colors = ['lightgreen' if gap < 0.05 else 'yellow' if gap < 0.10 else 'lightcoral' 
                 for gap in overfitting_gaps]
        
        bars = axes[0, 1].bar(arch_names, overfitting_gaps, color=colors, alpha=0.7)
        axes[0, 1].set_xlabel('Architecture')
        axes[0, 1].set_ylabel('Overfitting Gap (Train - Test)')
        axes[0, 1].set_title('Overfitting Analysis')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Good threshold')
        axes[0, 1].axhline(y=0.10, color='red', linestyle='--', alpha=0.7, label='Overfitting threshold')
        axes[0, 1].legend()
        
        # Add value labels
        for bar, gap in zip(bars, overfitting_gaps):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                          f'{gap:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Training Convergence
        best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
        best_result = results[best_model_name]
        
        if best_result['loss_curve'] is not None:
            epochs = range(1, len(best_result['loss_curve']) + 1)
            axes[1, 0].plot(epochs, best_result['loss_curve'], 'b-', linewidth=2)
            axes[1, 0].set_xlabel('Epochs')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title(f'{best_model_name} - Training Loss Curve')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Training curve not available', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Training Curve')
        
        # 4. Performance Summary
        summary_text = f"""Neural Network Results Summary:

Best Architecture: {best_model_name}
Test Accuracy: {best_result['test_accuracy']:.4f}
Cross-validation: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}
F1-Score: {best_result['test_f1']:.4f}
Overfitting Gap: {best_result['overfitting_gap']:.4f}
Training Iterations: {best_result['n_iter']}

Comparison with XGBoost:
XGBoost (focused): 0.7451
Neural Network:    {best_result['test_accuracy']:.4f}
Difference:        {best_result['test_accuracy'] - 0.7451:+.4f}

Status: {'🎉 NN Superior!' if best_result['test_accuracy'] > 0.7451 else '📊 Similar Performance' if abs(best_result['test_accuracy'] - 0.7451) < 0.02 else '📉 XGBoost Better'}"""
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Performance Summary')
        
        plt.tight_layout()
        plt.savefig('/Users/lilianahotsko/Desktop/nasa-challenge/ml_tabular/neural_network_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def detailed_evaluation(self, models, results, X_test, y_test):
        """Detailed evaluation of the best neural network"""
        print(f"\n📊 DETAILED NEURAL NETWORK EVALUATION")
        print("="*60)
        
        # Best model
        best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
        best_model = models[best_model_name]
        best_result = results[best_model_name]
        best_pred = best_result['predictions']
        
        print(f"🏆 Best Model: {best_model_name}")
        print(f"Architecture: {best_model.hidden_layer_sizes}")
        print(f"Test Accuracy: {best_result['test_accuracy']:.4f}")
        print(f"F1-Score: {best_result['test_f1']:.4f}")
        
        # Classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, best_pred, target_names=self.label_encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, best_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=self.label_encoder.classes_, 
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'Neural Network - Confusion Matrix\n{best_model_name}: {best_result["test_accuracy"]:.4f}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('/Users/lilianahotsko/Desktop/nasa-challenge/ml_tabular/neural_network_confusion_matrix.png',
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return best_model_name, best_result['test_accuracy']
    
    def hyperparameter_optimization(self, X_train, y_train):
        """Optimize neural network hyperparameters"""
        print(f"\n🎛️ NEURAL NETWORK HYPERPARAMETER OPTIMIZATION")
        print("="*60)
        
        # Test different alpha values (regularization)
        alpha_range = np.logspace(-4, -1, 10)
        
        print("Testing regularization strength...")
        train_scores, val_scores = validation_curve(
            MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=500),
            X_train, y_train, param_name='alpha', param_range=alpha_range,
            cv=3, scoring='accuracy'
        )
        
        best_alpha_idx = np.argmax(val_scores.mean(axis=1))
        best_alpha = alpha_range[best_alpha_idx]
        best_val_score = val_scores.mean(axis=1)[best_alpha_idx]
        
        print(f"Best alpha: {best_alpha:.6f}")
        print(f"Best CV score: {best_val_score:.4f}")
        
        # Plot validation curve
        plt.figure(figsize=(10, 6))
        plt.semilogx(alpha_range, train_scores.mean(axis=1), 'b-', label='Training Score', linewidth=2)
        plt.fill_between(alpha_range, train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color='blue')
        
        plt.semilogx(alpha_range, val_scores.mean(axis=1), 'r-', label='Validation Score', linewidth=2)
        plt.fill_between(alpha_range, val_scores.mean(axis=1) - val_scores.std(axis=1),
                        val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1, color='red')
        
        plt.xlabel('Alpha (Regularization)')
        plt.ylabel('Accuracy')
        plt.title('Neural Network Validation Curve - Alpha Optimization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('/Users/lilianahotsko/Desktop/nasa-challenge/ml_tabular/nn_validation_curve.png',
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return best_alpha, best_val_score
    
    def run_neural_network_pipeline(self):
        """Run complete neural network pipeline"""
        print("🧠 NEURAL NETWORK TOI CLASSIFICATION PIPELINE")
        print("="*80)
        
        # Step 1: Load focused data
        df = self.load_focused_data()
        
        # Step 2: Preprocess for neural networks
        df_processed = self.preprocess_for_neural_network(df)
        
        # Step 3: Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df_processed)
        
        # Step 4: Hyperparameter optimization
        best_alpha, best_cv_score = self.hyperparameter_optimization(X_train, y_train)
        
        # Step 5: Train neural networks
        models, results, best_model_name = self.train_neural_networks(X_train, X_test, y_train, y_test)
        
        # Step 6: Plot analysis
        self.plot_neural_network_analysis(results, y_test)
        
        # Step 7: Detailed evaluation
        final_best_model, final_accuracy = self.detailed_evaluation(models, results, X_test, y_test)
        
        # Step 8: Final comparison
        print(f"\n🔄 FINAL COMPARISON")
        print("="*60)
        
        xgboost_accuracy = 0.7451
        improvement = final_accuracy - xgboost_accuracy
        
        print(f"XGBoost (focused):     {xgboost_accuracy:.4f}")
        print(f"Neural Network:        {final_accuracy:.4f}")
        print(f"Improvement:           {improvement:+.4f}")
        
        if improvement > 0.02:
            print("🎉 Neural Network shows significant improvement!")
        elif improvement > 0.005:
            print("✅ Neural Network shows modest improvement")
        else:
            print("📊 Similar performance to XGBoost")
        
        print(f"\n🎉 NEURAL NETWORK PIPELINE COMPLETE!")
        print(f"Best Architecture: {final_best_model}")
        print(f"Final Accuracy: {final_accuracy:.4f}")
        
        return {
            'best_model': final_best_model,
            'best_accuracy': final_accuracy,
            'models': models,
            'results': results,
            'best_alpha': best_alpha
        }

def main():
    """Run neural network classification"""
    print("🧠 Starting Neural Network TOI Classification")
    
    classifier = SimpleNeuralNetworkTOI()
    result = classifier.run_neural_network_pipeline()
    
    return classifier, result

if __name__ == "__main__":
    classifier, result = main()
