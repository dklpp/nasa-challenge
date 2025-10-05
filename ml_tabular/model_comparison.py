"""
Model Comparison Script: XGBoost vs LightGBM for TOI Classification
This script compares the performance of XGBoost and LightGBM models
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class ModelComparison:
    def __init__(self, data_dir="/Users/lilianahotsko/Desktop/nasa-challenge/ml_tabular"):
        """Initialize model comparison with data directory"""
        self.data_dir = data_dir
        self.models = {}
        self.results = {}
        
    def load_test_data(self):
        """Load the test data"""
        print("Loading test data...")
        self.X_test = pd.read_csv(f"{self.data_dir}/X_test.csv")
        self.y_test = pd.read_csv(f"{self.data_dir}/y_test.csv")['target'].values
        print(f"Test data shape: {self.X_test.shape}")
        return self.X_test, self.y_test
    
    def load_models(self):
        """Load both XGBoost and LightGBM models"""
        print("Loading trained models...")
        
        # Load XGBoost model
        try:
            with open(f"{self.data_dir}/xgboost_toi_model.pkl", 'rb') as f:
                xgb_data = pickle.load(f)
                self.models['XGBoost'] = xgb_data['model']
                print("✓ XGBoost model loaded")
        except FileNotFoundError:
            print("✗ XGBoost model not found")
        
        # Load LightGBM model
        try:
            with open(f"{self.data_dir}/lightgbm_toi_model.pkl", 'rb') as f:
                lgb_data = pickle.load(f)
                self.models['LightGBM'] = lgb_data['model']
                print("✓ LightGBM model loaded")
        except FileNotFoundError:
            print("✗ LightGBM model not found")
        
        return self.models
    
    def evaluate_models(self):
        """Evaluate both models on test data"""
        print("\n" + "="*50)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*50)
        
        for model_name, model in self.models.items():
            print(f"\n{model_name} Results:")
            print("-" * 30)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Store results
            self.results[model_name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
    
    def create_comparison_visualizations(self):
        """Create comparison visualizations"""
        print("\nCreating comparison visualizations...")
        
        # 1. Accuracy Comparison Bar Chart
        plt.figure(figsize=(10, 6))
        
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        
        colors = ['#1f77b4', '#ff7f0e']  # Blue for XGBoost, Orange for LightGBM
        bars = plt.bar(models, accuracies, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, max(accuracies) + 0.02)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Side-by-side Confusion Matrices
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            y_pred = self.results[model_name]['predictions']
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name} Confusion Matrix', fontweight='bold')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Per-class Performance Comparison
        self.plot_per_class_metrics()
    
    def plot_per_class_metrics(self):
        """Plot per-class precision, recall, and F1-score comparison"""
        # Get class labels (assuming 0-5 for the 6 TOI classes)
        class_labels = ['APC', 'CP', 'FA', 'FP', 'KP', 'PC']
        metrics = ['precision', 'recall', 'f1-score']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for metric_idx, metric in enumerate(metrics):
            model_data = []
            
            for model_name in self.models.keys():
                metric_values = []
                report = self.results[model_name]['classification_report']
                
                for class_idx in range(len(class_labels)):
                    metric_values.append(report[str(class_idx)][metric])
                
                model_data.append(metric_values)
            
            # Create grouped bar chart
            x = np.arange(len(class_labels))
            width = 0.35
            
            axes[metric_idx].bar(x - width/2, model_data[0], width, label='XGBoost', alpha=0.8)
            axes[metric_idx].bar(x + width/2, model_data[1], width, label='LightGBM', alpha=0.8)
            
            axes[metric_idx].set_xlabel('Classes')
            axes[metric_idx].set_ylabel(metric.capitalize())
            axes[metric_idx].set_title(f'{metric.capitalize()} by Class')
            axes[metric_idx].set_xticks(x)
            axes[metric_idx].set_xticklabels(class_labels)
            axes[metric_idx].legend()
            axes[metric_idx].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/per_class_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL COMPARISON SUMMARY")
        print("="*60)
        
        # Overall accuracy comparison
        print("\n1. OVERALL ACCURACY:")
        print("-" * 30)
        for model_name, results in self.results.items():
            print(f"{model_name:12}: {results['accuracy']:.4f}")
        
        # Determine winner
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        accuracy_diff = max(self.results[model]['accuracy'] for model in self.results) - \
                       min(self.results[model]['accuracy'] for model in self.results)
        
        print(f"\n🏆 Best Overall: {best_model}")
        print(f"   Accuracy difference: {accuracy_diff:.4f}")
        
        # Per-class analysis
        print("\n2. PER-CLASS PERFORMANCE:")
        print("-" * 30)
        
        class_labels = ['APC', 'CP', 'FA', 'FP', 'KP', 'PC']
        
        for class_idx, class_name in enumerate(class_labels):
            print(f"\n{class_name} (Class {class_idx}):")
            for model_name in self.models.keys():
                report = self.results[model_name]['classification_report']
                precision = report[str(class_idx)]['precision']
                recall = report[str(class_idx)]['recall']
                f1 = report[str(class_idx)]['f1-score']
                print(f"  {model_name:12}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        # Model characteristics
        print("\n3. MODEL CHARACTERISTICS:")
        print("-" * 30)
        print("XGBoost:")
        print("  + Generally more robust to overfitting")
        print("  + Better handling of missing values")
        print("  + More mature ecosystem")
        print("  - Slower training on large datasets")
        
        print("\nLightGBM:")
        print("  + Faster training speed")
        print("  + Lower memory usage")
        print("  + Good performance on categorical features")
        print("  - May overfit on small datasets")
        
        # Recommendations
        print("\n4. RECOMMENDATIONS:")
        print("-" * 30)
        if accuracy_diff < 0.01:
            print("📊 Performance is very similar between both models")
            print("🚀 Consider LightGBM for faster training/inference")
            print("🛡️  Consider XGBoost for more robust production deployment")
        else:
            print(f"🎯 {best_model} shows superior performance")
            print("💡 Consider ensemble methods combining both models")
        
        # Save summary to file
        summary_file = f"{self.data_dir}/model_comparison_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("MODEL COMPARISON SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            f.write("ACCURACY COMPARISON:\n")
            for model_name, results in self.results.items():
                f.write(f"{model_name}: {results['accuracy']:.4f}\n")
            
            f.write(f"\nBest Model: {best_model}\n")
            f.write(f"Accuracy Difference: {accuracy_diff:.4f}\n")
        
        print(f"\n📄 Summary saved to: {summary_file}")

def main():
    """Main comparison pipeline"""
    print("🔍 Starting Model Comparison Analysis...")
    
    # Initialize comparison
    comparison = ModelComparison()
    
    # Load data and models
    X_test, y_test = comparison.load_test_data()
    models = comparison.load_models()
    
    if len(models) < 2:
        print("❌ Need both XGBoost and LightGBM models to compare!")
        return
    
    # Evaluate models
    comparison.evaluate_models()
    
    # Create visualizations
    comparison.create_comparison_visualizations()
    
    # Generate summary report
    comparison.generate_summary_report()
    
    print("\n✅ Model comparison completed successfully!")
    print(f"📊 Visualizations saved in: {comparison.data_dir}")

if __name__ == "__main__":
    main()
