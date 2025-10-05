"""
Simple CNN for TOI Classification
Applying CNN to tabular data for the focused 4-class problem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class CNNTOIClassifier:
    def __init__(self, data_path="/Users/lilianahotsko/Desktop/nasa-challenge/data/clean/TOI_2025.10.04_08.44.51.csv"):
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = None
        self.history = None
        
    def load_and_prepare_focused_data(self):
        """Load and prepare the focused 4-class dataset"""
        print("🎯 LOADING FOCUSED DATA FOR CNN")
        print("="*60)
        
        df = pd.read_csv(self.data_path)
        
        # Filter to focused classes only
        classes_to_keep = ['CP', 'FA', 'FP', 'KP']
        df_filtered = df[df['tfopwg_disp'].isin(classes_to_keep)].copy()
        
        print(f"Filtered dataset: {df_filtered.shape}")
        print(f"Classes: {classes_to_keep}")
        
        # Remove identifiers
        id_cols = ['toi', 'tid', 'rastr', 'decstr', 'toi_created', 'rowupdate']
        df_filtered = df_filtered.drop(columns=[col for col in id_cols if col in df_filtered.columns])
        
        # Handle missing values
        error_cols = [col for col in df_filtered.columns if 'err' in col.lower()]
        limit_cols = [col for col in df_filtered.columns if 'lim' in col.lower()]
        
        for col in error_cols + limit_cols:
            df_filtered[col] = df_filtered[col].fillna(0)
        
        numerical_cols = df_filtered.select_dtypes(include=[np.number]).columns
        measurement_cols = [col for col in numerical_cols 
                          if col not in error_cols and col not in limit_cols]
        
        for col in measurement_cols:
            if df_filtered[col].isnull().sum() > 0:
                df_filtered[col] = df_filtered[col].fillna(df_filtered[col].median())
        
        # Essential feature engineering for CNN
        if 'pl_rade' in df_filtered.columns and 'pl_orbper' in df_filtered.columns:
            df_filtered['density_proxy'] = df_filtered['pl_rade'] / (df_filtered['pl_orbper'] ** (2/3))
            
        if 'pl_insol' in df_filtered.columns:
            df_filtered['habitable'] = ((df_filtered['pl_insol'] >= 0.5) & 
                                      (df_filtered['pl_insol'] <= 2.0)).astype(int)
            df_filtered['insol_log'] = np.log10(df_filtered['pl_insol'] + 1e-8)
        
        if 'pl_eqt' in df_filtered.columns:
            df_filtered['earth_like_temp'] = ((df_filtered['pl_eqt'] >= 200) & 
                                            (df_filtered['pl_eqt'] <= 350)).astype(int)
            df_filtered['temp_log'] = np.log10(df_filtered['pl_eqt'] + 1e-8)
        
        if 'st_teff' in df_filtered.columns and 'st_rad' in df_filtered.columns:
            df_filtered['stellar_luminosity'] = (df_filtered['st_rad'] ** 2) * ((df_filtered['st_teff'] / 5778) ** 4)
        
        print(f"Features after engineering: {df_filtered.shape[1]}")
        
        return df_filtered
    
    def prepare_data_for_cnn(self, df):
        """Prepare data specifically for CNN architecture"""
        print(f"\n🔧 PREPARING DATA FOR CNN")
        print("="*60)
        
        # Encode target
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
        print(f"Class distribution: {dict(zip(self.label_encoder.classes_, np.bincount(y)))}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features for CNN
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to categorical for CNN
        n_classes = len(self.label_encoder.classes_)
        y_train_cat = keras.utils.to_categorical(y_train, n_classes)
        y_test_cat = keras.utils.to_categorical(y_test, n_classes)
        
        print(f"Train: {X_train_scaled.shape}")
        print(f"Test: {X_test_scaled.shape}")
        print(f"Number of classes: {n_classes}")
        
        return X_train_scaled, X_test_scaled, y_train_cat, y_test_cat, y_train, y_test
    
    def reshape_for_cnn(self, X_train, X_test):
        """Reshape tabular data for CNN input"""
        print(f"\n📐 RESHAPING DATA FOR CNN")
        print("="*60)
        
        n_features = X_train.shape[1]
        
        # Strategy 1: Treat as 1D signal
        # Reshape to (samples, features, 1) for 1D CNN
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        print(f"Reshaped for 1D CNN:")
        print(f"  Train: {X_train_reshaped.shape}")
        print(f"  Test: {X_test_reshaped.shape}")
        
        return X_train_reshaped, X_test_reshaped
    
    def create_simple_cnn(self, input_shape, n_classes):
        """Create a simple CNN architecture for tabular data"""
        print(f"\n🏗️ CREATING SIMPLE CNN ARCHITECTURE")
        print("="*60)
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # First Conv1D block
            layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second Conv1D block
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            # Third Conv1D block
            layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Global pooling and dense layers
            layers.GlobalAveragePooling1D(),
            
            # Dense layers with dropout for regularization
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(n_classes, activation='softmax')
        ])
        
        # Compile with appropriate settings
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("CNN Architecture:")
        model.summary()
        
        return model
    
    def train_cnn(self, model, X_train, X_test, y_train, y_test):
        """Train the CNN with proper callbacks"""
        print(f"\n🚀 TRAINING CNN")
        print("="*60)
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train the model
        print("Starting CNN training...")
        self.history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return model
    
    def evaluate_cnn(self, model, X_test, y_test_cat, y_test_orig):
        """Comprehensive evaluation of the CNN"""
        print(f"\n📊 CNN EVALUATION")
        print("="*60)
        
        # Predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Accuracy
        accuracy = accuracy_score(y_test_orig, y_pred)
        print(f"🎯 Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test_orig, y_pred, target_names=self.label_encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(y_test_orig, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_, 
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'CNN TOI Classification - Confusion Matrix\nAccuracy: {accuracy:.4f}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('/Users/lilianahotsko/Desktop/nasa-challenge/ml_tabular/cnn_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy, y_pred
    
    def plot_training_curves(self):
        """Plot CNN training curves"""
        print(f"\n📈 PLOTTING TRAINING CURVES")
        print("="*60)
        
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('CNN Training & Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot training & validation loss
        ax2.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('CNN Training & Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/lilianahotsko/Desktop/nasa-challenge/ml_tabular/cnn_training_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Training analysis
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        best_val_acc = max(self.history.history['val_accuracy'])
        
        print(f"Training Analysis:")
        print(f"  Final training accuracy: {final_train_acc:.4f}")
        print(f"  Final validation accuracy: {final_val_acc:.4f}")
        print(f"  Best validation accuracy: {best_val_acc:.4f}")
        print(f"  Overfitting gap: {final_train_acc - final_val_acc:.4f}")
        
        if final_train_acc - final_val_acc < 0.05:
            print("  ✅ Good generalization")
        elif final_train_acc - final_val_acc < 0.10:
            print("  ⚠️ Mild overfitting")
        else:
            print("  🚨 Significant overfitting")
    
    def compare_with_traditional_ml(self, cnn_accuracy):
        """Compare CNN results with traditional ML"""
        print(f"\n🔄 COMPARISON WITH TRADITIONAL ML")
        print("="*60)
        
        # Results from previous focused classification
        xgboost_accuracy = 0.7451  # Best from focused classification
        improvement = cnn_accuracy - xgboost_accuracy
        
        print(f"XGBoost (focused): {xgboost_accuracy:.4f}")
        print(f"CNN (focused):     {cnn_accuracy:.4f}")
        print(f"Improvement:       {improvement:+.4f}")
        
        if improvement > 0.02:
            print("🎉 CNN shows significant improvement!")
        elif improvement > 0.005:
            print("✅ CNN shows modest improvement")
        elif improvement > -0.005:
            print("📊 Similar performance")
        else:
            print("📉 Traditional ML performs better")
        
        print(f"\n💡 CNN INSIGHTS:")
        print(f"• Deep learning can capture complex feature interactions")
        print(f"• Regularization (dropout, batch norm) helps generalization")
        print(f"• 1D convolutions work well for tabular data patterns")
        print(f"• Early stopping prevents overfitting")
    
    def run_cnn_pipeline(self):
        """Run the complete CNN pipeline"""
        print("🧠 CNN TOI CLASSIFICATION PIPELINE")
        print("="*80)
        
        # Step 1: Load and prepare data
        df = self.load_and_prepare_focused_data()
        
        # Step 2: Prepare for CNN
        X_train, X_test, y_train_cat, y_test_cat, y_train_orig, y_test_orig = self.prepare_data_for_cnn(df)
        
        # Step 3: Reshape for CNN
        X_train_reshaped, X_test_reshaped = self.reshape_for_cnn(X_train, X_test)
        
        # Step 4: Create CNN model
        input_shape = X_train_reshaped.shape[1:]
        n_classes = len(self.label_encoder.classes_)
        self.model = self.create_simple_cnn(input_shape, n_classes)
        
        # Step 5: Train CNN
        self.model = self.train_cnn(self.model, X_train_reshaped, X_test_reshaped, 
                                   y_train_cat, y_test_cat)
        
        # Step 6: Evaluate CNN
        accuracy, predictions = self.evaluate_cnn(self.model, X_test_reshaped, 
                                                 y_test_cat, y_test_orig)
        
        # Step 7: Plot training curves
        self.plot_training_curves()
        
        # Step 8: Compare with traditional ML
        self.compare_with_traditional_ml(accuracy)
        
        print(f"\n🎉 CNN PIPELINE COMPLETE!")
        print(f"Final CNN Accuracy: {accuracy:.4f}")
        
        return {
            'model': self.model,
            'accuracy': accuracy,
            'history': self.history,
            'classes': self.label_encoder.classes_
        }

def main():
    """Run CNN classification"""
    print("🧠 Starting CNN TOI Classification")
    
    # Check TensorFlow/GPU availability
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    
    classifier = CNNTOIClassifier()
    result = classifier.run_cnn_pipeline()
    
    return classifier, result

if __name__ == "__main__":
    classifier, result = main()
