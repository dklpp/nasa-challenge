"""
Data preprocessing pipeline for TOI (TESS Objects of Interest) dataset
This module handles cleaning, feature engineering, and preparation for XGBoost training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class TOIPreprocessor:
    def __init__(self, data_path):
        """
        Initialize the preprocessor with the path to the TOI dataset
        
        Args:
            data_path (str): Path to the TOI CSV file
        """
        self.data_path = data_path
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'tfopwg_disp'  # TOI disposition (PC, FP, KP, APC)
        
    def load_data(self):
        """Load the TOI dataset"""
        print("Loading TOI dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Target distribution:\n{self.df[self.target_column].value_counts()}")
        return self.df
    
    def explore_data(self):
        """Explore the dataset to understand its structure"""
        print("\n=== DATA EXPLORATION ===")
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nColumn types:")
        print(self.df.dtypes.value_counts())
        
        print(f"\nMissing values:")
        missing_counts = self.df.isnull().sum()
        missing_pct = (missing_counts / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_counts,
            'Missing %': missing_pct
        }).sort_values('Missing %', ascending=False)
        print(missing_df[missing_df['Missing Count'] > 0].head(10))
        
        print(f"\nTarget variable distribution:")
        print(self.df[self.target_column].value_counts())
        
        # Identify numerical and categorical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        print(f"\nNumerical columns ({len(numerical_cols)}): {numerical_cols[:10]}...")
        print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
        
        return numerical_cols, categorical_cols
    
    def clean_data(self):
        """Clean the dataset by handling missing values and outliers"""
        print("\n=== DATA CLEANING ===")
        
        # Remove columns that are mostly identifiers or timestamps
        id_columns = ['toi', 'tid', 'rastr', 'decstr', 'toi_created', 'rowupdate']
        self.df = self.df.drop(columns=[col for col in id_columns if col in self.df.columns])
        print(f"Removed identifier columns: {[col for col in id_columns if col in self.df.columns]}")
        
        # Handle missing values for numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # For error columns (err1, err2), fill with 0 as they represent uncertainties
        error_cols = [col for col in numerical_cols if 'err' in col.lower()]
        for col in error_cols:
            self.df[col] = self.df[col].fillna(0)
        
        # For limit columns (lim), fill with 0 as they are flags
        limit_cols = [col for col in numerical_cols if 'lim' in col.lower()]
        for col in limit_cols:
            self.df[col] = self.df[col].fillna(0)
        
        # For main measurement columns, use median imputation
        measurement_cols = [col for col in numerical_cols 
                          if col not in error_cols and col not in limit_cols]
        for col in measurement_cols:
            if self.df[col].isnull().sum() > 0:
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
                print(f"Filled {col} missing values with median: {median_val:.4f}")
        
        print(f"Dataset shape after cleaning: {self.df.shape}")
        return self.df
    
    def feature_engineering(self):
        """Create new features from existing ones"""
        print("\n=== FEATURE ENGINEERING ===")
        
        # Create planet habitability features
        if 'pl_eqt' in self.df.columns:
            # Earth-like temperature range (200-350K)
            self.df['is_habitable_temp'] = ((self.df['pl_eqt'] >= 200) & 
                                          (self.df['pl_eqt'] <= 350)).astype(int)
        
        # Create planet size categories based on radius
        if 'pl_rade' in self.df.columns:
            self.df['planet_size_category'] = pd.cut(
                self.df['pl_rade'], 
                bins=[0, 1.25, 2.0, 4.0, float('inf')], 
                labels=['Super-Earth', 'Sub-Neptune', 'Neptune', 'Jupiter+']
            )
        
        # Create stellar magnitude brightness category
        if 'st_tmag' in self.df.columns:
            self.df['star_brightness'] = pd.cut(
                self.df['st_tmag'],
                bins=[0, 8, 10, 12, float('inf')],
                labels=['Bright', 'Medium', 'Faint', 'Very_Faint']
            )
        
        # Create orbital period categories
        if 'pl_orbper' in self.df.columns:
            self.df['orbital_period_category'] = pd.cut(
                self.df['pl_orbper'],
                bins=[0, 1, 10, 100, float('inf')],
                labels=['Ultra-short', 'Short', 'Medium', 'Long']
            )
        
        # Create insolation flux categories (Earth = 1)
        if 'pl_insol' in self.df.columns:
            self.df['insolation_category'] = pd.cut(
                self.df['pl_insol'],
                bins=[0, 0.5, 2, 10, float('inf')],
                labels=['Cold', 'Temperate', 'Hot', 'Very_Hot']
            )
        
        print("Created new categorical features for planet characteristics")
        return self.df
    
    def encode_features(self):
        """Encode categorical variables and prepare features for ML"""
        print("\n=== FEATURE ENCODING ===")
        
        # Get all columns except target
        feature_cols = [col for col in self.df.columns if col != self.target_column]
        
        # Separate numerical and categorical features
        numerical_features = self.df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = self.df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"Numerical features: {len(numerical_features)}")
        print(f"Categorical features: {len(categorical_features)}")
        
        # Encode categorical features
        for col in categorical_features:
            if col in self.df.columns:
                le = LabelEncoder()
                # Handle missing values in categorical columns
                if self.df[col].dtype.name == 'category':
                    # For categorical columns, add 'Unknown' to categories first
                    if 'Unknown' not in self.df[col].cat.categories:
                        self.df[col] = self.df[col].cat.add_categories(['Unknown'])
                    self.df[col] = self.df[col].fillna('Unknown')
                else:
                    # For object columns, directly fill na
                    self.df[col] = self.df[col].fillna('Unknown')
                
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
                print(f"Encoded {col}: {len(le.classes_)} unique values")
        
        # Prepare final feature set
        encoded_categorical = [f'{col}_encoded' for col in categorical_features if col in self.df.columns]
        self.feature_columns = numerical_features + encoded_categorical
        
        # Encode target variable
        target_le = LabelEncoder()
        self.df[f'{self.target_column}_encoded'] = target_le.fit_transform(self.df[self.target_column])
        self.label_encoders[self.target_column] = target_le
        
        print(f"Target classes: {target_le.classes_}")
        print(f"Total features for ML: {len(self.feature_columns)}")
        
        return self.feature_columns
    
    def prepare_ml_data(self, test_size=0.2, random_state=42):
        """Prepare data for machine learning"""
        print("\n=== PREPARING ML DATA ===")
        
        # Get features and target
        X = self.df[self.feature_columns]
        y = self.df[f'{self.target_column}_encoded']
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Training target distribution:\n{pd.Series(y_train).value_counts().sort_index()}")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_info(self):
        """Get information about the processed features"""
        info = {
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'label_encoders': self.label_encoders,
            'target_classes': self.label_encoders[self.target_column].classes_ if self.target_column in self.label_encoders else None
        }
        return info

def main():
    """Main preprocessing pipeline"""
    # Initialize preprocessor
    data_path = "/Users/lilianahotsko/Desktop/nasa-challenge/data/clean/TOI_2025.10.04_08.44.51.csv"
    preprocessor = TOIPreprocessor(data_path)
    
    # Load and explore data
    df = preprocessor.load_data()
    numerical_cols, categorical_cols = preprocessor.explore_data()
    
    # Clean and engineer features
    df_clean = preprocessor.clean_data()
    df_features = preprocessor.feature_engineering()
    
    # Encode features
    feature_columns = preprocessor.encode_features()
    
    # Prepare ML data
    X_train, X_test, y_train, y_test = preprocessor.prepare_ml_data()
    
    # Save processed data
    output_dir = "/Users/lilianahotsko/Desktop/nasa-challenge/ml_tabular"
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    pd.Series(y_train).to_csv(f"{output_dir}/y_train.csv", index=False, header=['target'])
    pd.Series(y_test).to_csv(f"{output_dir}/y_test.csv", index=False, header=['target'])
    
    print(f"\n=== PREPROCESSING COMPLETE ===")
    print(f"Processed data saved to {output_dir}")
    print(f"Ready for XGBoost training!")
    
    return preprocessor, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocessor, X_train, X_test, y_train, y_test = main()
