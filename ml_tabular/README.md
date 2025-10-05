# TOI XGBoost Classification Pipeline

This directory contains a complete machine learning pipeline for classifying TESS Objects of Interest (TOI) using XGBoost.

## Files

- `data_preprocessing.py` - Data cleaning, feature engineering, and preprocessing pipeline
- `xgboost_trainer.py` - XGBoost model training and evaluation
- `run_pipeline.py` - Complete pipeline runner
- `requirements.txt` - Python dependencies

## Dataset

The pipeline processes the TOI dataset (`data/clean/TOI_2025.10.04_08.44.51.csv`) containing:
- **7,703 observations** with 65 features
- **Target classes**: PC (Planet Candidate), FP (False Positive), CP (Confirmed Planet), KP (Known Planet), APC (Ambiguous Planet Candidate), FA (False Alarm)

## Preprocessing Pipeline

1. **Data Exploration**: Analyzed 60 numerical and 5 categorical features
2. **Data Cleaning**: 
   - Removed identifier columns (toi, tid, rastr, decstr, toi_created, rowupdate)
   - Handled missing values using median imputation for measurements, zeros for errors/limits
3. **Feature Engineering**:
   - Created habitability temperature indicator (200-350K range)
   - Planet size categories (Super-Earth, Sub-Neptune, Neptune, Jupiter+)
   - Stellar brightness categories based on TESS magnitude
   - Orbital period categories (Ultra-short, Short, Medium, Long)
   - Insolation flux categories (Cold, Temperate, Hot, Very Hot)
4. **Encoding**: Label encoding for categorical variables
5. **Final Dataset**: 63 features ready for ML training

## Model Performance

**XGBoost Classifier** with optimized hyperparameters:
- `colsample_bytree`: 0.8
- `learning_rate`: 0.1
- `max_depth`: 6
- `n_estimators`: 300
- `subsample`: 0.8

### Results:
- **Test Accuracy**: 72.81%
- **Cross-validation Mean**: 71.89% (±0.88%)
- **Best performing classes**: PC (Planet Candidate) with 91% recall, KP (Known Planet) with 77% precision

### Class Performance:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| APC   | 0.39      | 0.16   | 0.23     | 92      |
| CP    | 0.65      | 0.52   | 0.58     | 137     |
| FA    | 0.20      | 0.05   | 0.08     | 20      |
| FP    | 0.65      | 0.48   | 0.55     | 239     |
| KP    | 0.77      | 0.56   | 0.65     | 117     |
| PC    | 0.76      | 0.91   | 0.83     | 936     |

## Usage

### Run Complete Pipeline:
```bash
python run_pipeline.py
```

### Run Individual Steps:
```bash
# Preprocessing only
python data_preprocessing.py

# Training only (requires preprocessed data)
python xgboost_trainer.py
```

## Generated Files

After running the pipeline:
- `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` - Preprocessed data splits
- `xgboost_toi_model.pkl` - Trained XGBoost model
- `confusion_matrix.png` - Confusion matrix visualization
- `feature_importance.png` - Top 20 most important features

## Key Insights

The model performs best at identifying Planet Candidates (PC) with high recall (91%) and shows good precision for Known Planets (KP) at 77%. The challenging classes are False Alarms (FA) and Ambiguous Planet Candidates (APC), likely due to class imbalance and inherent difficulty in distinguishing these categories.

The most important features for classification include stellar and planetary physical properties, orbital characteristics, and the engineered categorical features based on astronomical knowledge.
