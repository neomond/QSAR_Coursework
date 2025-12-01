# QSAR Biodegradability Prediction

## Overview
This coursework implements a comprehensive machine learning pipeline for predicting chemical biodegradability from QSAR (Quantitative Structure-Activity Relationship) features.

## Project Structure

### Main Files (for submission)
1. **main_run.py** - Main script that generates all results (upload as main_run.py in ZIP file)
2. **utils_qsar.py** - Utility functions for analysis and visualization (include in ZIP file)
3. **QSAR_Report_Simple.pdf** - Complete report in IEEE-style format (upload separately as PDF)

### Generated Outputs
All figures and results are automatically saved to `/mnt/user-data/outputs/`:

#### Figures
- `pca_analysis.png` - PCA variance and 2D/3D projections
- `feature_distributions.png` - Top 10 feature distributions by class
- `correlation_matrix.png` - Feature correlation heatmap
- `model_comparison.png` - Performance metrics across all models
- `confusion_matrices.png` - Confusion matrices for all models
- `roc_curves.png` - ROC curves comparing all models
- `feature_importance.png` - Top 20 most important features

## Running the Code

### Requirements
- Python 3.8+
- NumPy
- SciPy
- scikit-learn
- matplotlib
- seaborn

### Installation
```bash
pip install numpy scipy scikit-learn matplotlib seaborn --break-system-packages
```

### Execution
```bash
python main_run.py
```

The script will:
1. Load the QSAR data from /mnt/user-data/uploads/QSAR_data.mat
2. Perform data preprocessing and exploratory analysis
3. Train 6 different machine learning models
4. Train a custom weighted ensemble classifier
5. Generate all figures and results
6. Display comprehensive performance metrics

Expected runtime: ~20-30 seconds

## Results Summary

### Best Performing Models
1. **Gradient Boosting**: 88.15% test accuracy, 0.9274 AUC (RECOMMENDED)
2. **Weighted Ensemble**: 88.15% test accuracy, 0.9373 AUC
3. **Random Forest**: 87.20% test accuracy, 0.9366 AUC

### Key Findings
- Dataset: 1,052 unique chemicals with 41 QSAR features
- Class imbalance: 33.7% biodegradable, 66.3% non-biodegradable
- PCA shows 95% variance captured in 8 components
- RobustScaler used for preprocessing (less sensitive to outliers)
- 5-fold stratified cross-validation for model selection
- Feature 35, 26, and 38 are most important for prediction

### Novel Contributions
1. **Custom Weighted Ensemble**: Meta-learning approach that learns optimal weights for combining base models
2. **RobustScaler**: Used instead of StandardScaler due to high outlier rate
3. **Comprehensive Analysis**: Multiple visualization and evaluation metrics

## Implementation Details

### Data Processing
- Duplicate removal (6 samples)
- Outlier detection using IQR method (retained for diversity)
- 80-20 stratified train-test split
- RobustScaler normalization

### Models Implemented
1. **Logistic Regression**: L2 regularization, baseline model
2. **K-Nearest Neighbors**: k=5, instance-based learning
3. **Random Forest**: 100 trees, bagging ensemble
4. **Support Vector Machine**: RBF kernel, C=10
5. **Gradient Boosting**: 100 trees, sequential boosting
6. **Neural Network**: 2 hidden layers (100, 50), early stopping
7. **Weighted Ensemble**: Novel meta-learning approach (custom implementation)

### Overfitting Prevention
- Stratified cross-validation (5 folds)
- Early stopping for neural networks
- Regularization (SVM, Logistic Regression)
- Limited tree depth (Gradient Boosting)
- Ensemble methods

### Evaluation Metrics
- Accuracy (train, test, cross-validation)
- Precision, Recall, F1-Score
- AUC-ROC
- Confusion matrices

## Code Structure

### main_run.py
```
1. Data Loading (QSAR_data.mat)
2. Data Preprocessing (cleaning, scaling, splitting)
3. Exploratory Data Analysis (PCA, distributions, correlations)
4. Baseline Models (Logistic Regression, KNN, Random Forest)
5. Advanced Models (SVM, Gradient Boosting, Neural Network)
6. Custom Ensemble (weighted combination with meta-learning)
7. Model Evaluation (metrics, visualizations, comparisons)
8. Feature Importance Analysis
```

### utils_qsar.py
Contains helper functions for:
- Outlier detection (IQR method)
- Visualizations (PCA, distributions, correlations, confusion matrices, ROC curves)
- Custom WeightedEnsembleClassifier class

## Report Structure (QSAR_Report_Simple.pdf)

Following the required format:
1. **Abstract**: Brief overview and results
2. **Introduction**: Problem statement, ethics, related work
3. **Data Processing**: Cleaning, scaling, splitting, EDA
4. **Methodology**: Model descriptions and overfitting prevention
5. **Model Analysis**: Performance metrics, comparisons, feature importance
6. **Conclusion**: Recommendations and future work
7. **References**: Mansouri et al. (2013) dataset paper

## Submission Checklist

For Blackboard submission:

### PDF File (separate upload)
- [ ] QSAR_Report_Simple.pdf (3 pages, IEEE-style format)

### ZIP File (single upload)
- [ ] main_run.py
- [ ] utils_qsar.py
- [ ] Comments at top of main_run.py explaining requirements

**Important Notes:**
- Do NOT include QSAR_data.mat in ZIP (it's already available to markers)
- Code assumes data is at: /mnt/user-data/uploads/QSAR_data.mat
- All outputs save to: /mnt/user-data/outputs/
- main_run.py generates all results from the report

## Marker Instructions

To reproduce all results:
```bash
python main_run.py
```

All figures will be saved to /mnt/user-data/outputs/ and match those in the report.

## Contact
[Nazrin Atayeva]
[natayeva1@sheffield.ac.uk]
University of Sheffield
Department of Automatic Control and Systems Engineering

## License
This code is provided for educational purposes as part of ELE4448 coursework.
