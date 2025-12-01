"""
QSAR Biodegradability Prediction - Main Script
================================================

This script implements a comprehensive machine learning pipeline for predicting
chemical biodegradability from QSAR features.

Author: Nazrin Atayeva
Course: ELE4448 Data Modelling and Machine Intelligence
Date: 1 December 2025

Requirements:
    - Python 3.8+
    - numpy
    - scipy
    - scikit-learn
    - matplotlib
    - seaborn
    
Usage:
    python main_run.py
    
The script will:
    1. Load and preprocess the QSAR data
    2. Perform exploratory data analysis
    3. Train multiple machine learning models
    4. Evaluate models with cross-validation
    5. Generate all figures and results for the report
"""

import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Create output directory
OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("QSAR BIODEGRADABILITY PREDICTION")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("[1/8] Loading QSAR data...")

# Load the data
data = scipy.io.loadmat('QSAR_data.mat')
X_full = data['QSAR_data']

# Separate features and labels
X = X_full[:, :-1]  # First 41 columns are features
y = X_full[:, -1]   # Last column is the label

print(f"    - Loaded {X.shape[0]} samples with {X.shape[1]} features")
print(f"    - Biodegradable samples: {int(np.sum(y == 1))} ({100*np.sum(y == 1)/len(y):.1f}%)")
print(f"    - Non-biodegradable samples: {int(np.sum(y == 0))} ({100*np.sum(y == 0)/len(y):.1f}%)")
print(f"    - Class imbalance ratio: {np.sum(y == 0)/np.sum(y == 1):.2f}:1")
print()

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("[2/8] Preprocessing data...")

# Check for duplicates
duplicates = np.sum([np.sum(np.all(X == X[i], axis=1)) for i in range(len(X))]) - len(X)
print(f"    - Duplicate samples found: {duplicates}")

# Remove duplicates if any
if duplicates > 0:
    X, unique_indices = np.unique(X, axis=0, return_index=True)
    y = y[unique_indices]
    print(f"    - After removing duplicates: {len(X)} samples")

# Check for missing values
missing = np.sum(np.isnan(X))
print(f"    - Missing values: {missing}")

# Check for outliers using IQR method
from utils_qsar import detect_outliers_iqr
outlier_mask = detect_outliers_iqr(X)
print(f"    - Outliers detected (IQR method): {np.sum(outlier_mask)} samples")
print(f"    - Keeping outliers (may contain important information)")

# Feature statistics before scaling
print(f"    - Feature range before scaling: [{np.min(X):.2f}, {np.max(X):.2f}]")
print(f"    - Feature mean before scaling: {np.mean(X):.2f}")
print(f"    - Feature std before scaling: {np.std(X):.2f}")

# Split data into train and test sets (80-20 split, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
print(f"    - Train set: {len(X_train)} samples")
print(f"    - Test set: {len(X_test)} samples")

# Apply RobustScaler (less sensitive to outliers than StandardScaler)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"    - Feature range after scaling: [{np.min(X_train_scaled):.2f}, {np.max(X_train_scaled):.2f}]")
print(f"    - Scaling method: RobustScaler (median and IQR-based)")
print()

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================================
print("[3/8] Performing exploratory data analysis...")

from utils_qsar import plot_feature_distributions, plot_correlation_matrix, plot_pca_analysis

# Plot feature distributions
plot_feature_distributions(X_train, y_train, OUTPUT_DIR)
print("    - Generated: feature_distributions.png")

# Plot correlation matrix
plot_correlation_matrix(X_train, OUTPUT_DIR)
print("    - Generated: correlation_matrix.png")

# PCA analysis
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
plot_pca_analysis(pca, X_train_pca, y_train, OUTPUT_DIR)
print("    - Generated: pca_analysis.png")

# Determine number of components to retain (95% variance)
cumvar = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cumvar >= 0.95) + 1
print(f"    - Components for 95% variance: {n_components}/{X.shape[1]}")
print()

# ============================================================================
# 4. MODEL TRAINING - BASELINE MODELS
# ============================================================================
print("[4/8] Training baseline models...")

models_baseline = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
}

results_baseline = {}
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

for name, model in models_baseline.items():
    print(f"    - Training {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                cv=cv_strategy, scoring='accuracy', n_jobs=-1)
    
    # Store results
    results_baseline[name] = {
        'model': model,
        'train_acc': accuracy_score(y_train, y_pred_train),
        'test_acc': accuracy_score(y_test, y_pred_test),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred_test': y_pred_test
    }
    
    if hasattr(model, 'predict_proba'):
        y_proba_test = model.predict_proba(X_test_scaled)[:, 1]
        results_baseline[name]['y_proba_test'] = y_proba_test
        results_baseline[name]['auc'] = roc_auc_score(y_test, y_proba_test)
    
    print(f"      Train Acc: {results_baseline[name]['train_acc']:.4f}")
    print(f"      Test Acc:  {results_baseline[name]['test_acc']:.4f}")
    print(f"      CV Acc:    {results_baseline[name]['cv_mean']:.4f} (+/- {results_baseline[name]['cv_std']:.4f})")
    if 'auc' in results_baseline[name]:
        print(f"      AUC:       {results_baseline[name]['auc']:.4f}")

print()

# ============================================================================
# 5. MODEL TRAINING - ADVANCED MODELS
# ============================================================================
print("[5/8] Training advanced models...")

models_advanced = {
    'Support Vector Machine': SVC(kernel='rbf', C=10, gamma='scale', 
                                  probability=True, random_state=RANDOM_SEED),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                    max_depth=5, random_state=RANDOM_SEED),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu',
                                   max_iter=500, random_state=RANDOM_SEED,
                                   early_stopping=True)
}

results_advanced = {}

for name, model in models_advanced.items():
    print(f"    - Training {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                cv=cv_strategy, scoring='accuracy', n_jobs=-1)
    
    # Store results
    results_advanced[name] = {
        'model': model,
        'train_acc': accuracy_score(y_train, y_pred_train),
        'test_acc': accuracy_score(y_test, y_pred_test),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred_test': y_pred_test
    }
    
    if hasattr(model, 'predict_proba'):
        y_proba_test = model.predict_proba(X_test_scaled)[:, 1]
        results_advanced[name]['y_proba_test'] = y_proba_test
        results_advanced[name]['auc'] = roc_auc_score(y_test, y_proba_test)
    
    print(f"      Train Acc: {results_advanced[name]['train_acc']:.4f}")
    print(f"      Test Acc:  {results_advanced[name]['test_acc']:.4f}")
    print(f"      CV Acc:    {results_advanced[name]['cv_mean']:.4f} (+/- {results_advanced[name]['cv_std']:.4f})")
    if 'auc' in results_advanced[name]:
        print(f"      AUC:       {results_advanced[name]['auc']:.4f}")

print()

# Combine all results
all_results = {**results_baseline, **results_advanced}

# ============================================================================
# 6. CUSTOM ENSEMBLE MODEL (Novel approach)
# ============================================================================
print("[6/8] Training custom weighted ensemble model...")

from utils_qsar import WeightedEnsembleClassifier

# Select top 3 models based on CV score
cv_scores_dict = {name: res['cv_mean'] for name, res in all_results.items()}
top_models = sorted(cv_scores_dict.items(), key=lambda x: x[1], reverse=True)[:3]
print(f"    - Top 3 models selected:")
for name, score in top_models:
    print(f"      * {name}: {score:.4f}")

# Create ensemble
# Use fresh copies of the base classifiers (not the fitted ones)
base_models = []
for name, _ in top_models:
    if name == 'Logistic Regression':
        base_models.append(LogisticRegression(random_state=RANDOM_SEED, max_iter=1000))
    elif name == 'K-Nearest Neighbors':
        base_models.append(KNeighborsClassifier(n_neighbors=5))
    elif name == 'Random Forest':
        base_models.append(RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1))
    elif name == 'Support Vector Machine':
        base_models.append(SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=RANDOM_SEED))
    elif name == 'Gradient Boosting':
        base_models.append(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=RANDOM_SEED))
    elif name == 'Neural Network':
        base_models.append(MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', max_iter=500, random_state=RANDOM_SEED, early_stopping=True))

ensemble = WeightedEnsembleClassifier(base_models)

# Train ensemble (fits the meta-learner)
print("    - Training ensemble meta-learner...")
ensemble.fit(X_train_scaled, y_train)

# Predictions
y_pred_train_ens = ensemble.predict(X_train_scaled)
y_pred_test_ens = ensemble.predict(X_test_scaled)
y_proba_test_ens = ensemble.predict_proba(X_test_scaled)[:, 1]

# Cross-validation
cv_scores_ens = cross_val_score(ensemble, X_train_scaled, y_train, 
                               cv=cv_strategy, scoring='accuracy', n_jobs=-1)

# Store results
all_results['Weighted Ensemble'] = {
    'model': ensemble,
    'train_acc': accuracy_score(y_train, y_pred_train_ens),
    'test_acc': accuracy_score(y_test, y_pred_test_ens),
    'cv_mean': cv_scores_ens.mean(),
    'cv_std': cv_scores_ens.std(),
    'y_pred_test': y_pred_test_ens,
    'y_proba_test': y_proba_test_ens,
    'auc': roc_auc_score(y_test, y_proba_test_ens)
}

print(f"    - Ensemble weights: {ensemble.weights}")
print(f"      Train Acc: {all_results['Weighted Ensemble']['train_acc']:.4f}")
print(f"      Test Acc:  {all_results['Weighted Ensemble']['test_acc']:.4f}")
print(f"      CV Acc:    {all_results['Weighted Ensemble']['cv_mean']:.4f} (+/- {all_results['Weighted Ensemble']['cv_std']:.4f})")
print(f"      AUC:       {all_results['Weighted Ensemble']['auc']:.4f}")
print()

# ============================================================================
# 7. MODEL EVALUATION AND COMPARISON
# ============================================================================
print("[7/8] Evaluating and comparing models...")

from utils_qsar import plot_model_comparison, plot_confusion_matrices, plot_roc_curves

# Compare all models
plot_model_comparison(all_results, OUTPUT_DIR)
print("    - Generated: model_comparison.png")

# Plot confusion matrices
plot_confusion_matrices(all_results, y_test, OUTPUT_DIR)
print("    - Generated: confusion_matrices.png")

# Plot ROC curves
plot_roc_curves(all_results, y_test, OUTPUT_DIR)
print("    - Generated: roc_curves.png")

# Detailed metrics for best model
best_model_name = max(all_results.items(), key=lambda x: x[1]['test_acc'])[0]
best_results = all_results[best_model_name]
y_pred_best = best_results['y_pred_test']

print(f"\n    Best Model: {best_model_name}")
print(f"    " + "="*60)
print(f"    Accuracy:  {accuracy_score(y_test, y_pred_best):.4f}")
print(f"    Precision: {precision_score(y_test, y_pred_best):.4f}")
print(f"    Recall:    {recall_score(y_test, y_pred_best):.4f}")
print(f"    F1-Score:  {f1_score(y_test, y_pred_best):.4f}")
if 'auc' in best_results:
    print(f"    AUC:       {best_results['auc']:.4f}")

print(f"\n    Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(f"    TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
print(f"    FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")

print()

# ============================================================================
# 8. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("[8/8] Analyzing feature importance...")

from utils_qsar import plot_feature_importance

# Get feature importance from Random Forest and Gradient Boosting
if 'Random Forest' in all_results:
    rf_importance = all_results['Random Forest']['model'].feature_importances_
elif 'Gradient Boosting' in all_results:
    rf_importance = all_results['Gradient Boosting']['model'].feature_importances_
else:
    rf_importance = None

if rf_importance is not None:
    plot_feature_importance(rf_importance, OUTPUT_DIR)
    print("    - Generated: feature_importance.png")
    
    # Print top 10 most important features
    top_features = np.argsort(rf_importance)[::-1][:10]
    print(f"\n    Top 10 Most Important Features:")
    for i, feat_idx in enumerate(top_features, 1):
        print(f"      {i:2d}. Feature {feat_idx:2d}: {rf_importance[feat_idx]:.4f}")

print()

# ============================================================================
# SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("="*80)
print("SUMMARY")
print("="*80)

# Create summary table
print("\nModel Performance Summary:")
print("-" * 80)
print(f"{'Model':<30} {'Train Acc':>10} {'Test Acc':>10} {'CV Acc':>10} {'AUC':>10}")
print("-" * 80)

for name, res in sorted(all_results.items(), key=lambda x: x[1]['test_acc'], reverse=True):
    auc_str = f"{res['auc']:.4f}" if 'auc' in res else "N/A"
    print(f"{name:<30} {res['train_acc']:>10.4f} {res['test_acc']:>10.4f} "
          f"{res['cv_mean']:>10.4f} {auc_str:>10}")

print("-" * 80)

# Recommendation
print("\nRECOMMENDATION:")
print(f"Based on the comprehensive evaluation, the {best_model_name} model is")
print(f"recommended for predicting chemical biodegradability.")
print(f"\nKey strengths:")
print(f"  - Highest test accuracy: {best_results['test_acc']:.4f}")
print(f"  - Robust cross-validation performance: {best_results['cv_mean']:.4f} "
      f"(+/- {best_results['cv_std']:.4f})")
print(f"  - Low overfitting (train-test gap: {best_results['train_acc'] - best_results['test_acc']:.4f})")

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print(f"\nAll results and figures saved to: {OUTPUT_DIR}")
print("="*80)