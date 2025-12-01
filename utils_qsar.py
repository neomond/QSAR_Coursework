"""
Utility Functions for QSAR Biodegradability Prediction
========================================================

This module contains helper functions for data processing, visualization,
and custom machine learning implementations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from matplotlib.patches import Rectangle

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def detect_outliers_iqr(X, threshold=1.5):
    """
    Detect outliers using the IQR (Interquartile Range) method.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    threshold : float, default=1.5
        Multiplier for IQR to define outlier boundaries
        
    Returns:
    --------
    outlier_mask : array-like, shape (n_samples,)
        Boolean mask indicating outlier samples
    """
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = np.any((X < lower_bound) | (X > upper_bound), axis=1)
    return outliers


def calculate_feature_stats(X, y):
    """
    Calculate statistical measures for each feature separated by class.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input features
    y : array-like, shape (n_samples,)
        Target labels
        
    Returns:
    --------
    stats : dict
        Dictionary containing statistics for each class
    """
    stats = {
        'class_0': {
            'mean': np.mean(X[y == 0], axis=0),
            'std': np.std(X[y == 0], axis=0),
            'median': np.median(X[y == 0], axis=0)
        },
        'class_1': {
            'mean': np.mean(X[y == 1], axis=0),
            'std': np.std(X[y == 1], axis=0),
            'median': np.median(X[y == 1], axis=0)
        }
    }
    return stats


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_feature_distributions(X, y, output_dir, n_features=10):
    """
    Plot distributions of top features with highest variance.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input features
    y : array-like, shape (n_samples,)
        Target labels
    output_dir : str
        Directory to save the plot
    n_features : int, default=10
        Number of features to plot
    """
    # Select features with highest variance
    variances = np.var(X, axis=0)
    top_indices = np.argsort(variances)[::-1][:n_features]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, feat_idx in enumerate(top_indices):
        ax = axes[i]
        
        # Plot histograms for each class
        ax.hist(X[y == 0, feat_idx], bins=30, alpha=0.6, label='Non-biodegradable', 
                color='coral', density=True)
        ax.hist(X[y == 1, feat_idx], bins=30, alpha=0.6, label='Biodegradable', 
                color='skyblue', density=True)
        
        ax.set_xlabel(f'Feature {feat_idx}', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'Feature {feat_idx} (Var: {variances[feat_idx]:.2f})', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_correlation_matrix(X, output_dir, n_features=20):
    """
    Plot correlation matrix for top features.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input features
    output_dir : str
        Directory to save the plot
    n_features : int, default=20
        Number of features to include
    """
    # Select features with highest variance
    variances = np.var(X, axis=0)
    top_indices = np.argsort(variances)[::-1][:n_features]
    X_subset = X[:, top_indices]
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X_subset.T)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20, fontsize=12)
    
    # Set ticks and labels
    ax.set_xticks(range(len(top_indices)))
    ax.set_yticks(range(len(top_indices)))
    ax.set_xticklabels([f'F{i}' for i in top_indices], rotation=45, ha='right')
    ax.set_yticklabels([f'F{i}' for i in top_indices])
    
    ax.set_title('Feature Correlation Matrix (Top 20 Features)', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_pca_analysis(pca, X_pca, y, output_dir):
    """
    Plot PCA analysis including variance explained and 2D projection.
    
    Parameters:
    -----------
    pca : PCA object
        Fitted PCA object
    X_pca : array-like
        Transformed data
    y : array-like
        Target labels
    output_dir : str
        Directory to save the plot
    """
    fig = plt.figure(figsize=(16, 6))
    
    # Plot 1: Explained variance
    ax1 = plt.subplot(131)
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    x = range(1, len(explained_var) + 1)
    ax1.bar(x[:20], explained_var[:20], alpha=0.6, color='steelblue', label='Individual')
    ax1.plot(x[:20], cumulative_var[:20], 'ro-', label='Cumulative', linewidth=2)
    ax1.axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('PCA Explained Variance', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: 2D PCA projection
    ax2 = plt.subplot(132)
    scatter1 = ax2.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], 
                          c='coral', alpha=0.6, s=30, label='Non-biodegradable')
    scatter2 = ax2.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], 
                          c='skyblue', alpha=0.6, s=30, label='Biodegradable')
    ax2.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontsize=12)
    ax2.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontsize=12)
    ax2.set_title('2D PCA Projection', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: 3D PCA projection
    ax3 = plt.subplot(133, projection='3d')
    ax3.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], X_pca[y == 0, 2],
               c='coral', alpha=0.6, s=20, label='Non-biodegradable')
    ax3.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], X_pca[y == 1, 2],
               c='skyblue', alpha=0.6, s=20, label='Biodegradable')
    ax3.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontsize=10)
    ax3.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontsize=10)
    ax3.set_zlabel(f'PC3 ({explained_var[2]*100:.1f}%)', fontsize=10)
    ax3.set_title('3D PCA Projection', fontsize=14)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_comparison(results, output_dir):
    """
    Plot comparison of model performance metrics.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results
    output_dir : str
        Directory to save the plot
    """
    models = list(results.keys())
    train_acc = [results[m]['train_acc'] for m in models]
    test_acc = [results[m]['test_acc'] for m in models]
    cv_mean = [results[m]['cv_mean'] for m in models]
    cv_std = [results[m]['cv_std'] for m in models]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Accuracy comparison
    ax1 = axes[0]
    x = np.arange(len(models))
    width = 0.25
    
    ax1.bar(x - width, train_acc, width, label='Train', alpha=0.8, color='skyblue')
    ax1.bar(x, test_acc, width, label='Test', alpha=0.8, color='coral')
    ax1.bar(x + width, cv_mean, width, label='CV Mean', alpha=0.8, color='lightgreen')
    
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Performance Comparison', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0.7, 1.0])
    
    # Plot 2: CV scores with error bars
    ax2 = axes[1]
    ax2.errorbar(x, cv_mean, yerr=cv_std, fmt='o-', markersize=8, 
                linewidth=2, capsize=5, capthick=2)
    ax2.set_ylabel('Cross-Validation Accuracy', fontsize=12)
    ax2.set_title('Cross-Validation Performance', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.7, 1.0])
    
    # Add horizontal line for best performance
    best_cv = max(cv_mean)
    ax2.axhline(y=best_cv, color='r', linestyle='--', alpha=0.5, 
               label=f'Best: {best_cv:.4f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(results, y_test, output_dir, models_subset=None):
    """
    Plot confusion matrices for all models.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results
    y_test : array-like
        True labels
    output_dir : str
        Directory to save the plot
    models_subset : list, optional
        Subset of models to plot
    """
    if models_subset is None:
        models = list(results.keys())
    else:
        models = models_subset
    
    n_models = len(models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, model_name in enumerate(models):
        ax = axes[idx]
        y_pred = results[model_name]['y_pred_test']
        cm = confusion_matrix(y_test, y_pred)
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Non-bio', 'Bio'])
        ax.set_yticklabels(['Non-bio', 'Bio'])
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
        ax.set_title(f'{model_name}\nAcc: {results[model_name]["test_acc"]:.4f}', 
                    fontsize=11)
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.2f})',
                             ha="center", va="center", color="black", fontsize=10)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide empty subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(results, y_test, output_dir):
    """
    Plot ROC curves for all models with probability predictions.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results
    y_test : array-like
        True labels
    output_dir : str
        Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
    
    for idx, (model_name, res) in enumerate(results.items()):
        if 'y_proba_test' in res:
            y_proba = res['y_proba_test']
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, lw=2, color=colors[idx],
                   label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    # Plot random classifier line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Model Comparison', fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importance, output_dir, n_features=20):
    """
    Plot feature importance scores.
    
    Parameters:
    -----------
    importance : array-like
        Feature importance scores
    output_dir : str
        Directory to save the plot
    n_features : int, default=20
        Number of top features to plot
    """
    # Get top features
    indices = np.argsort(importance)[::-1][:n_features]
    top_importance = importance[indices]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(n_features)
    ax.barh(y_pos, top_importance, align='center', color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'Feature {i}' for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {n_features} Most Important Features', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# CUSTOM MACHINE LEARNING MODELS
# ============================================================================

class WeightedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom Weighted Ensemble Classifier
    
    This ensemble combines predictions from multiple base classifiers using
    learned weights. The weights are optimized using logistic regression
    on the base classifier predictions.
    
    This is a novel approach that goes beyond simple voting or averaging.
    
    Parameters:
    -----------
    base_classifiers : list
        List of base classifier objects (will be cloned and fitted)
    """
    
    def __init__(self, base_classifiers):
        self.base_classifiers = base_classifiers
        self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        self.weights = None
        self.fitted_classifiers = None
    
    def fit(self, X, y):
        """
        Fit the base classifiers and meta-learner.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : object
        """
        from sklearn.base import clone
        
        # Clone and fit base classifiers
        self.fitted_classifiers = []
        for clf in self.base_classifiers:
            cloned_clf = clone(clf)
            cloned_clf.fit(X, y)
            self.fitted_classifiers.append(cloned_clf)
        
        # Get predictions from base classifiers
        base_predictions = self._get_base_predictions(X)
        
        # Fit meta-learner
        self.meta_learner.fit(base_predictions, y)
        
        # Extract weights (coefficients of logistic regression)
        self.weights = np.abs(self.meta_learner.coef_[0])
        self.weights = self.weights / np.sum(self.weights)  # Normalize
        
        return self
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predicted class labels
        """
        # Get predictions from base classifiers
        base_predictions = self._get_base_predictions(X)
        
        # Use meta-learner for final prediction
        return self.meta_learner.predict(base_predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        y_proba : array-like, shape (n_samples, n_classes)
            Predicted class probabilities
        """
        # Get predictions from base classifiers
        base_predictions = self._get_base_predictions(X)
        
        # Use meta-learner for final probability prediction
        return self.meta_learner.predict_proba(base_predictions)
    
    def _get_base_predictions(self, X):
        """
        Get predictions from all fitted base classifiers.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        predictions : array-like, shape (n_samples, n_base_classifiers)
            Predictions from base classifiers
        """
        if self.fitted_classifiers is None:
            raise ValueError("Ensemble not fitted yet")
            
        predictions = []
        for clf in self.fitted_classifiers:
            if hasattr(clf, 'predict_proba'):
                # Use probability of positive class
                pred = clf.predict_proba(X)[:, 1]
            else:
                # Use binary predictions
                pred = clf.predict(X)
            predictions.append(pred)
        
        return np.column_stack(predictions)