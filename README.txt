================================================================================
QSAR BIODEGRADABILITY PREDICTION - SUBMISSION NOTES
================================================================================

Author: Nazrin Atayeva
Student Email: natayeva1@sheffield.ac.uk
Date: December 2025

================================================================================
IMPORTANT: DATA FILE PATH
================================================================================

The code expects the data file to be in the SAME DIRECTORY as the scripts.

File required: QSAR_data.mat
Expected location: Same folder as main_run.py and utils_qsar.py

If you encounter a file not found error, please ensure QSAR_data.mat is in 
the same directory before running the code.

================================================================================
HOW TO RUN
================================================================================

1. Extract all files from the ZIP
2. Place QSAR_data.mat in the same folder
3. Run: python main_run.py

The code will:
- Load and preprocess the data
- Train 7 machine learning models
- Generate all figures in ./outputs/ directory
- Display results in the terminal

Expected runtime: ~20-30 seconds

================================================================================
OUTPUT FILES
================================================================================

The code generates these figures in ./outputs/:
1. feature_distributions.png
2. correlation_matrix.png
3. pca_analysis.png
4. model_comparison.png
5. confusion_matrices.png
6. roc_curves.png
7. feature_importance.png

All figures referenced in the report will be automatically generated.

================================================================================
REQUIREMENTS
================================================================================

Python 3.8+
Libraries: numpy, scipy, scikit-learn, matplotlib, seaborn

All libraries are standard and should be available in typical Python 
environments.

================================================================================
NOTES
================================================================================

- Code uses RANDOM_SEED=42 for reproducibility
- Results should match exactly what's reported in the PDF
- Progress indicators show which step is currently running
- No errors or warnings should appear during execution

If you encounter any issues, please contact: natayeva1@sheffield.ac.uk

================================================================================