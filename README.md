# Stroke Prediction Final Term Project

Complete CS634 final-term project implementation for the Kaggle healthcare stroke dataset. The repository includes:

- **Random Forest** (Mandatory model)
- **Conv1D** neural network (Deep learning)
- **SVM (RBF)** (traditional ML)

All models share the same preprocessing pipeline, run under 10-fold stratified cross-validation, and compute the full Module 8 metric suite (TP, TN, FP, FN, P, N, TPR, TNR, FPR, FNR, Precision, Recall, F1, Accuracy, Error Rate, BACC, TSS, HSS, ROC/AUC, BS, BSS). Outputs are persisted as CSV/PNG assets for inclusion in the written report.

## Repository Layout

'''
data/                         # healthcare-dataset-stroke-data.csv (Kaggle download)
notebooks/final_project.ipynb # submission-ready Jupyter notebook
reports/                      # metrics CSV + generated figures (Figures in a seperate sub-folder "figures")
src/final_project.py          # main 10-fold training/evaluation script
requirements.txt              # Python dependencies

'''

## Setup

Compatible Python versions: **3.10 – 3.12** (developed/tested on Python 3.12.0, the latest release supported by dependencies such as TensorFlow 2, scikit-learn 1.4, and seaborn 0.13).

'''powershell (Powershell Terminal inside VSCode)
# In VS Code: View > Command Palette > "Python: Create Environment" (select Venv + current interpreter)
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

'''

Download 'healthcare-dataset-stroke-data.csv' from Kaggle and place it in 'data/healthcare-dataset-stroke-data.csv'.

## Running the Pipeline

'''powershell
# Run from the project root inside the activated VS Code terminal
python src/final_project.py

'''

This command:

1. Generates all figures (correlation heatmap, pairplot, confusion matrices, ROC curves, model comparison bar chart, feature importances) under 'reports/figures/'.
2. Writes 'reports/metrics_all_models.csv', which contains every fold plus the averaged metrics for the three models.
3. Prints the output paths so they can be referenced in the written report.

## Notebook

Open 'notebooks/final_project.ipynb' after running the script (or after installing requirements) to reproduce the screenshots requested in the assignment. The notebook reuses the exact code from 'src/final_project.py'.


## Notes on Replicating

- Seeds ('numpy' and TensorFlow) are fixed in 'src/final_project.py' to keep results stable across runs.
- All project files use relative paths computed from the repository root, so graders can run the code without editing anything.
