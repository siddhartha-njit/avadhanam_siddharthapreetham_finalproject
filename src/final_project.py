import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import matplotlib            
matplotlib.use("Agg") 

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, brier_score_loss, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "healthcare-dataset-stroke-data.csv"
OUT_DIR   = BASE_DIR / "reports"
FIG_DIR   = OUT_DIR / "figures"
RANDOM_STATE = 42
KFOLDS = 10

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


CATEGORICAL = ["gender","ever_married","work_type","Residence_type","smoking_status"]
NUMERIC     = ["age","hypertension","heart_disease","avg_glucose_level","bmi"]
TARGET      = "stroke"


def load_dataset(csv_path=DATA_PATH):
    csv_path = Path(csv_path)
    if not csv_path.is_absolute():
        csv_path = BASE_DIR / csv_path
    df = pd.read_csv(csv_path)

    expected = set(CATEGORICAL + NUMERIC + [TARGET])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    if df["bmi"].isna().any():
        df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    
    df[TARGET] = df[TARGET].astype(int)
    return df

def _make_onehot():
    kwargs = {"handle_unknown": "ignore"}
    try:
        kwargs["sparse_output"] = False
        return OneHotEncoder(**kwargs)
    except TypeError:
        kwargs.pop("sparse_output", None)
        kwargs["sparse"] = False
        return OneHotEncoder(**kwargs)


def to_numpy(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def build_preprocessor():
    cat = Pipeline(steps=[("onehot", _make_onehot())])
    num = Pipeline(steps=[("scaler", StandardScaler())])
    pre = ColumnTransformer(
        transformers=[
            ("cat", cat, CATEGORICAL),
            ("num", num, NUMERIC),
        ]
    )
    return pre

def cm_counts(cm):
    TP, FN = cm[0,0], cm[0,1]
    FP, TN = cm[1,0], cm[1,1]
    P = TP + FN
    N = TN + FP
    return TP, TN, FP, FN, P, N

def compute_all_metrics(cm, y_true, y_prob, auc_val=None):
    TP, TN, FP, FN, P, N = cm_counts(cm)
    TPR = TP / P if P else 0.0
    TNR = TN / N if N else 0.0
    FPR = FP / N if N else 0.0
    FNR = FN / P if P else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    f1 = (2*TP) / (2*TP + FP + FN) if (2*TP + FP + FN) else 0.0
    acc = (TP + TN) / (P + N) if (P + N) else 0.0
    err = 1 - acc
    bacc = 0.5 * (TPR + TNR)
    tss = TPR - FPR
    denom = ((TP+FN)*(FN+TN) + (TP+FP)*(FP+TN))
    hss = (2*(TP*TN - FP*FN))/denom if denom else 0.0

    if auc_val is None:
        auc_val = roc_auc_score(y_true, y_prob)
    bs = brier_score_loss(y_true, y_prob)
    pbar = np.mean(y_true)
    bs_ref = np.mean((y_true - pbar)**2)  # climatology baseline = p*(1-p)
    bss = 1.0 - (bs / bs_ref) if bs_ref > 0 else np.nan

    return {
        "TP":TP,"TN":TN,"FP":FP,"FN":FN,"P":P,"N":N,
        "TPR":TPR,"TNR":TNR,"FPR":FPR,"FNR":FNR,
        "Precision":precision,"Recall":TPR,"F1":f1,
        "Accuracy":acc,"ErrorRate":err,"BACC":bacc,
        "TSS":tss,"HSS":hss,"AUC":auc_val,"BS":bs,"BSS":bss
    }

def train_predict_rf(Xtr, ytr, Xte):
    clf = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)
    yprob = clf.predict_proba(Xte)[:,1]
    return clf, yhat, yprob

def train_predict_svm(Xtr, ytr, Xte):
    clf = SVC(kernel="rbf", C=2.0, gamma="scale",
              class_weight="balanced", probability=True, random_state=RANDOM_STATE)
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)
    yprob = clf.predict_proba(Xte)[:,1]
    return clf, yhat, yprob

def build_conv1d(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, kernel_size=3, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_predict_conv1d(Xtr, ytr, Xte, epochs=30, batch_size=64):
    Xtr_r = Xtr.reshape((Xtr.shape[0], Xtr.shape[1], 1))
    Xte_r = Xte.reshape((Xte.shape[0], Xte.shape[1], 1))
    model = build_conv1d((Xtr_r.shape[1], 1))
    pos = np.sum(ytr)
    neg = ytr.shape[0] - pos
    if pos > 0 and neg > 0:
        total = ytr.shape[0]
        class_weight = {
            0: total / (2 * neg),
            1: total / (2 * pos)
        }
    else:
        class_weight = None
    model.fit(
        Xtr_r,
        ytr,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        class_weight=class_weight,
    )
    yprob = model.predict(Xte_r, verbose=0).ravel()
    yhat  = (yprob >= 0.5).astype(int)
    return model, yhat, yprob

def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_pairplot(df, out_path, hue=TARGET):
    cols = NUMERIC + [hue]
    if not set(cols).issubset(df.columns):
        raise ValueError("Pairplot columns missing from dataframe")
    pair_df = df[cols].copy()
    grid = sns.pairplot(
        pair_df,
        hue=hue,
        corner=True,
        diag_kind="hist",
        plot_kws={"alpha":0.6, "s":20, "edgecolor":"none"},
        diag_kws={"bins":20, "edgecolor":"black"},
    )
    grid.fig.suptitle("Pairplot (Numeric Features vs Stroke)", y=1.02)
    grid.fig.savefig(out_path, bbox_inches="tight")
    plt.close(grid.fig)

def run_model_evaluation(df, n_splits=KFOLDS):
    X_raw = df[CATEGORICAL + NUMERIC]
    y     = df[TARGET].values.astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    cm_sum_rf = np.zeros((2,2), dtype=int)
    cm_sum_svm = np.zeros((2,2), dtype=int)
    cm_sum_cnn = np.zeros((2,2), dtype=int)

    oof_probs = {"RandomForest": [], "SVM_RBF": [], "Conv1D": []}
    oof_true  = []

    fold = 0
    for tr_idx, te_idx in skf.split(X_raw, y):
        fold += 1
        pre = build_preprocessor()
        Xtr = to_numpy(pre.fit_transform(X_raw.iloc[tr_idx]))
        Xte = to_numpy(pre.transform(X_raw.iloc[te_idx]))
        ytr, yte = y[tr_idx], y[te_idx]

        rf, yhat_rf, yprob_rf = train_predict_rf(Xtr, ytr, Xte)
        cm_rf = confusion_matrix(yte, yhat_rf, labels=[1,0])
        met_rf = compute_all_metrics(cm_rf, yte, yprob_rf)
        rows.append({"Fold":fold, "Model":"RandomForest", **met_rf})
        cm_sum_rf += cm_rf
        oof_probs["RandomForest"].append(yprob_rf)

        svm, yhat_svm, yprob_svm = train_predict_svm(Xtr, ytr, Xte)
        cm_svm = confusion_matrix(yte, yhat_svm, labels=[1,0])
        met_svm = compute_all_metrics(cm_svm, yte, yprob_svm)
        rows.append({"Fold":fold, "Model":"SVM_RBF", **met_svm})
        cm_sum_svm += cm_svm
        oof_probs["SVM_RBF"].append(yprob_svm)

        cnn, yhat_cnn, yprob_cnn = train_predict_conv1d(Xtr, ytr, Xte)
        cm_cnn = confusion_matrix(yte, yhat_cnn, labels=[1,0])
        met_cnn = compute_all_metrics(cm_cnn, yte, yprob_cnn)
        rows.append({"Fold":fold, "Model":"Conv1D", **met_cnn})
        cm_sum_cnn += cm_cnn
        oof_probs["Conv1D"].append(yprob_cnn)

        oof_true.append(yte)

    for k in oof_probs:
        oof_probs[k] = np.concatenate(oof_probs[k], axis=0)
    y_true_all = np.concatenate(oof_true, axis=0)

    dfm = pd.DataFrame(rows)
    metric_cols = [c for c in dfm.columns if c not in ("Fold", "Model")]
    avg = (
        dfm.groupby("Model")[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    avg_with_fold = avg.copy()
    avg_with_fold.insert(0, "Fold", "AVG")
    fold_metrics = pd.concat([dfm, avg_with_fold], ignore_index=True)
    confusions = {
        "RandomForest": cm_sum_rf,
        "SVM_RBF": cm_sum_svm,
        "Conv1D": cm_sum_cnn,
    }

    return {
        "fold_metrics": fold_metrics,
        "avg_metrics": avg,
        "confusions": confusions,
        "oof_probs": oof_probs,
        "y_true": y_true_all,
    }

def plot_corr_heatmap(df, out_path, cmap="Blues"):
    plt.figure()
    corr = df[NUMERIC].corr()
    im = plt.imshow(corr, interpolation="nearest", cmap=cmap)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.xticks(range(len(NUMERIC)), NUMERIC, rotation=45, ha="right")
    plt.yticks(range(len(NUMERIC)), NUMERIC)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_confusion_avg(cm_sum, title, out_path):
    total = cm_sum.sum()
    mat = cm_sum / total if total else cm_sum
    plt.figure()
    plt.imshow(mat, cmap="Blues")
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks([0,1], ["Pred 1","Pred 0"])
    plt.yticks([0,1], ["Actual 1","Actual 0"])
    for (i,j), val in np.ndenumerate(mat):
        plt.text(j, i, f"{val:.3f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_roc_all(models_probs, y_true_all, out_path):
    plt.figure()
    for name, probs in models_probs.items():
        fpr, tpr, _ = roc_curve(y_true_all, probs)
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (OOF, all folds)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_rf_feature_importance(rf, preprocessor, out_path, top_k=20):
    try:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(CATEGORICAL)
    except Exception:
        cat_names = np.array([])
    num_names = np.array(NUMERIC)
    feature_names = np.concatenate([cat_names, num_names]) if len(cat_names) else num_names
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:top_k]
    plt.figure(figsize=(8, max(4, top_k*0.25)))
    plt.barh(range(len(idx)), importances[idx][::-1])
    plt.yticks(range(len(idx)), feature_names[idx][::-1], fontsize=9)
    plt.xlabel("Importance")
    plt.title("Random Forest — Top Feature Importances")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_model_compare_bar(avg_df, out_path, metrics=("AUC","F1","BACC","TSS","HSS")):
    subset = avg_df.set_index("Model")[list(metrics)]
    ax = subset.plot(kind="bar")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), borderaxespad=0)
    plt.title("Model Comparison (Averages)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def main():
    ensure_dirs()
    df = load_dataset(DATA_PATH)

    corr_path = FIG_DIR / "correlation_heatmap.png"
    plot_corr_heatmap(df, corr_path, cmap="Blues")
    pairplot_path = FIG_DIR / "pairplot_numeric.png"
    plot_pairplot(df, pairplot_path, hue=TARGET)

    results = run_model_evaluation(df, n_splits=KFOLDS)
    csv_path = OUT_DIR / "metrics_all_models.csv"
    results["fold_metrics"].to_csv(csv_path, index=False)

    plot_confusion_avg(results["confusions"]["RandomForest"], "RandomForest (Avg Confusion, normalized)",
                       FIG_DIR / "confusion_avg_rf.png")
    plot_confusion_avg(results["confusions"]["SVM_RBF"], "SVM RBF (Avg Confusion, normalized)",
                       FIG_DIR / "confusion_avg_svm.png")
    plot_confusion_avg(results["confusions"]["Conv1D"], "Conv1D (Avg Confusion, normalized)",
                       FIG_DIR / "confusion_avg_conv1d.png")

    plot_roc_all(results["oof_probs"], results["y_true"], FIG_DIR / "roc_all_models.png")

    pre_full = build_preprocessor()
    X_full = to_numpy(pre_full.fit_transform(df[CATEGORICAL + NUMERIC]))
    rf_full = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_full.fit(X_full, df[TARGET].values.astype(int))
    plot_rf_feature_importance(rf_full, pre_full, FIG_DIR / "rf_feature_importance.png", top_k=20)

    plot_model_compare_bar(results["avg_metrics"], FIG_DIR / "model_compare_bar.png")

    print(f"Done. CSV saved to {csv_path}")
    print(f"Figures saved to {FIG_DIR}")

if __name__ == "__main__":
    main()
