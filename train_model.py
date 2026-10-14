"""
CropSense - model training script.

Loads the crop recommendation dataset, compares four classifiers,
saves the best model and writes charts used in the report.

Run:  python train_model.py
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "Crop_recommendation.csv")
MODEL_DIR = os.path.join(BASE, "model")
OUT_DIR = os.path.join(BASE, "outputs")
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA)
    df = df.dropna().drop_duplicates()
    print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns, {df['label'].nunique()} crops")
    return df


def plot_dataset(df):
    # Distribution of each input feature
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, FEATURES):
        sns.histplot(df[col], kde=True, ax=ax, color="#2e7d32")
        ax.set_title(col)
    axes.flat[-1].axis("off")
    fig.suptitle("Distribution of input features")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "feature_distribution.png"), dpi=150)
    plt.close(fig)

    # Correlation between features
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df[FEATURES].corr(), annot=True, fmt=".2f", cmap="Greens", ax=ax)
    ax.set_title("Feature correlation")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)


def train_and_compare(df):
    X, y = df[FEATURES], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        cv = cross_val_score(model, X, y, cv=5).mean()
        results[name] = {"test_accuracy": round(acc * 100, 2), "cv_accuracy": round(cv * 100, 2)}
        print(f"{name:20s}  test accuracy = {acc*100:.2f}%   5-fold CV = {cv*100:.2f}%")

    best_name = max(results, key=lambda n: (results[n]["test_accuracy"], results[n]["cv_accuracy"]))
    best = models[best_name]
    print(f"\nBest model: {best_name}")

    y_pred = best.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    # Accuracy comparison chart
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results)
    accs = [results[n]["test_accuracy"] for n in names]
    bars = ax.bar(names, accs, color=["#a5d6a7", "#81c784", "#66bb6a", "#2e7d32"])
    ax.bar_label(bars, fmt="%.2f%%")
    ax.set_ylim(min(accs) - 5, 101)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Model accuracy comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)

    # Confusion matrix of the best model
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # Feature importance (Random Forest)
    rf = models["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    imp.plot.barh(ax=ax, color="#2e7d32")
    ax.set_title("Feature importance (Random Forest)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)

    # Retrain best model on the full dataset and save
    best.fit(X, y)
    joblib.dump(best, os.path.join(MODEL_DIR, "crop_model.pkl"))

    summary = {
        "best_model": best_name,
        "results": results,
        "macro_precision": round(report["macro avg"]["precision"] * 100, 2),
        "macro_recall": round(report["macro avg"]["recall"] * 100, 2),
        "macro_f1": round(report["macro avg"]["f1-score"] * 100, 2),
        "feature_importance": {k: round(v, 4) for k, v in imp.sort_values(ascending=False).items()},
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    data = load_data()
    plot_dataset(data)
    train_and_compare(data)
    print("Saved model to model/crop_model.pkl and charts to outputs/")
