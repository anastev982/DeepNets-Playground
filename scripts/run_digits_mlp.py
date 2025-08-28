# This script trains an MLPClassifier on the Digits dataset using scikit-learn, evaluates with CV and test set, and saves results (confusion matrix, plots, metrics JSON).

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np, json, os

RNG = 42

def build_clf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=RNG))
    ])

def main():
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RNG)

    clf = build_clf()

    # CV on train (macro-F1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    cv_scores = cross_val_score(clf, Xtr, ytr, cv=cv, scoring="f1_macro")
    print(f"CV f1_macro: {cv_scores.mean():.3f} ± {cv_scores.std():.3f} {np.round(cv_scores,3)}")

    # Fit + test
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    acc = accuracy_score(yte, yp)
    f1m = f1_score(yte, yp, average="macro")
    print(f"TEST acc={acc:.3f} f1_macro={f1m:.3f}")
    print(classification_report(yte, yp, digits=3))

    os.makedirs("outputs", exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(yte, yp)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix — Digits (Test)")
    plt.xlabel("Predicted"); 
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            plt.text(j, i, v, ha="center", va="center",
                     color=("white" if v > cm.max()/2 else "black"))
    plt.tight_layout(); 
    plt.savefig("outputs/confusion_digits.png", dpi=150); 
    plt.close()

    # Sample predictions grid (✓/✗)
    n = 25
    rs = np.random.RandomState(RNG)
    idx = rs.choice(len(Xte), size=n, replace=False)
    imgs = Xte[idx].reshape(-1, 8, 8); true = yte[idx]; pred = yp[idx]
    plt.figure(figsize=(8,8))
    for k in range(n):
        ax = plt.subplot(5,5,k+1)
        ax.imshow(imgs[k], cmap="gray_r")
        ok = (pred[k] == true[k])
        ax.set_title(f"p={pred[k]} {'✓' if ok else '✗'}", fontsize=9, color=("black" if ok else "crimson"))
        ax.axis("off")
    plt.suptitle("Digits — Predictions (✓ correct, ✗ mistakes)", y=0.93)
    plt.tight_layout(); plt.savefig("outputs/examples_digits.png", dpi=150); plt.close()

    # Metrics JSON (handy for review)
    metrics = {
        "cv": {"n_splits": 5, "f1_macro_mean": float(cv_scores.mean()), "f1_macro_std": float(cv_scores.std()),
               "per_fold": [float(x) for x in cv_scores]},
        "test": {"accuracy": float(acc), "f1_macro": float(f1m)}
    }
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved → outputs/confusion_digits.png, outputs/examples_digits.png, outputs/metrics.json")

if __name__ == "__main__":
    main()

