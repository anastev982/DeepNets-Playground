import os


def plot_learning_curve(clf_builder, X, y, steps=(0.1,0.2,0.4,0.6,0.8,1.0)):
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score
    from sklearn.utils import shuffle
    Xs, ys = shuffle(X, y, random_state=RNG)
    tr_scores, te_scores, sizes = [], [], []
    for frac in steps:
        n = max(50, int(len(Xs)*frac))
        Xn, yn = Xs[:n], ys[:n]
        Xtr, Xte, ytr, yte = train_test_split(Xn, yn, test_size=0.2, stratify=yn, random_state=RNG)
        clf = clf_builder(); clf.fit(Xtr, ytr)
        tr = f1_score(ytr, clf.predict(Xtr), average="macro")
        te = f1_score(yte, clf.predict(Xte), average="macro")
        sizes.append(n); tr_scores.append(tr); te_scores.append(te)
    plt.figure(figsize=(6,4))
    plt.plot(sizes, tr_scores, marker="o", label="train F1")
    plt.plot(sizes, te_scores, marker="o", label="test F1")
    plt.title("Learning Curve — Digits")
    plt.xlabel("Training samples"); plt.ylabel("macro-F1"); plt.legend(); plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/learning_curve_digits.png", dpi=150); plt.close()
    print("Saved → outputs/learning_curve_digits.png")

