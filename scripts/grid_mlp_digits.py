from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

RNG = 42
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RNG)

candidates = [
    (128, 64),
    (256, 128),
    (64, 64),
    (128,),        # single layer
    (256,),
]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)

best = None
for h in candidates:
    clf = Pipeline([("scaler", StandardScaler()),
                    ("mlp", MLPClassifier(hidden_layer_sizes=h, max_iter=300, random_state=RNG))])
    scores = cross_val_score(clf, Xtr, ytr, cv=cv, scoring="f1_macro")
    print(f"h={h}: CV f1_macro={scores.mean():.3f} ± {scores.std():.3f} {np.round(scores,3)}")
    if best is None or scores.mean() > best[0]:
        best = (scores.mean(), h)
print(f"\nBEST: {best[1]} with CV f1_macro={best[0]:.3f}")

# Final model with best architecture
best_h = best[1]
final_clf = Pipeline([("scaler", StandardScaler()),
                      ("mlp", MLPClassifier(hidden_layer_sizes=best_h, max_iter=300, random_state=RNG))])
# Fit on full train set
final_clf.fit(Xtr, ytr)

# Evaluate on test set
yp = final_clf.predict(Xte)
print("TEST f1_macro:", f1_score(yte, yp, average='macro'))
print(classification_report(yte, yp))
print(confusion_matrix(yte, yp))