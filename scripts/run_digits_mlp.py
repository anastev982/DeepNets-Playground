from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
RANDOM_STATE = 42
X, y = load_digits(return_X_y=True)

train/test split

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
stratify=y, random_state=RANDOM_STATE)

pipeline: scaling + MLP

clf = Pipeline([
("scaler", StandardScaler()),
("mlp", MLPClassifier(hidden_layer_sizes=(128,64),
max_iter=300, random_state=RANDOM_STATE))
])

cross-validation

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores = cross_val_score(clf, Xtr, ytr, cv=cv, scoring="f1_macro")
print(f"CV f1_macro: {scores.mean():.3f} ± {scores.std():.3f} {np.round(scores,3)}")

fit + evaluate

clf.fit(Xtr, ytr)
yp = clf.predict(Xte)
print(f"TEST acc={accuracy_score(yte, yp):.3f} f1_macro={f1_score(yte, yp, average='macro'):.3f}")
print(classification_report(yte, yp, digits=3))
