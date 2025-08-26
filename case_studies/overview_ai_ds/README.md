# Nauka o podacima i veštačka inteligencija — Sažetak

Ovaj pregled povezuje tri mini case-study projekta koji zajedno pokrivaju gradivo predmeta:
- **Nauka o podacima (DS):** problem → podaci → priprema → model → validacija → zaključak
- **Veštačka inteligencija (AI):** neuronske mreže (MLP) i primena na realnim podacima
- **NLP (tekst):** vektorizacija teksta i klasifikacija

## 1) DS lifecycle (Breast Cancer)
- **Cilj:** binarna klasifikacija (malign/benign).
- **Tehnike:** standardizacija, Logistic Regression, Random Forest.
- **Validacija:** stratified 5-fold CV + test (macro-F1, accuracy, ROC AUC).
- **Repo putanja:** `case_studies/ds_lifecycle/01_pipeline_breast_cancer.ipynb`
- **Takeaway:** jednostavan, transparentan model često daje vrhunske rezultate kada su feature-i informativni.

## 2) AI / DNN (Digits MLP)
- **Cilj:** klasifikacija cifara 8×8.
- **Model:** MLP (scikit-learn), pipeline sa skaliranjem.
- **Validacija:** 5-fold CV + test; čuvanje metrika/plotova.
- **Pokretanje:** `python scripts/run_digits_mlp.py`
- **Takeaway:** mali MLP je odličan „most” ka dubljim mrežama (CNN), i demonstrira osnovne AI principe.

## 3) NLP (Text Classification)
- **Cilj:** klasifikacija teksta po temama/sentimentu.
- **Tehnike:** TF-IDF (unigram+bigram) + Logistic Regression / Naive Bayes.
- **Validacija:** stratified CV + test; konfuziona matrica.
- **Repo putanja:** `case_studies/nlp_sentiment/01_sentiment_analysis.ipynb`
- **Takeaway:** i bez dubokih mreža, klasična NLP cevovod (TF-IDF + LogReg/NB) daje snažne baseline rezultate.

## Zašto je ovo povezano
- **Sinteza:** kompletne putanje od podataka do zaključka (DS).
- **Evaluacija:** korektne protokole merenja (CV + test, macro-F1).
- **AI:** neuronske mreže na ciframa + primene ML na tabularnim i tekstualnim podacima.

## Šta bih naglasila na usmenom
- Kako sam dizajnirala validaciju i zašto macro-F1.
- Zašto StandardScaler uz LogReg.
- Zašto je NB dobar baseline za tekst.
- Šta bih sledeće radila: tuning hiperparametara, interpretabilnost, kalibracija, veći modeli (CNN/transformeri).
