# Big Data — Submission Guide

Ovo je uputstvo za pregled, bez potrebe da se kod izvršava.

## Šta je ovde
- **Glavni projekat**: kompletan kod je u root repozitorijumu.
- **Ovaj dokument**: vodič za deo "Veliki podaci" (Dask, CSV→Parquet, spajanje, čišćenje).

## Gde je relevantni kod (bez pokretanja)
- Učitavanje i spajanje CSV/ZIP (Dask): `row_data.py`
- Konverzija CSV → Parquet: `convert_to_parquet.py`
- Centralni cleaning pipeline: `clean_data.py`
- (Opcionalno) Notebook demonstracija: `big_data.ipynb`

> Napomena: Dataseti nisu uključeni zbog veličine. Kod je organizovan za rad sa višegigabajt­nim fajlovima (Pandas → Dask, Parquet za brži I/O).

## Ključne poruke (tl;dr)
- **Scaling:** prešli smo sa Pandas na **Dask** da bismo obradili multi-GB setove.  
- **Optimizacija:** **Parquet** format značajno ubrzava I/O i štedi RAM.  
- **Modularnost:** odvojene skripte za učitavanje, spajanje, čišćenje i konverziju.  

## Ako želite da pokrenete
Potrebne zavisnosti:
Primer pokretanja (ako imate podatke):
```bash
# konverzija CSV → Parquet
python convert_to_parquet.py data/raw_data/processed/merged_clean_data.csv

# čišćenje i spajanje
python clean_data.py

# notebook
jupyter notebook big_data.ipynb


