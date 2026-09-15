"""Seleção de features e pré-processamento (ColumnTransformer) do modelo supervisionado.

Tudo vive dentro de um Pipeline sklearn e só é ajustado no treino, o que fecha o vazamento
via imputação/encoding. As colunas que vazam o alvo são barradas em `separar_features`.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

ALVO = config.COLUNA_ALVO
IDS = ["id_aluno", "id_municipio", "id_escola", "nome_municipio"]
DROP = IDS + [ALVO, "nome_mesorregiao"]
CATEGORICAS = ["rede", "sigla_uf", "nome_regiao", "amazonia_legal", "capital_uf", "ano"]


def separar_features(df, categoricas=CATEGORICAS):
    """Retorna (X, y, numericas, categoricas). Descarta ids, alvo, vazamento e colunas 100% nulas."""
    y = df[ALVO].astype(int)
    fora = set(DROP) | set(config.COLUNAS_LEAKAGE)
    categoricas = [c for c in categoricas if c in df.columns]
    numericas = [
        c for c in df.columns
        if c not in fora and c not in categoricas and df[c].notna().any()
    ]
    X = df[categoricas + numericas].copy()
    for c in categoricas:
        X[c] = X[c].astype(str).where(X[c].notna(), np.nan)
    vazadas = set(X.columns) & set(config.COLUNAS_LEAKAGE)
    if vazadas:
        raise ValueError(f"colunas de vazamento entre as features: {sorted(vazadas)}")
    return X, y, numericas, categoricas


def construir_preprocessador(numericas, categoricas) -> ColumnTransformer:
    """Numéricas: mediana + padronização. Categóricas: constante 'NA' + One-Hot."""
    num = Pipeline([
        ("imput", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat = Pipeline([
        ("imput", SimpleImputer(strategy="constant", fill_value="NA")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num, numericas),
        ("cat", cat, categoricas),
    ])
