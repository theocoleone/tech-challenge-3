"""Seleção de features e Pipeline (engenharia de atributos + ColumnTransformer + modelo).

Tudo vive dentro de um Pipeline sklearn e só é ajustado no treino, o que fecha o vazamento
via imputação/encoding. As colunas que vazam o alvo são barradas em `separar_features`.
`ano` fica fora: AC, DF e SP só existem em 2024 (confunde com UF) e um ano novo seria uma
categoria desconhecida na predição.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config
from src.modeling.features import NOVAS, engenharia_atributos

ALVO = config.COLUNA_ALVO
IDS = ["id_aluno", "id_municipio", "nome_municipio"]
DROP = IDS + [ALVO, "nome_mesorregiao", "ano"]
CATEGORICAS = ["rede", "sigla_uf", "nome_regiao", "amazonia_legal", "capital_uf"]


def separar_features(df, categoricas=CATEGORICAS):
    """Retorna (X, y, grupos, numericas, categoricas).

    X traz as colunas brutas (a engenharia de atributos roda dentro do Pipeline); `numericas`
    já inclui as derivadas. Descarta ids, alvo, `ano`, vazamento e colunas 100% nulas.
    """
    y = df[ALVO].astype(int)
    grupos = df["id_municipio"]
    fora = set(DROP) | set(config.COLUNAS_LEAKAGE)
    categoricas = [c for c in categoricas if c in df.columns]
    brutas = [
        c for c in df.columns
        if c not in fora and c not in categoricas and df[c].notna().any()
    ]
    X = df[categoricas + brutas].copy()
    for c in categoricas:
        X[c] = X[c].astype(str).where(X[c].notna(), np.nan)
    vazadas = set(X.columns) & set(config.COLUNAS_LEAKAGE)
    if vazadas:
        raise ValueError(f"colunas de vazamento entre as features: {sorted(vazadas)}")
    return X, y, grupos, brutas + NOVAS, categoricas


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


def construir_pipeline(modelo, numericas, categoricas) -> Pipeline:
    return Pipeline([
        ("atributos", FunctionTransformer(engenharia_atributos)),
        ("pre", construir_preprocessador(numericas, categoricas)),
        ("modelo", modelo),
    ])


def nomes_features(pipe: Pipeline) -> list:
    return list(pipe.named_steps["pre"].get_feature_names_out())
