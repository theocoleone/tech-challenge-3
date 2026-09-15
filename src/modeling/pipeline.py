"""Features e pré-processamento (ColumnTransformer) do modelo supervisionado do TC3.

Todo o pré-processamento vive dentro de um Pipeline sklearn e só é `fit` no treino —
isso previne data leakage de forma sistemática (imputação/encoding aprendidos só no treino).
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ALVO = "alfabetizado"
IDS = ["id_aluno", "id_municipio", "id_escola"]

# Fora das features: identificadores, alvo e alta cardinalidade com sinal fraco.
DROP = IDS + [ALVO, "nome_mesorregiao"]

CATEGORICAS = ["rede", "sigla_uf", "nome_regiao", "amazonia_legal", "capital_uf", "ano"]


def separar_features(df):
    """Retorna (X, y, numericas, categoricas). Descarta ids, alvo e colunas 100% nulas."""
    y = df[ALVO].astype(int)
    fora = set(DROP)
    categoricas = [c for c in CATEGORICAS if c in df.columns]
    numericas = [
        c for c in df.columns
        if c not in fora and c not in categoricas and df[c].notna().any()
    ]
    X = df[categoricas + numericas].copy()
    for c in categoricas:            # bool/int viram string p/ o OneHot tratar como categoria
        X[c] = X[c].astype(str)
    return X, y, numericas, categoricas


def construir_preprocessador(numericas, categoricas) -> ColumnTransformer:
    """Numéricas: imputação (mediana) + padronização. Categóricas: imputação + One-Hot."""
    num = Pipeline([
        ("imput", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),          # ajuda LogReg; inócuo p/ árvores
    ])
    cat = Pipeline([
        ("imput", SimpleImputer(strategy="constant", fill_value="NA")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num, numericas),
        ("cat", cat, categoricas),
    ])
