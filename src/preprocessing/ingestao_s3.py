"""Ingestão da base de alunos (grão aluno) a partir da camada Silver no S3.

Carrega `silver/alunos` (2023+2024), define o alvo binário `alfabetizado` e deixa de
fora as colunas de vazamento (proficiência, presença, preenchimento) — ver Planning §2.
"""
import pandas as pd

from src import config
from src.data_access import ler_parquet_s3

# Colunas do arquivo que carregamos. Ficam de FORA (data leakage / inúteis):
#   proficiencia, presenca, preenchimento_caderno (medidos no exame),
#   caderno (administrativo), serie (constante "2"), peso_aluno (peso amostral),
#   e todas as *_desc (redundantes com os códigos).
_COLS_ALUNOS = [
    "id_aluno",       # identificador (não é feature)
    "id_municipio",   # chave de join com covariáveis municipais
    "id_escola",      # chave de join com o Censo Escolar
    "rede",           # feature: rede de ensino (contexto, não vaza o alvo)
    "sigla_uf",       # feature territorial
    "alfabetizado",   # ALVO (string "0"/"1")
]


def carregar_alunos() -> pd.DataFrame:
    """Lê silver/alunos (2023+2024), tipa o alvo e a partição `ano`."""
    df = ler_parquet_s3(config.SILVER_ALUNOS, columns=_COLS_ALUNOS)
    df["ano"] = df["ano"].astype(int)                    # partição vem como string do path
    df["alfabetizado"] = df["alfabetizado"].astype(int)  # alvo binário 0/1
    df["id_escola"] = df["id_escola"].astype(str)
    df["id_municipio"] = df["id_municipio"].astype(str)
    return df


if __name__ == "__main__":
    d = carregar_alunos()
    print(f"alunos: {len(d):,} linhas x {d.shape[1]} colunas")
    print("distribuição do alvo (alfabetizado):")
    print(d["alfabetizado"].value_counts(normalize=True).round(4))
