"""Ingestão da base de alunos (grão aluno) a partir da camada Silver no S3.

Carrega `silver/alunos` (2023+2024) com o alvo `alfabetizado` e a presença na prova.
Proficiência e preenchimento do caderno ficam de fora por vazarem o resultado; `presenca`
entra só para definir população e checar sensibilidade do rótulo, nunca como feature.
"""
import pandas as pd

from src import config
from src.data_access import ler_parquet_s3

_COLS_ALUNOS = ["id_aluno", "id_municipio", "nome", "rede", "sigla_uf", "presenca", "alfabetizado"]


def carregar_alunos() -> pd.DataFrame:
    """Lê silver/alunos (2023+2024), tipa alvo, presença e a partição `ano`."""
    df = ler_parquet_s3(config.SILVER_ALUNOS, columns=_COLS_ALUNOS)
    df = df.rename(columns={"nome": "nome_municipio"})
    df["ano"] = df["ano"].astype(int)
    df["alfabetizado"] = df["alfabetizado"].astype(int)
    df["presenca"] = df["presenca"].astype(int)
    df["id_municipio"] = df["id_municipio"].astype(str)
    return df


if __name__ == "__main__":
    d = carregar_alunos()
    print(f"alunos: {len(d):,} linhas x {d.shape[1]} colunas")
    print("distribuição do alvo (alfabetizado):")
    print(d["alfabetizado"].value_counts(normalize=True).round(4))
    print("presença na prova:")
    print(d["presenca"].value_counts(normalize=True).round(4))
