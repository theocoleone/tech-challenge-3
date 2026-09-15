"""Ingestão da base de alunos (grão aluno) a partir da camada Silver no S3.

Carrega `silver/alunos` (2023+2024) e o alvo `alfabetizado`. Colunas medidas no exame
(proficiência, presença, preenchimento) ficam de fora — docs/decisoes.md, D02 e D03.
"""
import pandas as pd

from src import config
from src.data_access import ler_parquet_s3

# id_escola é anonimizado no SAEB e não casa com o Censo (D04); rede e sigla_uf são features.
_COLS_ALUNOS = ["id_aluno", "id_municipio", "id_escola", "rede", "sigla_uf", "alfabetizado"]


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
