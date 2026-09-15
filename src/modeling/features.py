"""Engenharia de atributos aplicada dentro do Pipeline (FunctionTransformer), linha a linha.

Nada aqui é ajustado nos dados, então não há o que vazar; a função entra como primeiro passo
do Pipeline para que o modelo salvo receba as colunas brutas da base.
"""
import numpy as np
import pandas as pd

_MATRICULAS = "esc_media_quantidade_matricula_fundamental_anos_iniciais"
_DOCENTES = "esc_media_quantidade_docente_fundamental_anos_iniciais"
_SALAS = "esc_media_quantidade_sala_utilizada"

NOVAS = ["alunos_por_docente", "matriculas_por_sala", "escolas_por_10k_hab",
         "log_populacao", "log_pib_per_capita"]


def _razao(num, den):
    return (num / den.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def engenharia_atributos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["alunos_por_docente"] = _razao(df[_MATRICULAS], df[_DOCENTES])
    df["matriculas_por_sala"] = _razao(df[_MATRICULAS], df[_SALAS])
    df["escolas_por_10k_hab"] = _razao(df["esc_n_escolas"] * 1e4, df["populacao"])
    df["log_populacao"] = np.log1p(df["populacao"])
    df["log_pib_per_capita"] = np.log1p(df["pib_per_capita"].clip(lower=0))
    return df
