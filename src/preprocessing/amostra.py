"""Amostra da base analítica para rodar o projeto sem acesso ao S3/BigQuery.

Sorteia municípios inteiros (todos os alunos, nos dois anos), proporcionalmente por UF,
para que as análises por município continuem fazendo sentido. Municípios muito grandes
ficam fora para a amostra caber no repositório.

Rodar da raiz do repo:  python -m src.preprocessing.amostra
"""
import os

import pandas as pd

from src import config

N_MUNICIPIOS = 150
MIN_ALUNOS, MAX_ALUNOS = 50, 3000
SEED = 42


def gerar_amostra(n_municipios=N_MUNICIPIOS, seed=SEED) -> pd.DataFrame:
    base = pd.read_parquet(config.BASE_LOCAL)
    por_muni = base.groupby("id_municipio").agg(n=("id_aluno", "size"), uf=("sigla_uf", "first"))
    elegiveis = por_muni[por_muni["n"].between(MIN_ALUNOS, MAX_ALUNOS)]

    cota = (elegiveis["uf"].value_counts(normalize=True) * n_municipios).round().astype(int)
    cota = cota[cota > 0]
    sorteados = pd.concat([
        elegiveis[elegiveis["uf"] == uf].sample(min(k, (elegiveis["uf"] == uf).sum()), random_state=seed)
        for uf, k in cota.items()
    ])
    amostra = base[base["id_municipio"].isin(sorteados.index)].reset_index(drop=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    amostra.to_parquet(config.AMOSTRA_LOCAL, index=False, compression="zstd")
    tamanho = os.path.getsize(config.AMOSTRA_LOCAL) / 1024 ** 2
    print(f"amostra: {len(sorteados)} municípios, {len(amostra):,} alunos, "
          f"{amostra['sigla_uf'].nunique()} UFs -> {os.path.relpath(config.AMOSTRA_LOCAL, config.RAIZ)} "
          f"({tamanho:.1f} MB)")
    return amostra


if __name__ == "__main__":
    gerar_amostra()
