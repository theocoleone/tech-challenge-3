"""Monta a base analítica (grão aluno) e persiste em local + S3.

Joins:
  alunos ⨝ censo_escolar   ON (id_municipio, ano)   contexto escolar do município no ano da prova
  alunos ⨝ socio_municipio ON  id_municipio         covariáveis municipais estáticas

Saída: data/base_analitica.parquet e s3://<bucket>/tc3/features/base_analitica.parquet.
Rodar da raiz do repo:  python -m src.preprocessing.montar_base
"""
import os

from src import config
from src.data_access import gravar_parquet_s3
from src.preprocessing.ingestao_bq import (
    carregar_censo_escolar_municipio,
    carregar_socioeconomico_municipio,
)
from src.preprocessing.ingestao_s3 import carregar_alunos

DATA_LOCAL = config.BASE_LOCAL


def montar_base(salvar_s3: bool = True):
    print("1/4 alunos (S3)...")
    alunos = carregar_alunos()
    print(f"     {len(alunos):,} alunos")

    print("2/4 censo escolar agregado por município (BigQuery)...")
    escola = carregar_censo_escolar_municipio()
    print(f"     {len(escola):,} município-anos de escola")

    print("3/4 socioeconômico municipal (BigQuery)...")
    muni = carregar_socioeconomico_municipio()
    print(f"     {len(muni):,} municípios")

    print("4/4 joins...")
    base = (
        alunos
        .merge(escola, on=["id_municipio", "ano"], how="left")
        .merge(muni, on="id_municipio", how="left")
    )
    print(f"     base final: {base.shape[0]:,} linhas x {base.shape[1]} colunas")

    casou_escola = base["esc_prop_internet"].notna().mean()
    casou_muni = base["idhm"].notna().mean()
    print(f"     match censo escolar: {casou_escola:.1%} | match município (idhm): {casou_muni:.1%}")
    print(f"     presença na prova: {base['presenca'].mean():.1%} | "
          f"taxa de alfabetização: {base['alfabetizado'].mean():.1%}")

    os.makedirs(os.path.dirname(DATA_LOCAL), exist_ok=True)
    base.to_parquet(DATA_LOCAL, index=False)
    print(f"     salvo local -> {os.path.relpath(DATA_LOCAL, config.RAIZ)}")

    if salvar_s3:
        try:
            uri = gravar_parquet_s3(base, f"{config.TC3_FEATURES}/base_analitica.parquet")
            print(f"     salvo S3 -> {uri}")
        except Exception as e:
            print(f"     [aviso] não foi possível gravar no S3 ({type(e).__name__}: {e}). "
                  f"Base local disponível para prosseguir.")
    return base


if __name__ == "__main__":
    montar_base()
