"""Monta a base analítica do TC3 (grão aluno) e persiste em local + S3.

Join:
  alunos ⨝ censo_escolar  ON (id_escola, ano)      -> contexto da escola (year-matched)
  alunos ⨝ socio_municipio ON  id_municipio         -> covariáveis municipais (estáticas)

Saída: data/base_analitica.parquet (para EDA local) e s3://<bucket>/tc3/features/ (para o SageMaker).
"""
import os

from src import config
from src.data_access import gravar_parquet_s3
from src.preprocessing.ingestao_bq import (
    carregar_censo_escolar,
    carregar_socioeconomico_municipio,
)
from src.preprocessing.ingestao_s3 import carregar_alunos

DATA_LOCAL = os.path.join(os.path.dirname(__file__), "..", "..", "data", "base_analitica.parquet")


def montar_base(salvar_s3: bool = True):
    print("1/4 alunos (S3)...")
    alunos = carregar_alunos()
    print(f"     {len(alunos):,} alunos")

    print("2/4 censo escolar (BigQuery)...")
    escola = carregar_censo_escolar()
    print(f"     {len(escola):,} escola-anos")

    print("3/4 socioeconômico municipal (BigQuery)...")
    muni = carregar_socioeconomico_municipio()
    print(f"     {len(muni):,} municípios")

    print("4/4 joins...")
    base = (
        alunos
        .merge(escola, on=["id_escola", "ano"], how="left")
        .merge(muni, on="id_municipio", how="left")
    )
    print(f"     base final: {base.shape[0]:,} linhas x {base.shape[1]} colunas")

    # cobertura dos joins (quanto casou) — sinaliza problemas de chave
    casou_escola = base["esc_tipo_localizacao"].notna().mean()
    casou_muni = base["idhm"].notna().mean()
    print(f"     match escola: {casou_escola:.1%} | match município (idhm): {casou_muni:.1%}")

    os.makedirs(os.path.dirname(DATA_LOCAL), exist_ok=True)
    base.to_parquet(DATA_LOCAL, index=False)
    print(f"     salvo local -> {os.path.relpath(DATA_LOCAL)}")

    if salvar_s3:
        try:
            uri = gravar_parquet_s3(base, f"{config.TC3_FEATURES}/base_analitica.parquet")
            print(f"     salvo S3 -> {uri}")
        except Exception as e:  # perfil CLI pode não ter PutObject; segue com o local
            print(f"     [aviso] não foi possível gravar no S3 ({type(e).__name__}: {e}). "
                  f"Base local disponível para prosseguir.")
    return base


if __name__ == "__main__":
    montar_base()
