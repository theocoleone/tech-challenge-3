"""Acesso a dados: Parquet particionado no S3 e queries no BigQuery (Base dos Dados).

Reaproveita o padrão do TC2 (`dashboard/gerar_dashboard.py`): lê Parquet direto do
S3 com boto3/pandas, reconstruindo as colunas de partição Hive a partir do path.
"""
import io
import re

import boto3
import pandas as pd

from src import config


def _boto_session() -> boto3.Session:
    """Sessão boto3: role IAM no SageMaker, profile nomeado localmente."""
    if config.rodando_no_sagemaker():
        return boto3.Session(region_name=config.REGION)
    return boto3.Session(profile_name=config.PROFILE, region_name=config.REGION)


def ler_parquet_s3(prefix: str, bucket: str = config.BUCKET, columns=None) -> pd.DataFrame:
    """Lê todos os `.parquet` sob `prefix`, reconstruindo partições Hive (ex.: `ano=2024`).

    Ex.: ler_parquet_s3("silver/alunos", columns=["id_aluno", "alfabetizado"])
    `columns` limita a leitura às colunas do arquivo (economiza memória); a coluna de
    partição (ex.: `ano`) é reconstruída do path mesmo fora dessa lista.
    """
    s3 = _boto_session().client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    partes = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".parquet"):
                continue
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            df = pd.read_parquet(io.BytesIO(body), columns=columns)
            for col, val in re.findall(r"/(\w+)=([^/]+)/", key):
                df[col] = val
            partes.append(df)
    if not partes:
        raise FileNotFoundError(f"Nenhum parquet em s3://{bucket}/{prefix}")
    return pd.concat(partes, ignore_index=True)


def gravar_parquet_s3(df: pd.DataFrame, key: str, bucket: str = config.BUCKET) -> str:
    """Grava um DataFrame como Parquet (snappy) em s3://bucket/key. Retorna o URI."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="snappy")
    buf.seek(0)
    _boto_session().client("s3").put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    return f"s3://{bucket}/{key}"


def query_bigquery(
    sql: str,
    billing_project: str = config.GCP_BILLING_PROJECT,
    maximum_bytes_billed: int = 20 * 1024 ** 3,  # teto de custo: 20 GB por query
) -> pd.DataFrame:
    """Executa SQL no BigQuery (Base dos Dados) via ADC local; fatura em `billing_project`.

    `maximum_bytes_billed` aborta a query se ela fosse varrer mais que o teto — proteção
    contra scan acidental caro nas tabelas grandes (ex.: Censo Escolar).
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=billing_project)
    job_config = bigquery.QueryJobConfig(maximum_bytes_billed=maximum_bytes_billed)
    return client.query(sql, location="US", job_config=job_config).to_dataframe()
