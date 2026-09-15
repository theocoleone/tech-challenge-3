"""Configuração central: caminhos locais, bucket S3, prefixes, profile AWS e alvo.

Leitura de dados aponta para o data lake do Tech Challenge 2 (conta 286958704145);
tudo que o TC3 grava fica sob `tc3/`.
"""
import os

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(RAIZ, "data")
REPORTS_DIR = os.path.join(RAIZ, "reports")
FIGURAS_DIR = os.path.join(REPORTS_DIR, "figuras")
MODELS_DIR = os.path.join(RAIZ, "models")
BASE_LOCAL = os.path.join(DATA_DIR, "base_analitica.parquet")
AMOSTRA_LOCAL = os.path.join(DATA_DIR, "amostra.parquet")

BUCKET = "fiap-tc2-286958704145"
PROFILE = "fiap-tech-challenge"
REGION = "us-east-1"

SILVER_ALUNOS = "silver/alunos"
GOLD_INDICADOR_MUNICIPIO = "gold/indicador_municipio"

TC3_FEATURES = "tc3/features"
TC3_MODELS = "tc3/models"
TC3_REPORTS = "tc3/reports"

GCP_BILLING_PROJECT = os.environ.get("GCP_BILLING_PROJECT", "aist-tech-challenge-2")

COLUNA_ALVO = "alfabetizado"

# Medidas no exame ou derivadas do alvo. Nunca entram como feature.
COLUNAS_LEAKAGE = [
    "proficiencia", "alfabetizado", "alfabetizado_desc",
    "presenca", "preenchimento_caderno", "percentual_participacao",
    "taxa_alfabetizacao", "media_portugues",
    "gap_para_meta_2030", "ranking_na_uf", "atingimento_meta_2030_pct", "variacao_pp",
] + [f"proporcao_aluno_nivel_{i}" for i in range(9)]


def base_local() -> str:
    """Base completa se existir; senão a amostra versionada. `TC3_BASE` força um caminho."""
    forcada = os.environ.get("TC3_BASE")
    if forcada:
        return os.path.abspath(forcada)
    if os.path.exists(BASE_LOCAL):
        return BASE_LOCAL
    if os.path.exists(AMOSTRA_LOCAL):
        print(f"[aviso] {os.path.relpath(BASE_LOCAL, RAIZ)} ausente; usando a amostra "
              f"{os.path.relpath(AMOSTRA_LOCAL, RAIZ)} (números diferem da base completa)")
        return AMOSTRA_LOCAL
    raise FileNotFoundError(
        "Nenhuma base em data/. Rode `python -m src.preprocessing.montar_base` "
        "(exige acesso ao S3/BigQuery) ou restaure data/amostra.parquet do repositório."
    )
