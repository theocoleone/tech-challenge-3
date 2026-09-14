"""Configuração central do TC3: bucket S3, prefixes, profile AWS e detecção de ambiente.

Toda a leitura de dados aponta para o data lake herdado do Tech Challenge 2
(conta AWS 286958704145). A gravação de features/modelos/relatórios do TC3 usa
prefixes dedicados sob `tc3/`.
"""
import os

# --- Data lake (TC2) ---
BUCKET = "fiap-tc2-286958704145"
PROFILE = "fiap-tech-challenge"
REGION = "us-east-1"

# Prefixes de LEITURA (Silver/Gold do TC2)
SILVER_ALUNOS = "silver/alunos"                     # grão aluno — base de treino
SILVER_DIRETORIO = "silver/diretorio_municipio"     # território limpo (sem leakage)
GOLD_INDICADOR_MUNICIPIO = "gold/indicador_municipio"
GOLD_METAS_UF = "gold/metas_vs_resultados_uf"
GOLD_EVOLUCAO_UF = "gold/evolucao_uf"

# Prefixes de ESCRITA (TC3)
TC3_FEATURES = "tc3/features"
TC3_MODELS = "tc3/models"
TC3_REPORTS = "tc3/reports"

# --- GCP / BigQuery (fontes externas via Base dos Dados) ---
# Projeto que fatura as queries. Local usa ADC do gcloud; no Glue/SageMaker,
# a chave da service account vem do Secrets Manager (gcp-service-account-bronze).
GCP_BILLING_PROJECT = os.environ.get("GCP_BILLING_PROJECT", "aist-tech-challenge-2")

# --- Alvo ---
COLUNA_ALVO = "alfabetizado"     # rótulo-fonte 0/1 (usar este, não recomputar de proficiencia)
CORTE_SAEB = 743                 # referência conceitual; proficiencia é LEAKAGE e não vira feature

# Colunas PROIBIDAS como feature — vazam o alvo (data leakage). Ver Planning §2.
COLUNAS_LEAKAGE = [
    "proficiencia", "alfabetizado", "alfabetizado_desc",
    "presenca", "preenchimento_caderno", "percentual_participacao",
    "taxa_alfabetizacao", "media_portugues",
    "gap_para_meta_2030", "ranking_na_uf", "atingimento_meta_2030_pct", "variacao_pp",
] + [f"proporcao_aluno_nivel_{i}" for i in range(9)]


def rodando_no_sagemaker() -> bool:
    """True em ambiente SageMaker (credenciais via role IAM); False local (usa PROFILE)."""
    return (
        any(k in os.environ for k in ("SM_CURRENT_HOST", "SAGEMAKER_INTERNAL_IMAGE_URI"))
        or os.path.exists("/opt/ml")
    )
