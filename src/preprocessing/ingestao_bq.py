"""Ingestão das fontes externas (BigQuery / Base dos Dados) para enriquecer a base.

Duas famílias, todas de variáveis NÃO-vazadas (pré-condições, não saídas do exame):
  - Escola  (Censo Escolar INEP) — infraestrutura/docentes/matrículas, por (id_escola, ano).
  - Município (Atlas ADH 2010 + IVS/AVS 2010 + PIB/pop 2023 + diretório), estático por id_municipio.
"""
import pandas as pd

from src.data_access import query_bigquery

# --- Censo Escolar: subconjunto curado de features de contexto da escola ---
_ESCOLA_FEATURES = [
    "tipo_localizacao",                                   # urbana/rural
    "agua_rede_publica", "energia_rede_publica", "esgoto_rede_publica",
    "internet", "banda_larga", "biblioteca",
    "laboratorio_informatica", "laboratorio_ciencias",
    "quadra_esportes", "refeitorio", "alimentacao", "area_verde", "patio_coberto",
    "acessibilidade_inexistente",
    "quantidade_docente_fundamental_anos_iniciais",
    "quantidade_matricula_fundamental_anos_iniciais",
    "quantidade_sala_utilizada",
    "quantidade_computador_aluno",
    "quantidade_funcionario",
]


def carregar_censo_escolar(anos=(2023, 2024)) -> pd.DataFrame:
    """Features da escola por (id_escola, ano). Prefixo `esc_` evita colisão com colunas do aluno."""
    cols = ", ".join(_ESCOLA_FEATURES)
    anos_sql = ", ".join(str(a) for a in anos)
    sql = f"""
        SELECT id_escola, ano, {cols}
        FROM `basedosdados.br_inep_censo_escolar.escola`
        WHERE ano IN ({anos_sql})
    """
    df = query_bigquery(sql)
    df = df.rename(columns={c: f"esc_{c}" for c in _ESCOLA_FEATURES})
    df["id_escola"] = df["id_escola"].astype(str)
    df["ano"] = df["ano"].astype(int)
    return df


def carregar_socioeconomico_municipio() -> pd.DataFrame:
    """Covariáveis municipais estáticas (ADH 2010 + AVS 2010 + PIB/pop 2023 + diretório)."""
    sql = """
        WITH adh AS (
          SELECT id_municipio, idhm, idhm_e, idhm_l, idhm_r, renda_pc, indice_gini,
                 taxa_analfabetismo_15_mais, taxa_freq_liquida_fundamental,
                 taxa_agua_encanada, taxa_energia_eletrica, prop_pobreza,
                 taxa_coleta_lixo, expectativa_vida
          FROM `basedosdados.mundo_onu_adh.municipio`
          WHERE ano = 2010
        ),
        avs AS (
          -- AVS é grão UDH (várias linhas por município), mesmo nos totais; agregamos
          -- as UDHs para o município pela média (evita fan-out no join).
          SELECT id_municipio,
                 AVG(ivs) AS ivs,
                 AVG(ivs_infraestrutura_urbana) AS ivs_infraestrutura_urbana,
                 AVG(ivs_capital_humano) AS ivs_capital_humano,
                 AVG(ivs_renda_trabalho) AS ivs_renda_trabalho
          FROM `basedosdados.br_ipea_avs.municipio`
          WHERE ano = 2010 AND raca_cor = 'total' AND sexo = 'total' AND localizacao = 'total'
          GROUP BY id_municipio
        ),
        pop AS (
          SELECT id_municipio, populacao
          FROM `basedosdados.br_ibge_populacao.municipio`
          WHERE ano = 2023
        ),
        pib AS (
          SELECT id_municipio, pib
          FROM `basedosdados.br_ibge_pib.municipio`
          WHERE ano = 2023
        ),
        dir AS (
          SELECT id_municipio, nome_regiao, nome_mesorregiao, amazonia_legal, capital_uf
          FROM `basedosdados.br_bd_diretorios_brasil.municipio`
        )
        SELECT dir.*,
               adh.* EXCEPT (id_municipio),
               avs.* EXCEPT (id_municipio),
               pop.populacao,
               SAFE_DIVIDE(pib.pib, pop.populacao) AS pib_per_capita
        FROM dir
        LEFT JOIN adh USING (id_municipio)
        LEFT JOIN avs USING (id_municipio)
        LEFT JOIN pop USING (id_municipio)
        LEFT JOIN pib USING (id_municipio)
    """
    df = query_bigquery(sql)
    df["id_municipio"] = df["id_municipio"].astype(str)
    return df


if __name__ == "__main__":
    e = carregar_censo_escolar()
    print(f"censo escolar: {len(e):,} linhas x {e.shape[1]} colunas")
    m = carregar_socioeconomico_municipio()
    print(f"socioeconômico municipal: {len(m):,} linhas x {m.shape[1]} colunas")
