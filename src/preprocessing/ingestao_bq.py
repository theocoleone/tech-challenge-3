"""Ingestão das fontes externas (BigQuery / Base dos Dados) para enriquecer a base.

Duas famílias, ambas de pré-condições (nada medido no exame):
  - Escola: Censo Escolar INEP agregado por (id_municipio, ano) — o id_escola do SAEB é
    anonimizado e não casa com o código INEP, então não há join escola a escola.
  - Município: Atlas ADH 2010 + IVS 2010 + população/PIB 2023 + diretório IBGE.
"""
import pandas as pd

from src.data_access import query_bigquery

# Proporção de escolas de anos iniciais com cada item + média de docentes/matrículas/salas.
_ESCOLA_BIN = [
    "agua_rede_publica", "energia_rede_publica", "esgoto_rede_publica",
    "internet", "banda_larga", "biblioteca",
    "laboratorio_informatica", "laboratorio_ciencias",
    "quadra_esportes", "refeitorio", "alimentacao", "area_verde", "patio_coberto",
]
_ESCOLA_QTD = [
    "quantidade_docente_fundamental_anos_iniciais",
    "quantidade_matricula_fundamental_anos_iniciais",
    "quantidade_sala_utilizada",
]  # quantidade_computador_aluno e quantidade_funcionario vêm 100% nulos em 2023/2024


def carregar_censo_escolar_municipio(anos=(2023, 2024)) -> pd.DataFrame:
    """Infra das escolas públicas de anos iniciais, agregada por (id_municipio, ano). Prefixo `esc_`.

    Só rede pública (rede != '4'): os alunos avaliados no SAEB Alfabetização são de escolas públicas,
    então o contexto escolar do município é medido nas mesmas redes.
    """
    prop = ",\n               ".join(
        f"AVG(CAST({c} AS FLOAT64)) AS esc_prop_{c}" for c in _ESCOLA_BIN
    )
    media = ",\n               ".join(
        f"AVG({c}) AS esc_media_{c}" for c in _ESCOLA_QTD
    )
    anos_sql = ", ".join(str(a) for a in anos)
    sql = f"""
        SELECT id_municipio, ano,
               COUNT(*) AS esc_n_escolas,
               AVG(CASE WHEN tipo_localizacao = '2' THEN 1.0 ELSE 0.0 END) AS esc_prop_rural,
               {prop},
               {media}
        FROM `basedosdados.br_inep_censo_escolar.escola`
        WHERE ano IN ({anos_sql})
          AND etapa_ensino_fundamental_anos_iniciais = 1
          AND rede != '4'
        GROUP BY id_municipio, ano
    """
    df = query_bigquery(sql)
    df["id_municipio"] = df["id_municipio"].astype(str)
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
    e = carregar_censo_escolar_municipio()
    print(f"censo escolar (município-ano): {len(e):,} linhas x {e.shape[1]} colunas")
    m = carregar_socioeconomico_municipio()
    print(f"socioeconômico municipal: {len(m):,} linhas x {m.shape[1]} colunas")
