"""EDA — Tech Challenge Fase 3 (predição de alfabetização, grão aluno).

Análise exploratória da base analítica (`data/base_analitica.parquet`): balanceamento
do alvo, valores faltantes, distribuições, correlações com o alvo e desigualdades por
categoria. Salva as figuras em `reports/figuras/` e imprime um resumo textual.

Rodar da raiz do repo:  python notebooks/01_eda.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config
from src.evaluation.exportar_agregados import exportar_agregados

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

FIG = config.FIGURAS_DIR
os.makedirs(FIG, exist_ok=True)

ALVO = config.COLUNA_ALVO
IDS = ["id_aluno", "id_municipio", "id_escola"]
CATEGORICAS = ["rede", "sigla_uf", "nome_regiao", "nome_mesorregiao",
               "amazonia_legal", "capital_uf", "ano"]


def salvar(nome):
    caminho = os.path.join(FIG, nome)
    plt.tight_layout()
    plt.savefig(caminho, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"   figura -> reports/figuras/{nome}")


def main():
    base = config.base_local()
    df = pd.read_parquet(base)
    numericas = [c for c in df.columns if c not in IDS + CATEGORICAS + [ALVO]]
    print(f"\n=== BASE === {df.shape[0]:,} linhas x {df.shape[1]} colunas")
    print(f"{len(numericas)} numéricas | {len(CATEGORICAS)} categóricas | alvo={ALVO}")

    # 1) Balanceamento do alvo (geral e por ano)
    print("\n=== ALVO (alfabetizado) ===")
    print(df[ALVO].value_counts(normalize=True).round(4).to_string())
    print("\ntaxa de alfabetização por ano:")
    print(df.groupby("ano")[ALVO].mean().round(4).to_string())
    df.groupby("ano")[ALVO].mean().plot.bar(title="Taxa de alfabetização por ano", ylim=(0, 1))
    salvar("01_alvo_por_ano.png")

    # 2) Valores faltantes
    print("\n=== NULOS (colunas com faltantes) ===")
    nulos = df[numericas + CATEGORICAS].isna().mean().sort_values(ascending=False)
    print(nulos[nulos > 0].round(4).to_string() or "  (nenhum)")

    # 3) Distribuições de numéricas-chave
    chave = ["idhm", "renda_pc", "indice_gini", "ivs", "pib_per_capita",
             "esc_prop_internet", "esc_media_quantidade_docente_fundamental_anos_iniciais"]
    chave = [c for c in chave if c in df.columns]
    print("\n=== DESCRIBE (numéricas-chave) ===")
    print(df[chave].describe().T.round(2).to_string())
    df[chave].astype("float64").hist(bins=40, figsize=(14, 8))
    salvar("02_distribuicoes.png")

    # 4) Correlação das numéricas com o alvo
    corr = df[numericas + [ALVO]].astype("float64").corr()[ALVO].drop(ALVO).sort_values()
    print("\n=== CORRELAÇÃO COM O ALVO — top 8 negativas ===")
    print(corr.head(8).round(3).to_string())
    print("=== top 8 positivas ===")
    print(corr.tail(8).round(3).to_string())
    corr.plot.barh(figsize=(7, 11), title="Correlação (Pearson) com alfabetizado")
    salvar("03_correlacao_alvo.png")

    # heatmap das correlações entre as numéricas-chave + alvo
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[chave + [ALVO]].astype("float64").corr(), annot=True, fmt=".2f",
                cmap="RdBu_r", center=0)
    plt.title("Correlação entre variáveis-chave")
    salvar("04_heatmap.png")

    # 5) Desigualdade por categoria
    print("\n=== TAXA DE ALFABETIZAÇÃO POR REGIÃO ===")
    por_regiao = df.groupby("nome_regiao")[ALVO].mean().sort_values()
    print(por_regiao.round(4).to_string())
    por_regiao.plot.bar(title="Taxa de alfabetização por região", ylim=(0, 1))
    salvar("05_taxa_por_regiao.png")

    print("\n=== TAXA POR REDE DE ENSINO ===")
    print(df.groupby("rede")[ALVO].mean().round(4).to_string())

    print("\n=== TAXA POR UF (top e bottom 5) ===")
    por_uf = df.groupby("sigla_uf")[ALVO].mean().sort_values()
    print("piores:\n", por_uf.head(5).round(4).to_string())
    print("melhores:\n", por_uf.tail(5).round(4).to_string())

    # 6) Relação monotônica IDHM x alvo
    df["idhm_decil"] = pd.qcut(df["idhm"], 10, labels=False, duplicates="drop")
    print("\n=== TAXA POR DECIL DE IDHM ===")
    por_decil = df.groupby("idhm_decil")[ALVO].mean()
    print(por_decil.round(4).to_string())
    por_decil.plot(marker="o", title="Alfabetização por decil de IDHM", ylim=(0, 1))
    salvar("06_idhm_decil.png")

    exportar_agregados(base)
    print("\n=== EDA concluída ===")


if __name__ == "__main__":
    main()
