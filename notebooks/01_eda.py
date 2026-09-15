"""EDA — Tech Challenge Fase 3 (predição de alfabetização, grão aluno).

Balanceamento do alvo, cobertura por UF e ano, presença na prova (sensibilidade do rótulo),
valores faltantes, distribuições, correlações e desigualdades por território. Salva as figuras
em `reports/figuras/`, o resumo com as hipóteses em `reports/eda_resumo.md` e os agregados
usados pelo dashboard em `reports/agregados_base.json`.

Rodar da raiz do repo:  python notebooks/01_eda.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.evaluation.exportar_agregados import exportar_agregados

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

FIG = config.FIGURAS_DIR
os.makedirs(FIG, exist_ok=True)
RESUMO = os.path.join(config.REPORTS_DIR, "eda_resumo.md")

ALVO = config.COLUNA_ALVO
IDS = ["id_aluno", "id_municipio", "nome_municipio"]
CATEGORICAS = ["rede", "sigla_uf", "nome_regiao", "nome_mesorregiao",
               "amazonia_legal", "capital_uf", "ano"]
NAO_FEATURES = IDS + CATEGORICAS + [ALVO, "presenca"]

_md = []


def md(texto=""):
    _md.append(texto)


def tabela_md(df, fmt="{:.3f}"):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    cols = [str(c) for c in df.columns]
    inteira = [pd.api.types.is_integer_dtype(df[c]) for c in df.columns]
    linhas = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
              "|" + "---|" * (len(cols) + 1)]
    for idx, row in zip(df.index, df.itertuples(index=False)):
        vals = [f"{v:,}" if i else (fmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
                for v, i in zip(row, inteira)]
        linhas.append("| " + " | ".join([str(idx)] + vals) + " |")
    return "\n".join(linhas)


def salvar(nome):
    caminho = os.path.join(FIG, nome)
    plt.tight_layout()
    plt.savefig(caminho, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"   figura -> reports/figuras/{nome}")


def main():
    base = config.base_local()
    df = pd.read_parquet(base)
    numericas = [c for c in df.columns if c not in NAO_FEATURES and df[c].notna().any()]
    md("# Resumo da análise exploratória\n")
    anos = ", ".join(str(a) for a in sorted(df["ano"].unique()))
    md(f"Base: `{os.path.relpath(base, config.RAIZ)}` — {len(df):,} alunos, {df.shape[1]} colunas, "
       f"{df['id_municipio'].nunique():,} municípios, anos {anos}.")
    print(f"\n=== BASE === {df.shape[0]:,} linhas x {df.shape[1]} colunas | "
          f"{len(numericas)} numéricas | {len(CATEGORICAS)} categóricas")

    # 1) Alvo
    taxa = df[ALVO].mean()
    por_ano = df.groupby("ano")[ALVO].mean()
    print(f"\n=== ALVO === taxa geral {taxa:.4f}")
    print(por_ano.round(4).to_string())
    md("## 1. Alvo\n")
    md(f"Taxa de alfabetização geral: **{taxa:.1%}** (alvo balanceado: {1 - taxa:.1%} de não alfabetizados).\n")
    md(tabela_md(por_ano.rename("taxa_alfabetizacao").to_frame().assign(
        n_alunos=df.groupby("ano").size()), "{:.4f}"))
    por_ano.plot.bar(title="Taxa de alfabetização por ano", ylim=(0, 1))
    salvar("01_alvo_por_ano.png")

    # 2) Cobertura UF x ano
    cob = df.pivot_table(index="sigla_uf", columns="ano", values=ALVO, aggfunc="size").fillna(0).astype(int)
    sem_2023 = cob.index[cob[2023] == 0].tolist() if 2023 in cob else []
    print("\n=== COBERTURA UF x ANO ===")
    print(cob.to_string())
    print("UFs sem 2023:", sem_2023)
    md("\n## 2. Cobertura por UF e ano\n")
    md(f"UFs presentes só em 2024: **{', '.join(sem_2023) or 'nenhuma'}** "
       f"({int(cob.loc[sem_2023, 2024].sum()) if sem_2023 else 0:,} alunos). A comparação 2023→2024 e a "
       f"variável `ano` ficam confundidas com a composição de UFs.\n")
    md(tabela_md(cob, "{:.0f}"))
    plt.figure(figsize=(5, 9))
    sns.heatmap(cob, annot=True, fmt=",d", cmap="Blues", cbar=False)
    plt.title("Alunos avaliados por UF e ano")
    salvar("14_cobertura_uf_ano.png")

    # 3) Presença e sensibilidade do rótulo
    presenca = df["presenca"].mean()
    taxa_presentes = df.loc[df["presenca"] == 1, ALVO].mean()
    ausentes_alf = df.loc[df["presenca"] == 0, ALVO].mean()
    part_muni = df.groupby(["id_municipio", "ano"])["presenca"].mean()
    print(f"\n=== PRESENÇA === {presenca:.1%} presentes | taxa entre presentes {taxa_presentes:.1%} | "
          f"taxa entre ausentes {ausentes_alf:.1%}")
    md("\n## 3. Presença na prova e sensibilidade do rótulo\n")
    md(f"- {presenca:.1%} dos alunos estiveram presentes; **todo aluno ausente está rotulado como não "
       f"alfabetizado** (taxa entre ausentes = {ausentes_alf:.0%}).")
    md(f"- Taxa com o rótulo da fonte: **{taxa:.1%}**; só entre presentes: **{taxa_presentes:.1%}**. "
       f"A diferença ({taxa_presentes - taxa:+.1%}) é o efeito do rótulo, não de aprendizagem.")
    md(f"- Município-anos com participação abaixo de 50%: {int((part_muni < 0.5).sum())} de "
       f"{len(part_muni):,}; abaixo de 80%: {int((part_muni < 0.8).sum())}.\n")
    uf_pres = df.groupby("sigla_uf").agg(taxa=(ALVO, "mean"), presenca=("presenca", "mean"))
    uf_pres["taxa_presentes"] = df[df["presenca"] == 1].groupby("sigla_uf")[ALVO].mean()
    uf_pres = uf_pres.sort_values("taxa")
    md(tabela_md(uf_pres, "{:.3f}"))
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(uf_pres))
    ax.bar(x - 0.2, uf_pres["taxa"], 0.4, label="rótulo da fonte (ausente = 0)")
    ax.bar(x + 0.2, uf_pres["taxa_presentes"], 0.4, label="só presentes")
    ax.set_xticks(x); ax.set_xticklabels(uf_pres.index, rotation=0, fontsize=8)
    ax.set_ylim(0, 1); ax.set_title("Taxa de alfabetização por UF: efeito dos ausentes no rótulo"); ax.legend()
    salvar("17_presenca_vs_alvo.png")

    # 4) Variação 2023 -> 2024 por UF
    var_uf = df.pivot_table(index="sigla_uf", columns="ano", values=ALVO, aggfunc="mean")
    if 2023 in var_uf and 2024 in var_uf:
        var_uf.columns = [f"taxa_{a}" for a in var_uf.columns]
        var_uf["variacao_pp"] = (var_uf["taxa_2024"] - var_uf["taxa_2023"]) * 100
        pres = df.pivot_table(index="sigla_uf", columns="ano", values="presenca", aggfunc="mean")
        pres.columns = [f"presenca_{a}" for a in pres.columns]
        tp = df[df["presenca"] == 1].pivot_table(index="sigla_uf", columns="ano", values=ALVO, aggfunc="mean")
        tp.columns = [f"taxa_presentes_{a}" for a in tp.columns]
        var_uf = var_uf.join(pres).join(tp).dropna().sort_values("variacao_pp")
        pior = var_uf.iloc[0]
        print("\n=== VARIAÇÃO 2023->2024 POR UF (p.p.) ===")
        print(var_uf.round(3).to_string())
        md("\n## 4. Variação 2023 → 2024 por UF\n")
        md(f"Mediana da variação: {var_uf['variacao_pp'].median():+.1f} p.p.; extremos "
           f"{var_uf.index[0]} ({pior['variacao_pp']:+.1f} p.p.) e {var_uf.index[-1]} "
           f"({var_uf['variacao_pp'].iloc[-1]:+.1f} p.p.). Separar o efeito da presença do efeito da "
           f"proficiência muda a leitura: em {var_uf.index[0]} a presença foi de {pior['presenca_2023']:.0%} "
           f"para {pior['presenca_2024']:.0%} e a taxa **entre presentes** de {pior['taxa_presentes_2023']:.1%} "
           f"para {pior['taxa_presentes_2024']:.1%}. Quando a taxa entre presentes cai junto, a queda é de "
           f"resultado medido; quando só a presença cai, é participação. Em qualquer caso, a diferença de um "
           f"único ano não é uma tendência.\n")
        md(tabela_md(var_uf, "{:.3f}"))
        cores = np.where(var_uf["variacao_pp"] < 0, "#a6462e", "#1f5c8c")
        var_uf["variacao_pp"].plot.barh(figsize=(7, 8), color=cores, title="Variação 2023→2024 da taxa (p.p.)")
        plt.axvline(0, color="gray", lw=1)
        salvar("15_variacao_uf.png")

    # 5) Rede e tamanho dos municípios
    rede = df.groupby("rede").agg(n=(ALVO, "size"), taxa=(ALVO, "mean"))
    n_muni = df.groupby("id_municipio").size()
    print("\n=== REDE ===\n" + rede.round(3).to_string())
    print(f"\n=== ALUNOS POR MUNICÍPIO === mediana {n_muni.median():.0f} | <50 alunos: {(n_muni < 50).sum()} municípios")
    md("\n## 5. Rede de ensino e porte dos municípios\n")
    md(tabela_md(rede, "{:.3f}"))
    md(f"\nAlunos por município: mediana {n_muni.median():.0f}, máximo {n_muni.max():,}; "
       f"**{(n_muni < 50).sum()} municípios têm menos de 50 alunos** avaliados nos dois anos — taxas "
       f"instáveis, por isso o ranking de risco exige mínimo de alunos.\n")
    np.log10(n_muni).plot.hist(bins=40, title="Alunos avaliados por município (log10)")
    plt.xlabel("log10(alunos)")
    salvar("16_alunos_por_municipio.png")
    dup_ano = int(df.duplicated(["id_aluno", "ano"]).sum())
    dup_geral = int(df["id_aluno"].duplicated().sum())
    md(f"Duplicidade de `id_aluno` dentro do mesmo ano: {dup_ano:,}. Entre anos há {dup_geral:,} "
       f"colisões de id ({dup_geral / len(df):.0%} das linhas): o identificador é anonimizado por edição, "
       f"então não existe painel aluno-a-aluno entre 2023 e 2024.")

    # 6) Nulos
    nulos = df[numericas + CATEGORICAS].isna().mean().sort_values(ascending=False)
    nulos = nulos[nulos > 0]
    print("\n=== NULOS ===")
    print(nulos.round(4).to_string() if len(nulos) else "  (nenhum)")
    md("\n## 6. Valores faltantes\n")
    md(tabela_md(nulos.rename("proporcao_nula"), "{:.4f}") if len(nulos) else "Nenhuma coluna com faltantes.")
    md("\nImputação por mediana no pipeline cobre os casos residuais; nenhuma coluna exige tratamento especial.")

    # 7) Distribuições e correlações
    chave = ["idhm", "renda_pc", "indice_gini", "ivs", "pib_per_capita",
             "esc_prop_internet", "esc_media_quantidade_docente_fundamental_anos_iniciais"]
    chave = [c for c in chave if c in df.columns]
    print("\n=== DESCRIBE (numéricas-chave) ===")
    print(df[chave].describe().T.round(2).to_string())
    df[chave].astype("float64").hist(bins=40, figsize=(14, 8))
    salvar("02_distribuicoes.png")

    corr = df[numericas + [ALVO]].astype("float64").corr()[ALVO].drop(ALVO).sort_values()
    print("\n=== CORRELAÇÃO COM O ALVO ===")
    print(corr.round(3).to_string())
    md("\n## 7. Correlação (Pearson) das numéricas com o alvo\n")
    md("Correlações individuais são fracas (|r| < 0,15): o contexto municipal explica pouco da variação "
       "entre alunos, o que antecipa o teto de desempenho do modelo.\n")
    md(tabela_md(pd.concat([corr.head(8), corr.tail(8)]).rename("r"), "{:.3f}"))
    corr.plot.barh(figsize=(7, 11), title="Correlação (Pearson) com alfabetizado")
    salvar("03_correlacao_alvo.png")
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[chave + [ALVO]].astype("float64").corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("Correlação entre variáveis-chave")
    salvar("04_heatmap.png")

    # 8) Desigualdade territorial
    por_regiao = df.groupby("nome_regiao")[ALVO].mean().sort_values()
    por_uf = df.groupby("sigla_uf")[ALVO].mean().sort_values()
    print("\n=== POR REGIÃO ===\n" + por_regiao.round(4).to_string())
    print("\n=== POR UF === piores:", por_uf.head(5).round(3).to_dict(), "| melhores:", por_uf.tail(5).round(3).to_dict())
    md("\n## 8. Desigualdade territorial\n")
    md(tabela_md(por_regiao.rename("taxa").to_frame(), "{:.3f}"))
    md(f"\nUF: de **{por_uf.index[0]} ({por_uf.iloc[0]:.1%})** a **{por_uf.index[-1]} ({por_uf.iloc[-1]:.1%})**. "
       f"O {por_uf.index[-1]} descola do resto do país ({por_uf.iloc[-1] - por_uf.iloc[-2]:+.1%} sobre o segundo, "
       f"{por_uf.index[-2]}).")
    por_regiao.plot.bar(title="Taxa de alfabetização por região", ylim=(0, 1))
    salvar("05_taxa_por_regiao.png")

    df["idhm_decil"] = pd.qcut(df["idhm"], 10, labels=False, duplicates="drop")
    melhor_uf = por_uf.index[-1]
    decil = pd.DataFrame({
        "idhm_medio": df.groupby("idhm_decil")["idhm"].mean(),
        "taxa": df.groupby("idhm_decil")[ALVO].mean(),
        "taxa_presentes": df[df["presenca"] == 1].groupby("idhm_decil")[ALVO].mean(),
        f"taxa_sem_{melhor_uf}": df[df["sigla_uf"] != melhor_uf].groupby("idhm_decil")[ALVO].mean(),
    })
    print("\n=== POR DECIL DE IDHM ===\n" + decil.round(4).to_string())
    md(f"\nPor decil de IDHM a taxa vai de {decil['taxa'].iloc[0]:.1%} (1º decil) a "
       f"{decil['taxa'].iloc[-1]:.1%} (10º), com máximo de {decil['taxa'].max():.1%}: o gradiente "
       f"socioeconômico existe só na base da distribuição e a curva fica plana a partir do 3º decil. "
       f"Sem o {melhor_uf} (IDHM médio-baixo e a maior taxa do país) o degrau inicial fica mais nítido.\n")
    md(tabela_md(decil, "{:.3f}"))
    decil[["taxa", "taxa_presentes", f"taxa_sem_{melhor_uf}"]].plot(
        marker="o", title="Alfabetização por decil de IDHM", ylim=(0, 1))
    plt.xlabel("decil de IDHM (0 = mais baixo)")
    salvar("06_idhm_decil.png")

    # 9) Hipóteses e decisões
    md("\n## 9. Hipóteses e decisões de modelagem derivadas da EDA\n")
    md(f"- **H1 — A geografia domina o sinal.** UF e região separam mais as taxas do que qualquer "
       f"indicador socioeconômico ({por_uf.index[-1]} {por_uf.iloc[-1]:.0%} vs {por_uf.index[0]} "
       f"{por_uf.iloc[0]:.0%}; região {por_regiao.iloc[0]:.0%}–{por_regiao.iloc[-1]:.0%}). "
       f"Decisão: UF, região, Amazônia Legal e capital entram como categóricas; modelos de árvore para "
       f"capturar interações.")
    md("- **H2 — O contexto socioeconômico pesa pouco no indivíduo e só na base da distribuição.** "
       "Correlações |r| < 0,15 e taxa plana acima do 3º decil de IDHM. Decisão: esperar ROC-AUC modesto no "
       "grão aluno, ler o valor do modelo na interpretação e no grão município.")
    md(f"- **H3 — O rótulo carrega a ausência.** {1 - presenca:.0%} dos alunos são ausentes e contam como não "
       f"alfabetizados. Decisão: manter o rótulo da fonte no supervisionado (é o indicador que o gestor "
       f"recebe), reportar a sensibilidade sem ausentes e, na camada municipal, filtrar município-anos com "
       f"participação baixa antes de ranquear ou projetar.")
    md(f"- **H4 — Cobertura desigual entre anos.** {', '.join(sem_2023) or 'Nenhuma UF'} só aparecem em 2024. "
       f"Decisão: `ano` fica fora das features (confundido com UF e inútil para prever 2025) e comparações "
       f"anuais só entre UFs com os dois anos.")
    md("- **H5 — Variações anuais extremas em algumas UFs, por mecanismos diferentes** (queda de presença em "
       "umas, queda de resultado entre presentes em outras). Decisão: a projeção de metas não extrapola a "
       "diferença de um único ano sem limites; usa cenários, limita a taxa a [0, 100] e encolhe a tendência "
       "municipal para a da UF.")
    md("- **H6 — Features são municipais, alunos não são independentes dentro do município.** Decisão: "
       "separar treino e teste por município (GroupShuffleSplit/StratifiedGroupKFold) para medir "
       "generalização a territórios novos, e reportar também o split aleatório como comparação.")

    with open(RESUMO, "w", encoding="utf-8") as f:
        f.write("\n\n".join(bloco.strip("\n") for bloco in _md) + "\n")
    print(f"\nresumo -> {os.path.relpath(RESUMO, config.RAIZ)}")
    exportar_agregados(base)
    print("=== EDA concluída ===")


if __name__ == "__main__":
    main()
