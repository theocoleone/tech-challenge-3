"""Aplicação estratégica (grão município) — as perguntas de negócio do TC3.

1) Quais regiões têm padrões semelhantes: clustering socioeconômico + escolar dos municípios
   (imputação -> padronização -> PCA 95% -> KMeans, k por silhouette; perfil também para k=3).
2) Quais municípios têm maior risco: ranking pela taxa de 2024 entre alunos presentes (a mesma
   definição do indicador da Gold), com a taxa incluindo ausentes, a participação e o risco previsto
   pelo modelo (média da probabilidade de "não alfabetizado") ao lado. Município-anos sem
   participação ou alunos suficientes saem do ranking e viram uma lista à parte.
3) Quem pode não atingir a meta 2030 (Gold): três cenários em vez de uma extrapolação única —
   manter a taxa de 2024, seguir a tendência da UF, ou a tendência própria encolhida para a da UF
   (peso n/(n+k)). A variação anual é limitada a ±5 p.p. (acima disso é choque, não tendência) e
   a projeção fica em [0, 100].

Rodar da raiz do repo:  python notebooks/03_aplicacao_estrategica.py
"""
import glob
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.data_access import ler_parquet_s3, publicar_artefatos
from src.modeling.pipeline import separar_features

warnings.filterwarnings("ignore")

REPORTS = config.REPORTS_DIR
FIG = config.FIGURAS_DIR
os.makedirs(FIG, exist_ok=True)
SEED = 42
MIN_ALUNOS = 50
PARTICIPACAO_MIN = 0.5
PARTICIPACAO_BAIXA = 0.8
K_ENCOLHIMENTO = 200
LIMITE_VEL = 5.0
HORIZONTE = 2030 - 2024

SOCIO = ["idhm", "idhm_e", "idhm_l", "idhm_r", "renda_pc", "indice_gini", "prop_pobreza",
         "expectativa_vida", "taxa_analfabetismo_15_mais", "taxa_freq_liquida_fundamental",
         "taxa_agua_encanada", "taxa_energia_eletrica", "taxa_coleta_lixo", "ivs",
         "ivs_infraestrutura_urbana", "ivs_capital_humano", "ivs_renda_trabalho",
         "pib_per_capita", "populacao"]


def salvar_json(obj, nome):
    with open(os.path.join(REPORTS, nome), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def risco_previsto(base):
    """Probabilidade média de 'não alfabetizado' por município e ano, segundo o campeão."""
    caminho = os.path.join(config.MODELS_DIR, "campeao.joblib")
    if not os.path.exists(caminho):
        print("[aviso] models/campeao.joblib ausente; risco_previsto fica vazio (rode 02_modelagem.py)")
        return pd.Series(np.nan, index=base.index)
    modelo = joblib.load(caminho)
    X, *_ = separar_features(base)
    proba = np.empty(len(X))
    for ini in range(0, len(X), 500_000):
        proba[ini:ini + 500_000] = modelo.predict_proba(X.iloc[ini:ini + 500_000])[:, 1]
    return pd.Series(1 - proba, index=base.index)


def montar_municipios(base):
    """Grão município: identificação, alvo observado, presença, risco previsto e covariáveis."""
    base = base.copy()
    base["p_nao_alf"] = risco_previsto(base)
    esc = [c for c in base.columns if c.startswith("esc_") and base[c].notna().any()]
    feats = SOCIO + esc
    g = base.groupby("id_municipio")
    muni = g.agg(nome_municipio=("nome_municipio", "first"), sigla_uf=("sigla_uf", "first"),
                 nome_regiao=("nome_regiao", "first"), n_alunos=("alfabetizado", "size"),
                 taxa_alf=("alfabetizado", "mean"), presenca=("presenca", "mean"),
                 risco_previsto=("p_nao_alf", "mean"))
    muni = muni.join(g[feats].mean())
    for ano in sorted(base["ano"].unique()):
        b = base[base["ano"] == ano]
        ga = b.groupby("id_municipio")
        muni[f"n_alunos_{ano}"] = ga.size()
        muni[f"taxa_alf_{ano}"] = ga["alfabetizado"].mean()
        muni[f"presenca_{ano}"] = ga["presenca"].mean()
        muni[f"taxa_presentes_{ano}"] = b[b["presenca"] == 1].groupby("id_municipio")["alfabetizado"].mean()
        muni[f"risco_previsto_{ano}"] = ga["p_nao_alf"].mean()
    muni["populacao"] = np.log1p(muni["populacao"])
    return muni, feats


def clusterizar(muni, feats):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=SEED)),
    ])
    Z = pipe.fit_transform(muni[feats])
    print(f"PCA: {Z.shape[1]} componentes p/ 95% da variância")
    sils, inercias = {}, {}
    for k in range(2, 8):
        km = KMeans(k, n_init=10, random_state=SEED).fit(Z)
        sils[k] = silhouette_score(Z, km.labels_, sample_size=3000, random_state=SEED)
        inercias[k] = km.inertia_
    print("silhouette por k:", {k: round(v, 3) for k, v in sils.items()})
    k = max(sils, key=sils.get)
    print(f"k escolhido (maior silhouette): {k} — estrutura {'fraca' if sils[k] < 0.5 else 'razoável'} "
          f"(silhouette {sils[k]:.2f})")
    muni["cluster"] = KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)
    muni["cluster_k3"] = KMeans(3, n_init=10, random_state=SEED).fit_predict(Z)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    pd.Series(sils).plot(marker="o", ax=ax[0]); ax[0].set_xlabel("k"); ax[0].set_ylabel("silhouette")
    ax[0].set_title("Silhouette por k")
    pd.Series(inercias).plot(marker="o", ax=ax[1], color="#a6462e"); ax[1].set_xlabel("k"); ax[1].set_ylabel("inércia")
    ax[1].set_title("Inércia (cotovelo)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "10_silhouette.png"), dpi=110); plt.close()

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=muni["cluster"], cmap="tab10", s=8, alpha=0.6)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(f"Municípios em {k} clusters (PCA)")
    plt.colorbar(sc, label="cluster")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "11_pca_clusters.png"), dpi=110); plt.close()
    return muni, k, {int(a): round(float(b), 4) for a, b in sils.items()}


def perfil_clusters(muni, col):
    perfil = muni.groupby(col).agg(
        n_municipios=("taxa_alf", "size"), taxa_alf=("taxa_alf", "mean"),
        risco_previsto=("risco_previsto", "mean"), idhm=("idhm", "mean"), renda_pc=("renda_pc", "mean"),
        ivs=("ivs", "mean"), prop_pobreza=("prop_pobreza", "mean"),
        esc_prop_internet=("esc_prop_internet", "mean"), esc_prop_rural=("esc_prop_rural", "mean"),
    ).round(3).sort_values("taxa_alf")
    nomes = {2: ["vulnerável", "desenvolvido"], 3: ["vulnerável", "intermediário", "desenvolvido"]}
    rotulos = nomes.get(len(perfil), [f"grupo {i + 1}" for i in range(len(perfil))])
    perfil["rotulo"] = rotulos
    perfil["ufs_predominantes"] = [
        ", ".join(muni.loc[muni[col] == c, "sigla_uf"].value_counts().head(4).index) for c in perfil.index
    ]
    return perfil


def projetar_metas(muni):
    gold = ler_parquet_s3(config.GOLD_INDICADOR_MUNICIPIO)
    gold["ano"] = gold["ano"].astype(int)
    taxa = gold.pivot_table(index="id_municipio", columns="ano", values="taxa_alfabetizacao")
    meta = gold[gold["ano"] == 2024].set_index("id_municipio")["meta_alfabetizacao_2030"]
    p = pd.DataFrame({"taxa_2023": taxa.get(2023), "taxa_2024": taxa.get(2024), "meta_2030": meta})
    p = p.join(muni[["nome_municipio", "sigla_uf", "n_alunos_2024", "presenca_2024", "taxa_alf_2024",
                     "taxa_presentes_2024", "cluster"]])
    comum = p.dropna(subset=["taxa_2024", "taxa_alf_2024"])
    dif_todos = (comum["taxa_2024"] - 100 * comum["taxa_alf_2024"]).abs().mean()
    dif_pres = (comum["taxa_2024"] - 100 * comum["taxa_presentes_2024"]).abs().mean()
    print(f"Gold x base ({len(comum):,} municípios): diferença média absoluta vs taxa com ausentes "
          f"{dif_todos:.2f} p.p. | vs taxa só de presentes {dif_pres:.2f} p.p.")
    p = p.drop(columns=["taxa_alf_2024", "taxa_presentes_2024"])
    confiavel = (p["n_alunos_2024"] >= MIN_ALUNOS) & (p["presenca_2024"] >= PARTICIPACAO_MIN)
    p["excluido"] = np.select(
        [p["taxa_2024"].isna(), p["taxa_2023"].isna(), p["meta_2030"].isna(), ~confiavel.fillna(False)],
        ["sem 2024", "sem 2023", "sem meta", "participação/alunos insuficientes"], default="")
    motivos = p.loc[p["excluido"] != "", "excluido"].value_counts().to_dict()
    p = p[p["excluido"] == ""].drop(columns="excluido")
    excluidos = int(len(taxa) - len(p))

    p["vel_municipio"] = (p["taxa_2024"] - p["taxa_2023"]).clip(-LIMITE_VEL, LIMITE_VEL)
    p["vel_uf"] = p.groupby("sigla_uf")["vel_municipio"].transform("mean")
    w = p["n_alunos_2024"] / (p["n_alunos_2024"] + K_ENCOLHIMENTO)
    p["vel_encolhida"] = w * p["vel_municipio"] + (1 - w) * p["vel_uf"]
    cenarios = {
        "mantem_2024": p["taxa_2024"],
        "tendencia_uf": (p["taxa_2024"] + HORIZONTE * p["vel_uf"]).clip(0, 100),
        "tendencia_encolhida": (p["taxa_2024"] + HORIZONTE * p["vel_encolhida"]).clip(0, 100),
    }
    resumo = {}
    for nome, proj in cenarios.items():
        p[f"proj_2030_{nome}"] = proj.round(2)
        p[f"gap_{nome}"] = (proj - p["meta_2030"]).round(2)
        p[f"nao_atinge_{nome}"] = proj < p["meta_2030"]
        resumo[nome] = {
            "municipios_nao_atingem": int((proj < p["meta_2030"]).sum()),
            "pct_nao_atingem": round(100 * float((proj < p["meta_2030"]).mean()), 1),
            "gap_medio_pp": round(float((proj - p["meta_2030"]).mean()), 2),
        }
    ref = "tendencia_encolhida"
    info = {
        "municipios_avaliados": int(len(p)),
        "municipios_excluidos": excluidos,
        "motivos_exclusao": motivos,
        "criterios_exclusao": f"sem 2023, 2024 ou meta na Gold, < {MIN_ALUNOS} alunos ou participação < {PARTICIPACAO_MIN:.0%} em 2024",
        "cenario_referencia": ref,
        "limite_velocidade_pp_ano": LIMITE_VEL,
        "encolhimento": f"peso do município = n/(n+{K_ENCOLHIMENTO}); o restante vem da tendência média da UF",
        "cenarios": resumo,
        "pct_nao_atingem": resumo[ref]["pct_nao_atingem"],
        "faixa_pct_nao_atingem": [min(r["pct_nao_atingem"] for r in resumo.values()),
                                  max(r["pct_nao_atingem"] for r in resumo.values())],
        "municipios_nao_atingem_meta_2030": resumo[ref]["municipios_nao_atingem"],
        "gap_medio_pp": resumo[ref]["gap_medio_pp"],
    }
    print(json.dumps(info, indent=2, ensure_ascii=False))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    titulos = {"mantem_2024": "mantém taxa de 2024", "tendencia_uf": "tendência da UF",
               "tendencia_encolhida": "tendência própria encolhida"}
    for ax, nome in zip(axes, cenarios):
        ax.hist(p[f"gap_{nome}"], bins=40, color="#4C72B0")
        ax.axvline(0, color="red", ls="--")
        ax.set_title(f"{titulos[nome]}\n{resumo[nome]['pct_nao_atingem']}% não atingem")
        ax.set_xlabel("projeção 2030 − meta (p.p.)")
    axes[0].set_ylabel("nº municípios")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "13_projecao_metas.png"), dpi=110); plt.close()
    return p, info


def main():
    base = pd.read_parquet(config.base_local())
    base["alfabetizado"] = base["alfabetizado"].astype(int)
    muni, feats = montar_municipios(base)
    ano_ref = int(base["ano"].max())
    print(f"municípios: {len(muni):,} | ano de referência: {ano_ref}")

    print("\n=== 1) CLUSTERING (regiões com padrões semelhantes) ===")
    muni, k, sils = clusterizar(muni, feats)
    perfil = perfil_clusters(muni, "cluster")
    perfil_k3 = perfil_clusters(muni, "cluster_k3")
    print(perfil.to_string())
    print("\nperfil alternativo, k=3:\n" + perfil_k3.to_string())
    perfil.plot.bar(y="taxa_alf", legend=False, title="Taxa de alfabetização média por cluster")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "12_taxa_por_cluster.png"), dpi=110); plt.close()

    print(f"\n=== 2) RANKING DE RISCO ({ano_ref}; >= {MIN_ALUNOS} alunos e participação >= {PARTICIPACAO_MIN:.0%}) ===")
    n, pres, taxa = f"n_alunos_{ano_ref}", f"presenca_{ano_ref}", f"taxa_alf_{ano_ref}"
    confiavel = (muni[n] >= MIN_ALUNOS) & (muni[pres] >= PARTICIPACAO_MIN)
    sem_dado = muni[~confiavel.fillna(False)].copy()
    sem_dado["motivo"] = np.select(
        [sem_dado[n].isna(), sem_dado[n] < MIN_ALUNOS], [f"sem avaliação em {ano_ref}", f"menos de {MIN_ALUNOS} alunos"],
        default=f"participação < {PARTICIPACAO_MIN:.0%}")
    indicador = f"taxa_presentes_{ano_ref}"
    colunas_risco = ["nome_municipio", "sigla_uf", "nome_regiao", n, pres, indicador, taxa,
                     f"risco_previsto_{ano_ref}", "idhm", "ivs", "cluster"]
    risco = muni[confiavel.fillna(False)][colunas_risco].sort_values(indicador).round(3)
    risco["alerta_participacao"] = risco[pres] < PARTICIPACAO_BAIXA
    print(risco.head(20).to_string())
    print(f"\n{len(sem_dado)} municípios fora do ranking: {sem_dado['motivo'].value_counts().to_dict()}")
    corr = risco[[indicador, f"risco_previsto_{ano_ref}"]].corr().iloc[0, 1]
    print(f"correlação indicador observado x risco previsto (municípios): {corr:.3f}")

    print("\n=== 3) PROJEÇÃO DE METAS 2030 (Gold, três cenários) ===")
    proj, proj_info = None, {}
    try:
        proj, proj_info = projetar_metas(muni)
    except Exception as e:
        print(f"[aviso] projeção de metas indisponível ({type(e).__name__}: {e})")

    muni.reset_index()[["id_municipio", "nome_municipio", "sigla_uf", "nome_regiao", "cluster", "cluster_k3",
                        "taxa_alf", "risco_previsto", "n_alunos", "presenca"]] \
        .round(4).to_csv(os.path.join(REPORTS, "clusters_municipios.csv"), index=False)
    perfil.to_json(os.path.join(REPORTS, "perfil_clusters.json"), indent=2, force_ascii=False)
    perfil_k3.to_json(os.path.join(REPORTS, "perfil_clusters_k3.json"), indent=2, force_ascii=False)
    salvar_json({"k_escolhido": k, "silhouette_por_k": sils, "n_features": len(feats)}, "clustering.json")
    risco.to_csv(os.path.join(REPORTS, "risco_municipios.csv"))
    risco.head(20).to_csv(os.path.join(REPORTS, "risco_top20.csv"))
    sem_dado[["nome_municipio", "sigla_uf", n, pres, "motivo"]].sort_values(["motivo", "sigla_uf"]) \
        .to_csv(os.path.join(REPORTS, "municipios_sem_dado_confiavel.csv"))
    if proj is not None:
        proj.round(3).to_csv(os.path.join(REPORTS, "metas_2030_municipios.csv"))
        salvar_json(proj_info, "projecao_metas.json")
    print("\nartefatos salvos em reports/.")
    if config.base_local() == config.BASE_LOCAL:
        publicar_artefatos(sorted(glob.glob(os.path.join(REPORTS, "*.json")) + glob.glob(os.path.join(REPORTS, "*.csv"))
                                  + glob.glob(os.path.join(REPORTS, "*.md"))), config.TC3_REPORTS)
    print("=== aplicação estratégica concluída ===")


if __name__ == "__main__":
    main()
