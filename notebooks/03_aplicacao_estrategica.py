"""Aplicação estratégica (grão município) — responde às perguntas de negócio do TC3.

1) Clustering de municípios ("quais regiões têm padrões semelhantes"): socioeconômico
   padronizado -> PCA(95%) -> KMeans (k por silhouette).
2) Ranking de risco educacional (menor taxa de alfabetização observada).
3) Projeção de metas 2030 (Gold): extrapola a tendência 2023->2024 e compara com a meta.

Rodar da raiz do repo:  python notebooks/03_aplicacao_estrategica.py
"""
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
from src.data_access import ler_parquet_s3

warnings.filterwarnings("ignore")

REPORTS = config.REPORTS_DIR
FIG = config.FIGURAS_DIR
os.makedirs(FIG, exist_ok=True)
SEED = 42

SOCIO = ["idhm", "idhm_e", "idhm_l", "idhm_r", "renda_pc", "indice_gini", "prop_pobreza",
         "expectativa_vida", "taxa_analfabetismo_15_mais", "taxa_freq_liquida_fundamental",
         "taxa_agua_encanada", "taxa_energia_eletrica", "taxa_coleta_lixo", "ivs",
         "ivs_infraestrutura_urbana", "ivs_capital_humano", "ivs_renda_trabalho",
         "pib_per_capita", "populacao"]


def montar_municipios(base):
    """Agrega a base ao grão município: taxa de alfabetização observada + socioeconômico + escola."""
    esc = [c for c in base.columns if c.startswith("esc_") and base[c].notna().any()]
    feats = SOCIO + esc
    agg = {c: "mean" for c in feats}
    agg["alfabetizado"] = "mean"
    muni = base.groupby("id_municipio").agg(agg).rename(columns={"alfabetizado": "taxa_alf"})
    muni["n_alunos"] = base.groupby("id_municipio").size()
    muni = muni.join(base.groupby("id_municipio")[["sigla_uf", "nome_regiao"]].first())
    muni["populacao"] = np.log1p(muni["populacao"])  # escala
    return muni, feats


def clusterizar(muni, feats):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=SEED)),
    ])
    Z = pipe.fit_transform(muni[feats])
    print(f"PCA: {Z.shape[1]} componentes p/ 95% da variância")

    # escolha de k por silhouette
    sils = {}
    for k in range(2, 8):
        lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)
        sils[k] = silhouette_score(Z, lab, sample_size=3000, random_state=SEED)
    print("silhouette por k:", {k: round(v, 3) for k, v in sils.items()})
    k = max(sils, key=sils.get)
    print(f"k escolhido (maior silhouette): {k}")

    km = KMeans(k, n_init=10, random_state=SEED).fit(Z)
    muni["cluster"] = km.labels_

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=km.labels_, cmap="tab10", s=8, alpha=0.6)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(f"Municípios em {k} clusters (PCA)")
    plt.colorbar(sc, label="cluster")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "11_pca_clusters.png"), dpi=110); plt.close()

    plt.figure(figsize=(6, 4))
    pd.Series(sils).plot(marker="o"); plt.xlabel("k"); plt.ylabel("silhouette")
    plt.title("Escolha de k (silhouette)"); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "10_silhouette.png"), dpi=110); plt.close()
    return muni, k


def main():
    base = pd.read_parquet(config.base_local())
    base["alfabetizado"] = base["alfabetizado"].astype(int)
    muni, feats = montar_municipios(base)
    print(f"municípios: {len(muni):,}")

    # 1) Clustering
    print("\n=== 1) CLUSTERING (regiões com padrões semelhantes) ===")
    muni, k = clusterizar(muni, feats)
    perfil = muni.groupby("cluster").agg(
        n_municipios=("taxa_alf", "size"),
        taxa_alf=("taxa_alf", "mean"),
        idhm=("idhm", "mean"), renda_pc=("renda_pc", "mean"),
        ivs=("ivs", "mean"), prop_pobreza=("prop_pobreza", "mean"),
        esc_prop_internet=("esc_prop_internet", "mean"),
    ).round(3).sort_values("taxa_alf")
    print(perfil.to_string())
    perfil.plot.bar(y="taxa_alf", legend=False, title="Taxa de alfabetização média por cluster")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "12_taxa_por_cluster.png"), dpi=110); plt.close()

    # 2) Ranking de risco (municípios com >=50 alunos, menor taxa)
    print("\n=== 2) RANKING DE RISCO (piores taxas, min 50 alunos) ===")
    risco = (muni[muni["n_alunos"] >= 50]
             .sort_values("taxa_alf")[["sigla_uf", "nome_regiao", "taxa_alf", "idhm", "ivs", "n_alunos", "cluster"]]
             .head(20).round(3))
    print(risco.to_string())

    # 3) Projeção de metas 2030 (Gold)
    print("\n=== 3) PROJEÇÃO DE METAS 2030 (Gold) ===")
    proj_info = {}
    try:
        gold = ler_parquet_s3(config.GOLD_INDICADOR_MUNICIPIO)
        gold["ano"] = gold["ano"].astype(int)
        piv = gold.pivot_table(index="id_municipio", columns="ano", values="taxa_alfabetizacao")
        meta = gold[gold["ano"] == 2024].set_index("id_municipio")["meta_alfabetizacao_2030"]
        vel = piv[2024] - piv[2023]
        proj_2030 = piv[2024] + vel * (2030 - 2024)
        gap = (proj_2030 - meta).dropna()
        n_risco_meta = int((gap < 0).sum())
        proj_info = {
            "municipios_avaliados": int(len(gap)),
            "municipios_nao_atingem_meta_2030": n_risco_meta,
            "pct_nao_atingem": round(100 * n_risco_meta / len(gap), 1),
            "gap_medio_pp": round(float(gap.mean()), 2),
        }
        print(proj_info)
        plt.figure(figsize=(7, 4))
        plt.hist(gap, bins=50, color="#4C72B0")
        plt.axvline(0, color="red", ls="--")
        plt.xlabel("gap projeção 2030 − meta 2030 (p.p.)"); plt.ylabel("nº municípios")
        plt.title("Projeção linear vs meta 2030")
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "13_projecao_metas.png"), dpi=110); plt.close()
    except Exception as e:
        print(f"[aviso] projeção de metas falhou ({type(e).__name__}: {e})")

    # Persistência
    muni.reset_index()[["id_municipio", "sigla_uf", "nome_regiao", "cluster", "taxa_alf", "n_alunos"]] \
        .to_csv(os.path.join(REPORTS, "clusters_municipios.csv"), index=False)
    perfil.to_json(os.path.join(REPORTS, "perfil_clusters.json"), indent=2, force_ascii=False)
    risco.to_csv(os.path.join(REPORTS, "risco_top20.csv"))
    if proj_info:
        with open(os.path.join(REPORTS, "projecao_metas.json"), "w") as f:
            json.dump(proj_info, f, indent=2, ensure_ascii=False)
    print("\nartefatos salvos em reports/.  === aplicação estratégica concluída ===")


if __name__ == "__main__":
    main()
