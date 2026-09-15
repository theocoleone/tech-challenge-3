"""Gera o dashboard estático (index.html) do TC3 com Plotly.

Reúne os destaques de EDA, modelagem e aplicação estratégica a partir dos artefatos em
`reports/` e da base local. Segue o padrão do TC2: HTML único, Plotly via CDN, publicável
em S3 + CloudFront. Rodar da raiz do repo:  python dashboard/gerar_dashboard.py
"""
import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(AQUI, "..", "data", "base_analitica.parquet")
REPORTS = os.path.join(AQUI, "..", "reports")
OUTPUT = os.path.join(AQUI, "index.html")

PALETA = ["#2E5EAA", "#E07A5F", "#3D9970", "#B5179E", "#F2A900", "#5C6B73"]


def carregar_json(nome):
    with open(os.path.join(REPORTS, nome), encoding="utf-8") as f:
        return json.load(f)


def limpar_nome(f):
    return (f.replace("cat__", "").replace("num__", "")
             .replace("sigla_uf_", "UF: ").replace("nome_regiao_", "Região: ")
             .replace("amazonia_legal_", "Amazônia Legal=").replace("_", " "))


def estilizar(fig, altura=380, titulo=""):
    fig.update_layout(
        template="plotly_white", height=altura, title=titulo,
        margin=dict(l=60, r=30, t=60, b=50), colorway=PALETA,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=13),
    )
    return fig


def main():
    base = pd.read_parquet(BASE, columns=["alfabetizado", "nome_regiao", "sigla_uf"])
    base["alfabetizado"] = base["alfabetizado"].astype(int)
    modelos = pd.DataFrame(carregar_json("metricas_modelos.json"))
    campeao = carregar_json("metricas_campeao.json")
    imp = pd.Series(carregar_json("importancias.json")).sort_values(ascending=True)
    perfil = pd.DataFrame(carregar_json("perfil_clusters.json"))
    proj = carregar_json("projecao_metas.json")

    taxa_nac = base["alfabetizado"].mean()

    # KPIs
    kpis = [
        ("Alunos analisados", f"{len(base):,}".replace(",", ".")),
        ("Taxa de alfabetização (nacional)", f"{taxa_nac:.1%}"),
        ("ROC-AUC do modelo campeão", f"{campeao['roc_auc']:.2f}"),
        ("Municípios fora da meta 2030", f"{proj['pct_nao_atingem']:.0f}%"),
    ]

    # fig 1 — taxa por região
    reg = base.groupby("nome_regiao")["alfabetizado"].mean().sort_values()
    fig1 = estilizar(px.bar(reg, orientation="h", labels={"value": "taxa", "nome_regiao": ""}),
                     titulo="Taxa de alfabetização por região")
    fig1.update_layout(showlegend=False, xaxis_tickformat=".0%")

    # fig 2 — taxa por UF
    uf = base.groupby("sigla_uf")["alfabetizado"].mean().sort_values()
    fig2 = estilizar(px.bar(uf, labels={"value": "taxa", "sigla_uf": ""}), altura=420,
                     titulo="Taxa de alfabetização por UF (Ceará e Bahia como extremos)")
    fig2.update_layout(showlegend=False, yaxis_tickformat=".0%")

    # fig 3 — comparação de modelos (ROC-AUC)
    m = modelos["roc_auc"].sort_values()
    fig3 = estilizar(px.bar(m, orientation="h", labels={"value": "ROC-AUC", "index": ""}),
                     titulo="Comparação de modelos (ROC-AUC)")
    fig3.update_layout(showlegend=False)
    fig3.update_traces(marker_color="#2E5EAA")

    # fig 4 — top-12 variáveis mais influentes
    top = imp.tail(12)
    top.index = [limpar_nome(i) for i in top.index]
    fig4 = estilizar(px.bar(top, orientation="h", labels={"value": "importância", "index": ""}),
                     titulo="Variáveis mais influentes (modelo campeão)")
    fig4.update_layout(showlegend=False)
    fig4.update_traces(marker_color="#3D9970")

    # fig 5 — perfil dos clusters
    perfil_t = perfil.T if "taxa_alf" not in perfil.columns else perfil
    fig5 = go.Figure()
    clusters = list(perfil.index)
    fig5.add_bar(name="Taxa alfabetização", x=clusters, y=perfil["taxa_alf"], marker_color="#2E5EAA")
    fig5.add_bar(name="IDHM", x=clusters, y=perfil["idhm"], marker_color="#F2A900")
    fig5 = estilizar(fig5, titulo="Perfil dos clusters de municípios (0=vulnerável, 1=desenvolvido)")
    fig5.update_layout(barmode="group", xaxis_title="cluster")

    # fig 6 — projeção de metas 2030
    atingem = 100 - proj["pct_nao_atingem"]
    fig6 = estilizar(go.Figure(go.Pie(
        labels=["Não atingem a meta 2030", "Atingem a meta 2030"],
        values=[proj["pct_nao_atingem"], atingem], hole=0.55,
        marker_colors=["#E07A5F", "#3D9970"])),
        titulo="Projeção linear vs meta 2030 (municípios)")

    figs = [fig1, fig2, fig3, fig4, fig5, fig6]

    # ---- montagem do HTML ----
    css = """
    :root{--tinta:#1b2a3a;--papel:#f7f5ef;--card:#ffffff;}
    *{box-sizing:border-box} body{margin:0;background:var(--papel);color:var(--tinta);
      font-family:Inter,system-ui,-apple-system,sans-serif;line-height:1.5}
    header{padding:40px 6vw 20px} h1{margin:0;font-size:1.9rem}
    .sub{color:#5c6b73;margin-top:6px;max-width:760px}
    .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px;padding:20px 6vw}
    .kpi{background:var(--card);border-radius:14px;padding:20px;box-shadow:0 1px 4px rgba(0,0,0,.06)}
    .kpi .v{font-size:1.8rem;font-weight:700;color:#2E5EAA} .kpi .l{color:#5c6b73;font-size:.9rem}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:18px;padding:10px 6vw 50px}
    .card{background:var(--card);border-radius:14px;padding:10px 16px;box-shadow:0 1px 4px rgba(0,0,0,.06)}
    footer{padding:20px 6vw 50px;color:#5c6b73;font-size:.85rem}
    """
    partes = [f"<!doctype html><html lang='pt-br'><head><meta charset='utf-8'>",
              "<meta name='viewport' content='width=device-width,initial-scale=1'>",
              "<title>TC3 — Alfabetização no Brasil</title>",
              "<link rel='preconnect' href='https://fonts.googleapis.com'>",
              "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap' rel='stylesheet'>",
              f"<style>{css}</style></head><body>",
              "<header><h1>Predição e Inteligência Analítica para Alfabetização no Brasil</h1>",
              "<div class='sub'>Tech Challenge Fase 3 — FIAP. Modelo supervisionado (grão aluno) + "
              "análise estratégica municipal sobre a camada Gold do TC2, enriquecida com IBGE, "
              "Atlas do Desenvolvimento Humano e Censo Escolar.</div></header>",
              "<section class='kpis'>"]
    for label, val in kpis:
        partes.append(f"<div class='kpi'><div class='v'>{val}</div><div class='l'>{label}</div></div>")
    partes.append("</section><section class='grid'>")
    for i, fig in enumerate(figs):
        html = fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
        partes.append(f"<div class='card'>{html}</div>")
    partes.append("</section>")
    partes.append("<footer>Autor: Theo Coleone de Camargo · Dados: Base dos Dados / INEP / IBGE / "
                  "Atlas PNUD · Modelo campeão: XGBoost · Métricas no grão individual são modestas "
                  "por design (contexto agregado); o valor está na interpretação e na análise municipal.</footer>")
    partes.append("</body></html>")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(partes))
    print(f"dashboard salvo -> {os.path.relpath(OUTPUT)}  ({os.path.getsize(OUTPUT)//1024} KB)")


if __name__ == "__main__":
    main()
