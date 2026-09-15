"""Gera o dashboard estático (index.html) do TC3 com Plotly.

Design: relatório editorial em tema escuro (Fraunces + Public Sans), seguindo o Web
Interface Guidelines (tabular-nums, text-wrap:balance, color-scheme, foco visível,
prefers-reduced-motion, HTML semântico) e o método de sistema de design coeso.
Reúne EDA + modelagem + estratégia a partir dos artefatos em `reports/` e da base local.
Rodar da raiz do repo:  python dashboard/gerar_dashboard.py
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

# Sistema de cores (tema escuro editorial)
INK = "#0E1116"
SURF = "#171C26"
TEXT = "#ECE6DA"
MUTED = "#9A9585"
AMBER = "#E8B04B"
TEAL = "#3DB9A0"
CORAL = "#E07A5F"
BLUE = "#6AA6E0"
PALETA = [AMBER, TEAL, CORAL, BLUE, "#B98BD9", "#C9C2B6"]
ESCALA = [[0.0, CORAL], [0.5, "#C9C2B6"], [1.0, TEAL]]  # baixo->alto
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


def carregar_json(nome):
    with open(os.path.join(REPORTS, nome), encoding="utf-8") as f:
        return json.load(f)


def limpar_nome(f):
    return (f.replace("cat__", "").replace("num__", "")
             .replace("sigla_uf_", "UF: ").replace("nome_regiao_", "Região: ")
             .replace("amazonia_legal_", "Amazônia Legal=").replace("_", " ").strip())


def estilizar(fig, altura=360):
    fig.update_layout(
        height=altura, colorway=PALETA, title=None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'Public Sans', system-ui, sans-serif", color=MUTED, size=13),
        margin=dict(l=64, r=24, t=14, b=44),
        hoverlabel=dict(bgcolor=SURF, font=dict(color=TEXT, family="'Public Sans', sans-serif")),
        legend=dict(font=dict(color=MUTED), bgcolor="rgba(0,0,0,0)"),
        coloraxis_showscale=False,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False, linecolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False, linecolor="rgba(255,255,255,0.12)")
    return fig


def div(fig, primeiro=False):
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if primeiro else False,
                       config=PLOT_CONFIG)


def main():
    base = pd.read_parquet(BASE, columns=["alfabetizado", "nome_regiao", "sigla_uf"])
    base["alfabetizado"] = base["alfabetizado"].astype(int)
    modelos = pd.DataFrame(carregar_json("metricas_modelos.json"))
    campeao = carregar_json("metricas_campeao.json")
    imp = pd.Series(carregar_json("importancias.json")).sort_values()
    perfil = pd.DataFrame(carregar_json("perfil_clusters.json"))
    proj = carregar_json("projecao_metas.json")
    taxa_nac = base["alfabetizado"].mean()

    kpis = [
        ("Alunos analisados", f"{len(base):,}".replace(",", "."), "2º ano do EF · 2023–2024"),
        ("Taxa de alfabetização", f"{taxa_nac:.0%}", "média nacional na base"),
        ("ROC-AUC do modelo", f"{campeao['roc_auc']:.2f}", "XGBoost · classe binária"),
        ("Municípios fora da meta 2030", f"{proj['pct_nao_atingem']:.0f}%", "por projeção de tendência"),
    ]

    # O problema — desigualdade territorial
    reg = base.groupby("nome_regiao")["alfabetizado"].mean().sort_values()
    f1 = px.bar(x=reg.values, y=reg.index, orientation="h", color=reg.values,
                color_continuous_scale=ESCALA)
    f1.update_layout(xaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None)
    f1.update_traces(hovertemplate="%{y}: %{x:.1%}<extra></extra>")

    uf = base.groupby("sigla_uf")["alfabetizado"].mean().sort_values()
    f2 = px.bar(x=uf.index, y=uf.values, color=uf.values, color_continuous_scale=ESCALA)
    f2.update_layout(yaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None,
                     xaxis={"categoryorder": "total ascending"})
    f2.update_traces(hovertemplate="%{x}: %{y:.1%}<extra></extra>")

    # O modelo
    m = modelos["roc_auc"].sort_values()
    f3 = px.bar(x=m.values, y=m.index, orientation="h")
    f3.update_traces(marker_color=AMBER, hovertemplate="%{y}: %{x:.3f}<extra></extra>")
    f3.update_layout(xaxis_title="ROC-AUC", yaxis_title=None)

    top = imp.tail(12)
    top.index = [limpar_nome(i) for i in top.index]
    f4 = px.bar(x=top.values, y=top.index, orientation="h")
    f4.update_traces(marker_color=TEAL, hovertemplate="%{y}: %{x:.3f}<extra></extra>")
    f4.update_layout(xaxis_title="importância", yaxis_title=None)

    # Inteligência aplicada
    f5 = go.Figure()
    rotulos = ["Cluster " + str(c) for c in perfil.index]
    f5.add_bar(name="Taxa alfabetização", x=rotulos, y=perfil["taxa_alf"], marker_color=AMBER)
    f5.add_bar(name="IDHM", x=rotulos, y=perfil["idhm"], marker_color=BLUE)
    f5.update_layout(barmode="group", yaxis_title=None)

    atingem = 100 - proj["pct_nao_atingem"]
    f6 = go.Figure(go.Pie(
        labels=["Não atingem", "Atingem"], values=[proj["pct_nao_atingem"], atingem],
        hole=0.62, marker_colors=[CORAL, TEAL], sort=False,
        textfont=dict(color=INK, family="'Public Sans', sans-serif")))
    f6.update_layout(showlegend=True)

    figs = [f1, f2, f3, f4, f5, f6]
    for f in figs:
        estilizar(f)
    # ajuste pós-estilo: legenda do gráfico de clusters (o estilizar() reseta o layout)
    f5.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0,
                                 font=dict(color=MUTED)), margin=dict(l=64, r=44, t=46, b=44))

    cards = [
        ("O problema", "Desigualdade regional", "Taxa de alfabetização por região",
         "O Norte fica ~13 pontos abaixo do Centro-Oeste — o eixo territorial é a primeira fratura.", f1),
        ("O problema", "Contraste entre estados", "Taxa de alfabetização por UF",
         "O Ceará (política do PAIC) descola de todos; Bahia, Rio Grande do Norte e Sergipe ficam na base.", f2),
        ("O modelo", "Comparação de algoritmos", "ROC-AUC no conjunto de teste",
         "Boosting e floresta lideram; a diferença modesta reflete o teto da predição individual.", f3),
        ("O modelo", "O que pesa na predição", "Variáveis mais influentes (campeão)",
         "Geografia domina — UF e região acima de qualquer variável socioeconômica contínua.", f4),
        ("Inteligência aplicada", "Dois Brasis", "Perfil dos clusters de municípios",
         "O agrupamento socioeconômico separa nitidamente municípios vulneráveis dos desenvolvidos.", f5),
        ("Inteligência aplicada", "Rota até 2030", "Projeção de tendência vs. meta",
         "Cerca de metade dos municípios não atinge a meta mantendo o ritmo atual — alerta de política pública.", f6),
    ]

    partes = [_HEAD, _hero(kpis)]
    secoes = ["O problema", "O modelo", "Inteligência aplicada"]
    for i, sec in enumerate(secoes):
        partes.append(f"<section class='bloco' aria-labelledby='s{i}'>"
                      f"<h2 id='s{i}' class='secao'>{sec}</h2><div class='grade'>")
        primeiro_global = (i == 0)
        for j, (s, _tag, titulo, dek, fig) in enumerate(cards):
            if s != sec:
                continue
            chart = div(fig, primeiro=(primeiro_global and j == 0))
            partes.append(
                f"<article class='card reveal' style='--d:{j % 2 * 80}ms'>"
                f"<h3>{titulo}</h3><p class='dek'>{dek}</p>{chart}</article>")
        partes.append("</div></section>")
    partes.append(_FOOT)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(partes))
    print(f"dashboard salvo -> {os.path.relpath(OUTPUT)}  ({os.path.getsize(OUTPUT)//1024} KB)")


def _hero(kpis):
    tiles = "".join(
        f"<div class='kpi reveal' style='--d:{i*70}ms'><span class='v'>{v}</span>"
        f"<span class='l'>{l}</span><span class='h'>{h}</span></div>"
        for i, (l, v, h) in enumerate(kpis))
    return f"""
<a href="#main" class="skip">Pular para o conteúdo</a>
<header class="hero">
  <p class="eyebrow">Tech Challenge · Fase 3 · Pós-graduação AI Scientist — FIAP</p>
  <h1>Quem o Brasil está alfabetizando — e quem fica&nbsp;para&nbsp;trás</h1>
  <p class="dek grande">Um modelo supervisionado no grão do aluno e uma leitura estratégica
  por município, sobre a base pública do 2º ano do Ensino Fundamental enriquecida com IBGE,
  Atlas do Desenvolvimento Humano e Censo Escolar.</p>
  <p class="assinatura">Theo Coleone de Camargo · dados 2023–2024</p>
</header>
<main id="main">
<section class="kpis" aria-label="Indicadores-chave">{tiles}</section>"""


_HEAD = """<!doctype html>
<html lang="pt-br" style="color-scheme: dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#0E1116">
<title>Alfabetização no Brasil — Tech Challenge Fase 3</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,900&family=Public+Sans:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root{
  --ink:#0E1116; --surf:#171C26; --surf2:#1E2430; --line:rgba(255,255,255,.09);
  --text:#ECE6DA; --muted:#9A9585; --amber:#E8B04B; --teal:#3DB9A0; --coral:#E07A5F;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0; background:var(--ink); color:var(--text);
  font-family:'Public Sans',system-ui,-apple-system,sans-serif; line-height:1.55;
  background-image:
    radial-gradient(60rem 40rem at 85% -10%, rgba(232,176,75,.10), transparent 60%),
    radial-gradient(50rem 40rem at -10% 20%, rgba(61,185,160,.08), transparent 55%);
  background-attachment:fixed;
}
body::after{ /* grão */
  content:""; position:fixed; inset:0; pointer-events:none; opacity:.035; z-index:1;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}
.skip{position:absolute;left:-999px;top:0;background:var(--amber);color:var(--ink);padding:10px 16px;border-radius:0 0 8px 0;z-index:10;font-weight:700}
.skip:focus{left:0}
.hero{max-width:1100px;margin:0 auto;padding:72px 6vw 24px;position:relative;z-index:2}
.eyebrow{font-size:.82rem;letter-spacing:.14em;text-transform:uppercase;color:var(--amber);margin:0 0 18px;font-weight:700}
h1{font-family:'Fraunces',serif;font-weight:900;font-size:clamp(2.4rem,6vw,4.6rem);line-height:1.02;
   margin:0;letter-spacing:-.01em;text-wrap:balance}
.dek{color:var(--muted);text-wrap:pretty;max-width:60ch}
.dek.grande{font-size:1.18rem;margin:22px 0 0}
.assinatura{color:var(--muted);font-size:.9rem;margin-top:16px;font-variant-numeric:tabular-nums}
.kpis{max-width:1100px;margin:40px auto 8px;padding:0 6vw;display:grid;
      grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:1px;background:var(--line);
      border:1px solid var(--line);border-radius:16px;overflow:hidden}
.kpi{background:var(--surf);padding:26px 24px;display:flex;flex-direction:column;gap:4px}
.kpi .v{font-family:'Fraunces',serif;font-weight:600;font-size:2.6rem;color:var(--amber);
        font-variant-numeric:tabular-nums;line-height:1}
.kpi .l{font-weight:700;font-size:.98rem;margin-top:6px}
.kpi .h{color:var(--muted);font-size:.82rem}
.bloco{max-width:1100px;margin:0 auto;padding:44px 6vw}
.secao{font-family:'Fraunces',serif;font-weight:600;font-size:1.9rem;margin:0 0 22px;
       padding-bottom:12px;border-bottom:1px solid var(--line);text-wrap:balance}
.grade{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:20px}
.card{background:var(--surf);border:1px solid var(--line);border-radius:16px;padding:22px 22px 8px;
      transition:transform .3s ease,border-color .3s ease}
.card:hover{transform:translateY(-3px);border-color:rgba(232,176,75,.4)}
.card h3{font-family:'Fraunces',serif;font-weight:600;font-size:1.25rem;margin:0 0 4px;text-wrap:balance}
.card .dek{font-size:.92rem;margin:0 0 6px}
footer{max-width:1100px;margin:0 auto;padding:36px 6vw 72px;color:var(--muted);font-size:.86rem;
       border-top:1px solid var(--line);text-wrap:pretty}
footer b{color:var(--text)}
.reveal{opacity:0;transform:translateY(14px);animation:rise .7s cubic-bezier(.2,.7,.2,1) forwards;animation-delay:var(--d,0ms)}
@keyframes rise{to{opacity:1;transform:none}}
:focus-visible{outline:2px solid var(--amber);outline-offset:3px;border-radius:6px}
@media (prefers-reduced-motion:reduce){
  html{scroll-behavior:auto}
  .reveal{animation:none;opacity:1;transform:none}
  .card{transition:none}
}
</style>
</head>
<body>"""

_FOOT = """</main>
<footer>
<b>Metodologia.</b> Modelo campeão: XGBoost (pipeline Scikit-learn com imputação, codificação e
padronização integradas; validação estratificada; tuning por RandomizedSearchCV; interpretação por SHAP).
As métricas no grão individual são modestas por design — prever um aluno a partir de contexto agregado tem
teto baixo; o valor está na interpretação e na leitura municipal. Alvo derivado do rótulo da fonte: alunos
ausentes na avaliação contam como não alfabetizados (limitação documentada).<br><br>
<b>Fontes.</b> Base dos Dados · INEP (SAEB Alfabetização, Censo Escolar) · IBGE · Atlas do Desenvolvimento
Humano (PNUD). Autor: Theo Coleone de Camargo — Tech Challenge Fase 3, FIAP.
</footer>
</body></html>"""


if __name__ == "__main__":
    main()
