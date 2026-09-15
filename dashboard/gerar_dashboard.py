"""Gera o dashboard estático (index.html) do TC3 — estética de relatório estatístico oficial.

Fundo branco, quase monocromático, tipografia IBM Plex (Serif nos títulos, Mono nos
rótulos/números), seções numeradas, tabelas com fios e muito respiro. Segue o Web Interface
Guidelines (tabular-nums, foco visível, HTML semântico, prefers-reduced-motion). Narrativa em
duas abas (Técnica e Business), pensada para apresentação com gravação de tela.
Rodar da raiz do repo:  python dashboard/gerar_dashboard.py
"""
import base64
import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

AQUI = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(AQUI, "..", "reports")
FLOW_PNG = os.path.join(AQUI, "..", "images", "arquitetura_tc3.png")
OUTPUT = os.path.join(AQUI, "index.html")

INK, MUTED, PAPER = "#16191d", "#5b6169", "#ffffff"
ACCENT, NEG, MID = "#1f5c8c", "#a6462e", "#9db9d0"
ESCALA = [[0.0, "#dce6ee"], [1.0, ACCENT]]
CFG = {"displayModeBar": False, "responsive": True}
_primeiro = [True]


def cj(nome):
    with open(os.path.join(REPORTS, nome), encoding="utf-8") as f:
        return json.load(f)


def limpar(f):
    return (f.replace("cat__", "").replace("num__", "").replace("sigla_uf_", "UF: ")
             .replace("nome_regiao_", "Região: ").replace("amazonia_legal_", "Amazônia Legal=")
             .replace("esc_prop_", "escola: ").replace("esc_media_", "escola méd.: ")
             .replace("_", " ").strip())


def estilizar(fig, altura=340):
    fig.update_layout(
        height=altura, colorway=[ACCENT, MID, NEG, "#7a8189"], title=None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'IBM Plex Sans',system-ui,sans-serif", color=MUTED, size=13),
        margin=dict(l=64, r=24, t=16, b=44), coloraxis_showscale=False,
        hoverlabel=dict(bgcolor=PAPER, font=dict(color=INK)), legend=dict(font=dict(color=MUTED), bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="#eef0f2", zeroline=False, linecolor="#c9ccd1", tickcolor="#c9ccd1")
    fig.update_yaxes(gridcolor="#eef0f2", zeroline=False, linecolor="#c9ccd1", tickcolor="#c9ccd1")
    return fig


def div(fig):
    html = fig.to_html(full_html=False, include_plotlyjs="cdn" if _primeiro[0] else False, config=CFG)
    _primeiro[0] = False
    return html


def card(titulo, dek, inner, wide=False):
    w = " wide" if wide else ""
    d = f"<p class='dek'>{dek}</p>" if dek else ""
    return f"<article class='card{w}'><h3>{titulo}</h3>{d}{inner}</article>"


def bloco(num, titulo):
    return f"<div class='secao'><span class='num'>{num}</span><h2>{titulo}</h2></div>"


def item(pergunta, resposta, chart):
    return (f"<details class='qa-item'><summary>{pergunta}</summary>"
            f"<div class='qa-body'><p class='a'>{resposta}</p>{chart}</div></details>")


def main():
    agg = cj("agregados_base.json")
    modelos = pd.DataFrame(cj("metricas_modelos.json"))
    campeao = cj("metricas_campeao.json")
    nome_campeao = campeao["modelo"].replace(" (tunado)", "")
    perfil = pd.DataFrame(cj("perfil_clusters.json"))
    proj = cj("projecao_metas.json")
    conf = cj("confusao.json")
    roc = cj("roc.json")
    shap_imp = pd.Series(cj("shap_importancia.json")).sort_values()

    kpis = [
        ("Alunos analisados", f"{agg['n_alunos']:,}".replace(",", "."), "2º ano do EF · 2023–2024"),
        ("Taxa de alfabetização", f"{agg['taxa_nacional']:.0%}", "média nacional na base"),
        ("ROC-AUC do modelo", f"{campeao['roc_auc']:.2f}", f"{nome_campeao} · classe binária"),
        ("Municípios fora da meta 2030", f"{proj['pct_nao_atingem']:.0f}%", "por projeção de tendência"),
    ]

    reg = pd.Series(agg["taxa_por_regiao"]).sort_values()
    f_reg = px.bar(x=reg.values, y=reg.index, orientation="h", color=reg.values, color_continuous_scale=ESCALA)
    f_reg.update_layout(xaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None)
    f_reg.update_traces(hovertemplate="%{y}: %{x:.1%}<extra></extra>")
    uf = pd.Series(agg["taxa_por_uf"]).sort_values()
    f_uf = px.bar(x=uf.index, y=uf.values, color=uf.values, color_continuous_scale=ESCALA)
    f_uf.update_layout(yaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None, xaxis={"categoryorder": "total ascending"})
    f_uf.update_traces(hovertemplate="%{x}: %{y:.1%}<extra></extra>")

    m = modelos["roc_auc"].sort_values()
    f_roc_auc = px.bar(x=m.values, y=m.index, orientation="h")
    f_roc_auc.update_traces(marker_color=ACCENT, hovertemplate="%{y}: %{x:.3f}<extra></extra>")
    f_roc_auc.update_layout(xaxis_title="ROC-AUC", yaxis_title=None)

    z = [[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]]
    fcm = go.Figure(go.Heatmap(
        z=z, x=["previsto: não alf.", "previsto: alf."], y=["real: não alf.", "real: alf."],
        colorscale=[[0, "#eef3f7"], [1, "#8fb0cb"]], showscale=False,
        text=[[f"{v:,}".replace(",", ".") for v in r] for r in z], texttemplate="%{text}",
        textfont=dict(color=INK, size=16)))
    fcm.update_layout(yaxis_autorange="reversed")
    froc = go.Figure()
    froc.add_scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=f"AUC {roc['auc']:.2f}",
                     line=dict(color=ACCENT, width=2.5), fill="tozeroy", fillcolor="rgba(31,92,140,.10)")
    froc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#c9ccd1", dash="dash", width=1), showlegend=False)
    froc.update_layout(xaxis_title="taxa de falso positivo", yaxis_title="taxa de verdadeiro positivo", legend=dict(x=.55, y=.1))

    shap_imp.index = [limpar(i) for i in shap_imp.index]
    fshap = px.bar(x=shap_imp.values, y=shap_imp.index, orientation="h")
    fshap.update_traces(marker_color=ACCENT, hovertemplate="%{y}: %{x:.3f}<extra></extra>")
    fshap.update_layout(xaxis_title="importância média |SHAP|", yaxis_title=None)

    imp = pd.Series(cj("importancias.json")).sort_values().tail(12)
    imp.index = [limpar(i) for i in imp.index]
    f_gain = px.bar(x=imp.values, y=imp.index, orientation="h")
    f_gain.update_traces(marker_color=ACCENT, hovertemplate="%{y}: %{x:.3f}<extra></extra>")
    f_gain.update_layout(xaxis_title="importância (ganho do modelo)", yaxis_title=None)

    f_clu = go.Figure()
    rot = ["Cluster " + str(c) for c in perfil.index]
    f_clu.add_bar(name="Taxa alfabetização", x=rot, y=perfil["taxa_alf"], marker_color=ACCENT)
    f_clu.add_bar(name="IDHM", x=rot, y=perfil["idhm"], marker_color=MID)
    f_clu.update_layout(barmode="group", yaxis_title=None)
    atingem = 100 - proj["pct_nao_atingem"]
    f_meta = go.Figure(go.Pie(labels=["Não atingem", "Atingem"], values=[proj["pct_nao_atingem"], atingem],
                              hole=0.62, marker_colors=[NEG, ACCENT], sort=False, textfont=dict(color="#ffffff")))

    for f in [f_reg, f_uf, f_roc_auc, fcm, froc, fshap, f_gain, f_clu, f_meta]:
        estilizar(f)
    f_clu.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0, font=dict(color=MUTED)),
                        margin=dict(l=64, r=44, t=46, b=44))
    fshap.update_layout(margin=dict(l=185, r=24, t=16, b=44))
    f_gain.update_layout(margin=dict(l=185, r=24, t=16, b=44))

    cols = [("acuracia", "Acurácia"), ("precisao_nao_alf", "Precisão (não alf.)"),
            ("recall_nao_alf", "Recall (não alf.)"), ("f1_macro", "F1-macro"),
            ("roc_auc", "ROC-AUC"), ("auc_pr", "AUC-PR")]
    linhas = ""
    for nome, row in modelos.sort_values("roc_auc", ascending=False).iterrows():
        cls = "campeao" if nome == nome_campeao else ""
        tds = "".join(f"<td>{row[c]:.3f}</td>" for c, _ in cols)
        linhas += f"<tr class='{cls}'><td>{nome}</td>{tds}</tr>"
    th = "".join(f"<th>{lab}</th>" for _, lab in cols)
    tabela = f"<table class='modelos'><thead><tr><th>Modelo</th>{th}</tr></thead><tbody>{linhas}</tbody></table>"

    contexto = ("<p class='texto'>Construí um <b>modelo supervisionado</b> que prevê se um aluno do 2º ano do "
                "Ensino Fundamental será <b>alfabetizado</b> ou <b>não alfabetizado</b>, a partir de variáveis "
                "<b>educacionais, territoriais e socioeconômicas</b>. Sobre o resultado, uma camada de análise "
                "traduz o modelo em <b>inteligência para a decisão pública</b>.</p>"
                "<p class='link-ext'><a href='https://github.com/theocoleone/tech-challenge-3' "
                "target='_blank' rel='noopener'>Código no GitHub <span aria-hidden='true'>↗</span></a></p>")
    tc2 = ("<p class='texto'>Este projeto parte da camada <b>Gold</b> do <b>Tech Challenge 2</b>, onde construí "
           "uma pipeline de dados em nuvem seguindo a <b>Arquitetura Medalhão</b> no S3: ingestão do BigQuery, "
           "camadas Bronze → Silver → Gold com <b>AWS Glue</b>, um caminho de <b>streaming</b> (SQS + Lambda) e "
           "um dashboard em CloudFront. O TC3 consome esses dados já tratados para a etapa de Machine Learning.</p>"
           "<p class='link-ext'><a href='https://d131u8q9uidloe.cloudfront.net' target='_blank' rel='noopener'>"
           "Abrir o dashboard do TC2 <span aria-hidden='true'>↗</span></a></p>")

    etapas = [("Base integrada", "S3 Gold do TC2 + Censo Escolar + Atlas ADH + IBGE"),
              ("EDA", "distribuições, correlações, desigualdade e hipóteses"),
              ("Pré-processamento", "imputação, encoding e scaling no ColumnTransformer"),
              ("Treino + validação", "4 modelos, Stratified K-Fold e tuning"),
              ("Avaliação", "métricas de teste + SHAP"),
              ("Aplicação", "as perguntas de negócio")]
    passos = "".join(
        f"<div class='step'><span class='n'>{i+1:02d}</span>"
        f"<div class='txt'><b>{t}</b><span class='d'>{d}</span></div></div>"
        for i, (t, d) in enumerate(etapas))
    flow = f"<div class='flow'>{passos}</div>"

    fontes = ("<ul class='fontes'>"
              "<li><b>silver/alunos (SAEB)</b> — 3,9 M de registros; grão do aluno e alvo.</li>"
              "<li><b>Censo Escolar (INEP)</b> — infraestrutura e docentes, agregados por município.</li>"
              "<li><b>Atlas do Desenvolvimento Humano (PNUD)</b> — IDHM, renda, Gini, pobreza, saneamento.</li>"
              "<li><b>IBGE</b> — população e PIB per capita.</li>"
              "<li><b>Diretório IBGE</b> — região, mesorregião, Amazônia Legal.</li>"
              "</ul><p class='texto'>Tudo integrado por <code>id_municipio</code> (e ano, no Censo), lido do "
              "S3 e do BigQuery.</p>")

    metodo = "<ol class='metodo'>" + "".join(f"<li>{s}</li>" for s in [
        "<b>Imputação de faltantes</b> — mediana nas variáveis numéricas (SimpleImputer).",
        "<b>Transformação de variáveis</b> — One-Hot nas categóricas e padronização nas numéricas.",
        "<b>Tratamento de data leakage</b> — removidas proficiência, presença e agregados do exame.",
        "<b>Pré-processamento integrado ao modelo</b> — tudo dentro de um Pipeline/ColumnTransformer, ajustado só no treino.",
        "<b>Treinamento e validação</b> — split estratificado + Stratified K-Fold; tuning por RandomizedSearchCV.",
        "<b>Replicabilidade e generalização</b> — semente fixa; teste intocado avaliado uma única vez.",
    ]) + "</ol>"

    with open(FLOW_PNG, "rb") as fp:
        flow_b64 = base64.b64encode(fp.read()).decode()
    figura = (f"<div class='figura'><img alt='Fluxograma da arquitetura AWS: BigQuery para S3, SageMaker, "
              f"CloudFront' src='data:image/png;base64,{flow_b64}'></div>")

    vuln = perfil.sort_values("taxa_alf").iloc[0]
    dev = perfil.sort_values("taxa_alf").iloc[-1]
    piores_uf = ", ".join(uf.head(5).index.tolist())
    n_vuln = f"{int(vuln['n_municipios']):,}".replace(",", ".")
    pct_fora = f"{proj['pct_nao_atingem']:.0f}"

    # ================= montagem =================
    p = [_HEAD, _hero(kpis), _tabbar(), "<main>"]

    # ---------- ABA TÉCNICA ----------
    p += ["<section id='tab-tecnica' role='tabpanel' aria-label='Visão técnica'>"]
    p += [bloco("1", "O que foi construído"),
          "<div class='grade'>",
          card("Objetivo do projeto", "Da predição do aluno à inteligência para a gestão.", contexto, wide=True),
          card("Herança do Tech Challenge 2", "", tc2, wide=True),
          card("Etapas do projeto", "", flow, wide=True),
          card("Como a base foi integrada", "", fontes, wide=True),
          card("Como foi feito", "", metodo, wide=True),
          "</div>"]
    p += [bloco("2", "Modelo e avaliação"),
          "<div class='grade'>",
          card("Comparação de modelos", f"ROC-AUC no teste — {nome_campeao} campeão.", tabela, wide=True),
          card("ROC-AUC por modelo", "", div(f_roc_auc)),
          card("Matriz de confusão", "", div(fcm)),
          card("Curva ROC", f"Área sob a curva = {roc['auc']:.2f}.", div(froc)),
          "</div>"]
    p += [bloco("3", "Arquitetura em nuvem"),
          "<div class='grade'>",
          card("Serviços e fluxo de dados", "Do BigQuery ao CloudFront — via S3 e SageMaker, com bucket privado (OAC).", figura, wide=True),
          "</div>"]
    p += ["</section>"]

    # ---------- ABA BUSINESS ----------
    p += ["<section id='tab-business' role='tabpanel' aria-label='Visão de negócio' hidden>"]
    p += [bloco("1", "Inteligência aplicada"),
          "<div class='perguntas'>",
          item("1 · Quais fatores mais impactam a alfabetização?",
               "A geografia (UF e região) e o porte/infraestrutura da escola lideram — acima de qualquer "
               "indicador socioeconômico isolado, segundo os valores SHAP.", div(fshap)),
          item("2 · Quais municípios apresentam maior risco educacional?",
               f"O risco se concentra no Norte e Nordeste; as UFs de menor taxa são {piores_uf}.", div(f_uf)),
          item("3 · Quais regiões possuem padrões semelhantes?",
               f"O agrupamento revela “dois Brasis”: vulnerável (taxa {vuln['taxa_alf']:.0%}, IDHM "
               f"{vuln['idhm']:.2f}, pobreza {vuln['prop_pobreza']:.0f}%) e desenvolvido (taxa "
               f"{dev['taxa_alf']:.0%}, IDHM {dev['idhm']:.2f}).", div(f_clu)),
          item("4 · Como prever municípios que podem não atingir metas futuras?",
               f"Extrapolando a tendência de 2023 para 2024, {pct_fora}% dos municípios não alcançam a meta "
               "de 2030 — uma lista objetiva de prioridades.", div(f_meta)),
          item("5 · Quais variáveis possuem maior influência nos modelos?",
               "No ganho do modelo, sigla_uf (Bahia e Ceará), nome_regiao e Amazônia Legal lideram.", div(f_gain)),
          "</div>"]
    p += [bloco("2", "Onde agir primeiro"),
          "<div class='perguntas'>",
          item("Priorizar recursos nos territórios mais vulneráveis",
               f"Concentrar esforço nos {n_vuln} municípios do cluster vulnerável e nas regiões de menor "
               "taxa, em vez de distribuir de forma uniforme.", div(f_reg)),
          item("Replicar o que funciona — a política do Ceará (PAIC)",
               "O Ceará descola de todos os estados: evidência de que ação pública consistente supera o "
               "contexto socioeconômico.", div(f_uf)),
          item("Monitorar as metas e agir de forma preventiva",
               f"Acompanhar os ~{pct_fora}% de municípios fora da rota de 2030 e agir antes da próxima "
               "avaliação, não depois.", div(f_meta)),
          item("Fechar déficits de infraestrutura escolar",
               "Internet, biblioteca e docentes aparecem entre os fatores relevantes (SHAP) — priorizar "
               "onde o Censo Escolar aponta carência.", div(fshap)),
          "</div>"]
    p += ["</section>"]

    p += ["</main>", _SCRIPT, _foot(nome_campeao)]

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(p))
    print(f"dashboard salvo -> {os.path.relpath(OUTPUT)}  ({os.path.getsize(OUTPUT)//1024} KB)")


def _hero(kpis):
    tiles = "".join(f"<div class='kpi'><span class='v'>{v}</span><span class='l'>{l}</span>"
                    f"<span class='h'>{h}</span></div>" for l, v, h in kpis)
    return f"""<a href="#main" class="skip">Pular para o conteúdo</a>
<header class="hero">
  <p class="eyebrow">Tech Challenge · Fase 3 · Pós-graduação AI Scientist — FIAP</p>
  <h1>Quem o Brasil está alfabetizando?</h1>
  <p class="dek">Um modelo supervisionado no grão do aluno e uma leitura estratégica por município,
  sobre a base pública do 2º ano do Ensino Fundamental enriquecida com IBGE, Atlas do Desenvolvimento
  Humano e Censo Escolar.</p>
  <p class="assinatura">Theo Coleone de Camargo · dados 2023–2024</p>
</header>
<section class="kpis" aria-label="Indicadores-chave">{tiles}</section>"""


def _tabbar():
    return ("<div class='tabs' role='tablist' aria-label='Visões do projeto'>"
            "<button role='tab' data-tab='tecnica' aria-selected='true'>Visão técnica</button>"
            "<button role='tab' data-tab='business' aria-selected='false'>Visão de negócio</button>"
            "</div>")


_SCRIPT = """<script>
(function(){
  var tabs=[].slice.call(document.querySelectorAll('[role=tab]'));
  var panels={tecnica:document.getElementById('tab-tecnica'),business:document.getElementById('tab-business')};
  function activate(name){
    tabs.forEach(function(t){var on=t.dataset.tab===name;t.setAttribute('aria-selected',on);t.classList.toggle('on',on);});
    for(var k in panels){panels[k].hidden=(k!==name);}
    window.dispatchEvent(new Event('resize'));
  }
  tabs.forEach(function(t,i){
    t.addEventListener('click',function(){activate(t.dataset.tab);});
    t.addEventListener('keydown',function(e){
      if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();
        var j=(i+(e.key==='ArrowRight'?1:tabs.length-1))%tabs.length;activate(tabs[j].dataset.tab);tabs[j].focus();}
    });
  });
  document.querySelectorAll('details.qa-item').forEach(function(d){
    d.addEventListener('toggle',function(){ if(d.open){window.dispatchEvent(new Event('resize'));} });
  });
  activate('tecnica');
})();
</script>"""


_HEAD = """<!doctype html>
<html lang="pt-br" style="color-scheme: light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#ffffff">
<title>Alfabetização no Brasil — Tech Challenge Fase 3</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{--paper:#ffffff;--ink:#16191d;--muted:#5b6169;--line:#e4e6e9;--line2:#c9ccd1;--accent:#1f5c8c;--neg:#a6462e;--surf:#f5f6f7}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--ink);font-family:'IBM Plex Sans',system-ui,-apple-system,sans-serif;
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
.skip{position:absolute;left:-999px;top:0;background:var(--accent);color:#fff;padding:10px 16px;z-index:20;font-weight:600}
.skip:focus{left:0}
.hero{max-width:960px;margin:0 auto;padding:64px 6vw 26px;text-align:center}
.eyebrow{font-family:'IBM Plex Mono',monospace;font-size:.75rem;letter-spacing:.18em;text-transform:uppercase;color:var(--accent);margin:0 0 20px}
h1{font-family:'IBM Plex Serif',serif;font-weight:600;font-size:clamp(2.1rem,5vw,3.7rem);line-height:1.08;margin:0;letter-spacing:-.01em;text-wrap:balance}
.dek{color:var(--muted);text-wrap:pretty;max-width:62ch;margin:20px auto 0;font-size:1.1rem}
.assinatura{font-family:'IBM Plex Mono',monospace;color:var(--muted);font-size:.82rem;margin-top:18px}
.kpis{max-width:1080px;margin:34px auto 0;padding:22px 6vw;display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
  border-top:2px solid var(--ink);border-bottom:1px solid var(--line2)}
.kpi{padding:4px 22px;border-left:1px solid var(--line)}
.kpi:first-child{border-left:0;padding-left:0}
.kpi .v{font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:2rem;color:var(--ink);line-height:1.15;font-variant-numeric:tabular-nums}
.kpi .l{display:block;font-weight:600;font-size:.9rem;margin-top:6px}
.kpi .h{display:block;color:var(--muted);font-size:.78rem;margin-top:2px}
.tabs{position:sticky;top:0;z-index:9;display:flex;gap:30px;justify-content:center;padding:0 6vw;margin-top:30px;
  background:rgba(255,255,255,.95);backdrop-filter:blur(8px);border-bottom:1px solid var(--line2)}
.tabs button{font-family:'IBM Plex Mono',monospace;font-size:.84rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
  background:none;border:0;border-bottom:2px solid transparent;padding:16px 4px;margin-bottom:-1px;cursor:pointer;transition:color .2s}
.tabs button:hover{color:var(--ink)}
.tabs button.on{color:var(--ink);border-bottom-color:var(--accent)}
main{max-width:1080px;margin:0 auto;padding:0 6vw 40px}
.secao{display:flex;align-items:baseline;gap:16px;margin:52px 0 22px;padding-bottom:14px;border-bottom:1px solid var(--line2);scroll-margin-top:72px}
.secao .num{font-family:'IBM Plex Mono',monospace;font-size:1.1rem;color:var(--accent);font-weight:600}
.secao h2{font-family:'IBM Plex Serif',serif;font-weight:600;font-size:1.6rem;margin:0;text-wrap:balance}
.texto{color:var(--ink);text-wrap:pretty;font-size:1.02rem;margin:0}
.texto.intro{color:var(--muted);max-width:74ch;margin:-4px 0 12px}
.texto b,.fontes b{color:var(--ink);font-weight:600}
code{font-family:'IBM Plex Mono',monospace;background:var(--surf);padding:1px 6px;border-radius:3px;font-size:.88em;color:var(--accent)}
.grade{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:20px}
.card{background:var(--paper);border:1px solid var(--line);border-radius:3px;padding:22px 24px 12px;display:flex;flex-direction:column}
.card.wide{grid-column:1/-1}
.card h3{font-family:'IBM Plex Serif',serif;font-weight:600;font-size:1.18rem;margin:0 0 4px;text-wrap:balance;min-height:1.4em}
.card .dek{color:var(--muted);font-size:.9rem;margin:0 0 12px;min-height:2.7em}
.card.wide .dek{min-height:0}
.flow{display:flex;flex-direction:column;gap:22px;margin-top:4px}
.flow .step{position:relative;display:grid;grid-template-columns:2.4em 1fr;gap:14px;align-items:baseline;border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:2px;padding:14px 18px;background:var(--paper)}
.flow .step .n{font-family:'IBM Plex Mono',monospace;font-size:.85rem;color:var(--accent);font-weight:600}
.flow .step .txt b{font-size:.98rem}
.flow .step .txt .d{color:var(--muted);font-size:.9rem;margin-left:8px}
.flow .step:not(:last-child)::after{content:'\\2193';position:absolute;left:1.2em;bottom:-17px;transform:translateX(-50%);color:var(--accent);font-size:1.05rem;line-height:1;font-weight:700}
@media(max-width:620px){.flow .step .txt .d{display:block;margin-left:0;margin-top:2px}}
.link-ext{margin:12px 0 2px}
.link-ext a{font-family:'IBM Plex Mono',monospace;font-size:.85rem;color:var(--accent);text-decoration:none;border-bottom:1px solid var(--accent);padding-bottom:1px}
.link-ext a:hover{color:var(--ink);border-color:var(--ink)}
.fontes{margin:0 0 10px;padding-left:18px;color:var(--muted)}
.fontes li{margin-bottom:6px}
table.modelos{width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums;font-size:.9rem;margin-bottom:12px}
table.modelos th,table.modelos td{padding:10px 12px;text-align:right;border-bottom:1px solid var(--line)}
table.modelos thead th{border-bottom:2px solid var(--ink);color:var(--muted);font-weight:600;text-transform:uppercase;font-size:.74rem;letter-spacing:.04em}
table.modelos th:first-child,table.modelos td:first-child{text-align:left;font-family:'IBM Plex Sans',sans-serif}
table.modelos tr.campeao td{background:#f0f5f9}
table.modelos tr.campeao td:first-child{color:var(--accent);font-weight:700;box-shadow:inset 3px 0 0 var(--accent)}
.metodo{list-style:none;counter-reset:m;padding:0;margin:0 0 12px;display:grid;gap:12px}
.metodo li{counter-increment:m;position:relative;padding-left:42px;color:var(--muted)}
.metodo li::before{content:counter(m,decimal-leading-zero);position:absolute;left:0;top:1px;font-family:'IBM Plex Mono',monospace;color:var(--accent);font-weight:600;font-size:.9rem}
.metodo b{color:var(--ink);font-weight:600}
.figura{border:1px solid var(--line);border-radius:3px;padding:14px;margin-bottom:12px;background:var(--paper)}
.figura img{width:100%;height:auto;display:block}
.perguntas{display:grid;gap:12px}
.qa-item{border:1px solid var(--line);border-radius:3px;overflow:hidden;background:var(--paper)}
.qa-item[open]{border-color:var(--line2)}
.qa-item summary{cursor:pointer;list-style:none;padding:18px 22px;font-family:'IBM Plex Serif',serif;font-weight:600;font-size:1.12rem;color:var(--ink);display:flex;gap:14px;align-items:baseline;text-wrap:balance}
.qa-item summary::-webkit-details-marker{display:none}
.qa-item summary::before{content:'+';font-family:'IBM Plex Mono',monospace;color:var(--accent);font-weight:600;font-size:1.2rem}
.qa-item[open] summary::before{content:'\\2212'}
.qa-item summary:hover{color:var(--accent)}
.qa-body{padding:0 22px 16px;border-top:1px solid var(--line)}
.qa-body .a{color:var(--muted);margin:12px 0 10px;text-wrap:pretty;max-width:78ch}
footer{max-width:1080px;margin:0 auto;padding:28px 6vw 72px;color:var(--muted);font-size:.84rem;border-top:2px solid var(--ink);text-wrap:pretty}
footer b{color:var(--ink)}
:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
</style></head><body>"""

def _foot(nome_campeao):
    return f"""<footer>
<b>Metodologia.</b> Modelo campeão: {nome_campeao} (pipeline Scikit-learn com imputação, codificação e padronização
integradas; validação estratificada; tuning por RandomizedSearchCV; interpretação por SHAP). As métricas no grão
individual são modestas por design — prever um aluno a partir de contexto agregado tem teto baixo; o valor está na
interpretação e na leitura municipal. Alvo derivado do rótulo da fonte: alunos ausentes na avaliação contam como não
alfabetizados (limitação documentada).<br><br>
<b>Fontes.</b> Base dos Dados · INEP (SAEB Alfabetização, Censo Escolar) · IBGE · Atlas do Desenvolvimento Humano
(PNUD).<br><b>Autor:</b> Theo Coleone de Camargo — Tech Challenge Fase 3, FIAP.
</footer>
</body></html>"""


if __name__ == "__main__":
    main()
