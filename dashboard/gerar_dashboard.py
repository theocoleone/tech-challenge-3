"""Gera o dashboard estático (index.html) do TC3 — estética de relatório estatístico oficial.

Fundo branco, quase monocromático, tipografia IBM Plex (Serif nos títulos, Mono nos
rótulos/números), seções numeradas, tabelas com fios e muito respiro. Duas abas (técnica e
negócio). Todo o conteúdo vem de `reports/` e `images/`; nenhum dado bruto é lido aqui.

Rodar da raiz do repo:  python dashboard/gerar_dashboard.py
"""
import base64
import json
import os
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

AQUI = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(AQUI, "..", "reports")
FIGURAS = os.path.join(REPORTS, "figuras")
FLOW_PNG = os.path.join(AQUI, "..", "images", "arquitetura_tc3.png")
OUTPUT = os.path.join(AQUI, "index.html")
GITHUB = "https://github.com/theocoleone/tech-challenge-3"

INK, MUTED, PAPER = "#16191d", "#5b6169", "#ffffff"
ACCENT, NEG, MID = "#1f5c8c", "#a6462e", "#9db9d0"
ESCALA = [[0.0, "#dce6ee"], [1.0, ACCENT]]
CFG = {"displayModeBar": False, "responsive": True}
_primeiro = [True]

AMIGAVEL = {
    "sigla_uf": "UF", "nome_regiao": "região", "amazonia_legal": "Amazônia Legal", "capital_uf": "capital da UF",
    "rede": "rede de ensino", "idhm": "IDHM", "idhm_e": "IDHM educação", "idhm_l": "IDHM longevidade",
    "idhm_r": "IDHM renda", "renda_pc": "renda per capita (2010)", "indice_gini": "índice de Gini",
    "prop_pobreza": "% de pobres (2010)", "expectativa_vida": "expectativa de vida",
    "taxa_analfabetismo_15_mais": "analfabetismo adulto", "taxa_freq_liquida_fundamental": "frequência líquida no fundamental",
    "taxa_agua_encanada": "domicílios com água encanada", "taxa_energia_eletrica": "domicílios com energia",
    "taxa_coleta_lixo": "coleta de lixo", "ivs": "IVS", "ivs_infraestrutura_urbana": "IVS infraestrutura urbana",
    "ivs_capital_humano": "IVS capital humano", "ivs_renda_trabalho": "IVS renda e trabalho",
    "populacao": "população", "log_populacao": "porte do município (log população)", "pib_per_capita": "PIB per capita",
    "log_pib_per_capita": "PIB per capita (log)", "esc_n_escolas": "nº de escolas públicas",
    "esc_prop_rural": "% de escolas rurais", "alunos_por_docente": "alunos por docente",
    "matriculas_por_sala": "matrículas por sala", "escolas_por_10k_hab": "escolas por 10 mil habitantes",
    "esc_media_quantidade_docente_fundamental_anos_iniciais": "docentes por escola",
    "esc_media_quantidade_matricula_fundamental_anos_iniciais": "matrículas por escola",
    "esc_media_quantidade_sala_utilizada": "salas por escola",
}
ITENS_ESCOLA = {
    "agua_rede_publica": "água da rede", "energia_rede_publica": "energia da rede", "esgoto_rede_publica": "esgoto da rede",
    "internet": "internet", "banda_larga": "banda larga", "biblioteca": "biblioteca",
    "laboratorio_informatica": "lab. de informática", "laboratorio_ciencias": "lab. de ciências",
    "quadra_esportes": "quadra", "refeitorio": "refeitório", "alimentacao": "alimentação escolar",
    "area_verde": "área verde", "patio_coberto": "pátio coberto",
}
REDES = {"1": "federal", "2": "estadual", "3": "municipal", "4": "privada"}


def cj(nome):
    with open(os.path.join(REPORTS, nome), encoding="utf-8") as f:
        return json.load(f)


def ccsv(nome, **kw):
    return pd.read_csv(os.path.join(REPORTS, nome), **kw)


def nome_amigavel(f):
    f = re.sub(r"^(num|cat)__", "", f)
    if f.startswith("sigla_uf_"):
        return "UF = " + f[len("sigla_uf_"):]
    if f.startswith("nome_regiao_"):
        return "região = " + f[len("nome_regiao_"):]
    if f.startswith("rede_"):
        return "rede " + REDES.get(f[len("rede_"):], f[len("rede_"):])
    if f.startswith("amazonia_legal_"):
        return "Amazônia Legal = " + ("sim" if f.endswith("1") else "não")
    if f.startswith("capital_uf_"):
        return "capital da UF = " + ("sim" if f.endswith("1") else "não")
    if f.startswith("esc_prop_") and f[len("esc_prop_"):] in ITENS_ESCOLA:
        return "% de escolas com " + ITENS_ESCOLA[f[len("esc_prop_"):]]
    return AMIGAVEL.get(f, f.replace("_", " "))


def fmt_int(v):
    return f"{int(v):,}".replace(",", ".")


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


def barras_h(serie, xtitulo, margem_esq=200, altura=340, cor=ACCENT, fmt=".3f"):
    s = serie.sort_values()
    fig = px.bar(x=s.values, y=s.index, orientation="h")
    fig.update_traces(marker_color=cor, hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>")
    fig.update_layout(xaxis_title=xtitulo, yaxis_title=None)
    estilizar(fig, altura)
    fig.update_layout(margin=dict(l=margem_esq, r=24, t=16, b=44))
    return fig


def card(titulo, dek, inner, wide=False):
    w = " wide" if wide else ""
    d = f"<p class='dek'>{dek}</p>" if dek else ""
    return f"<article class='card{w}'><h3>{titulo}</h3>{d}{inner}</article>"


def bloco(num, titulo):
    return f"<div class='secao'><span class='num'>{num}</span><h2>{titulo}</h2></div>"


def item(pergunta, resposta, chart):
    return (f"<details class='qa-item'><summary>{pergunta}</summary>"
            f"<div class='qa-body'><p class='a'>{resposta}</p>{chart}</div></details>")


def figura(caminho, alt):
    with open(caminho, "rb") as fp:
        b64 = base64.b64encode(fp.read()).decode()
    return f"<div class='figura'><img alt='{alt}' src='data:image/png;base64,{b64}'></div>"


def fig_relatorio(nome, alt):
    return figura(os.path.join(FIGURAS, nome), alt)


def tabela(df, colunas, destaque=None, classe="modelos"):
    """colunas: lista de (coluna, rótulo, formato). `destaque(row)` marca a linha como campeã."""
    th = "".join(f"<th>{rot}</th>" for _, rot, _ in colunas)
    linhas = ""
    for _, row in df.iterrows():
        tds = ""
        for col, _, f in colunas:
            v = row[col]
            if isinstance(v, str):
                s = v
            elif pd.isna(v):
                s = "—"
            elif callable(f):
                s = f(v)
            else:
                s = format(v, f)
            tds += f"<td>{s}</td>"
        cls = " class='campeao'" if destaque and destaque(row) else ""
        linhas += f"<tr{cls}>{tds}</tr>"
    return f"<table class='{classe}'><thead><tr>{th}</tr></thead><tbody>{linhas}</tbody></table>"


def main():
    agg = cj("agregados_base.json")
    cv = cj("cv_modelos.json")
    modelos = pd.DataFrame(cj("metricas_modelos.json"))
    tuning = cj("tuning.json")
    campeao = cj("metricas_campeao.json")
    nome_campeao = tuning["campeao"]
    treino_teste = cj("treino_vs_teste.json")
    comparacao = cj("comparacao_split.json")
    temporal = cj("validacao_temporal.json") if os.path.exists(os.path.join(REPORTS, "validacao_temporal.json")) else None
    threshold = cj("threshold.json")
    conf, roc = cj("confusao.json"), cj("roc.json")
    shap_var = pd.Series(cj("shap_por_variavel.json"))
    shap_imp = pd.Series(cj("shap_importancia.json")).head(15)
    importancias = pd.Series(cj("importancias.json")).head(15)
    perfil = pd.DataFrame(cj("perfil_clusters.json"))
    perfil_k3 = pd.DataFrame(cj("perfil_clusters_k3.json"))
    clustering = cj("clustering.json")
    proj = cj("projecao_metas.json")
    risco = ccsv("risco_municipios.csv")
    top20 = risco.head(20)
    sem_dado = ccsv("municipios_sem_dado_confiavel.csv")

    anos = f"{agg['anos'][0]}–{agg['anos'][-1]}"
    ano_ref = str(agg["anos"][-1])
    n_muni_ranking = len(risco)
    corr_risco = risco[[f"taxa_presentes_{ano_ref}", f"risco_previsto_{ano_ref}"]].corr().iloc[0, 1]

    kpis = [
        ("Alunos analisados", fmt_int(agg["n_alunos"]), f"2º ano do EF · {anos} · {fmt_int(agg['n_municipios'])} municípios"),
        ("Taxa de alfabetização", f"{agg['taxa_nacional']:.0%}",
         f"rótulo da fonte; {agg['taxa_nacional_presentes']:.0%} só entre presentes"),
        ("ROC-AUC em municípios novos", f"{campeao['roc_auc']:.2f}", f"{campeao['modelo']} · teste por município"),
        ("Municípios fora da meta 2030", f"{proj['pct_nao_atingem']:.0f}%",
         f"cenário de referência; {proj['faixa_pct_nao_atingem'][0]:.0f}–{proj['faixa_pct_nao_atingem'][1]:.0f}% nos três cenários"),
    ]

    # ---------------- gráficos descritivos ----------------
    reg = pd.Series(agg["taxa_por_regiao"]).sort_values()
    f_reg = px.bar(x=reg.values, y=reg.index, orientation="h", color=reg.values, color_continuous_scale=ESCALA)
    f_reg.update_layout(xaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None)
    f_reg.update_traces(hovertemplate="%{y}: %{x:.1%}<extra></extra>")

    uf = pd.Series(agg["taxa_por_uf"]).sort_values()
    uf_pres = pd.Series(agg["taxa_presentes_por_uf"]).reindex(uf.index)
    f_uf = go.Figure()
    f_uf.add_bar(name="rótulo da fonte (ausente = não alfabetizado)", x=uf.index, y=uf.values, marker_color=ACCENT)
    f_uf.add_bar(name="só alunos presentes", x=uf_pres.index, y=uf_pres.values, marker_color=MID)
    f_uf.update_layout(barmode="group", yaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None,
                       legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0))
    f_uf.update_traces(hovertemplate="%{x}: %{y:.1%}<extra></extra>")

    # ---------------- validação ----------------
    std = {m: cv["modelos"][m]["roc_auc"]["validacao_desvio"] for m in modelos.index}
    tr_cv = {m: cv["modelos"][m]["roc_auc"]["treino_media"] for m in modelos.index}
    ordem = modelos["roc_auc"].sort_values(ascending=False).index.tolist()
    f_cv = go.Figure()
    f_cv.add_bar(name="treino (média dos folds)", x=ordem, y=[tr_cv[m] for m in ordem], marker_color=MID)
    f_cv.add_bar(name="validação (municípios fora do fold)", x=ordem, y=[modelos.loc[m, "roc_auc"] for m in ordem],
                 marker_color=ACCENT, error_y=dict(type="data", array=[std[m] for m in ordem], color=INK, thickness=1))
    f_cv.update_layout(barmode="group", yaxis_title="ROC-AUC", xaxis_title=None, yaxis_range=[0.5, 0.72],
                       legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0))

    tab_modelos = modelos.copy()
    tab_modelos["desvio"] = pd.Series(std)
    tab_modelos["treino"] = pd.Series(tr_cv)
    tab_modelos = tab_modelos.reset_index().rename(columns={"index": "modelo"}).sort_values("roc_auc", ascending=False)
    cols_modelos = [("modelo", "Modelo", str), ("roc_auc", "ROC-AUC (val.)", ".3f"), ("desvio", "± desvio", ".3f"),
                    ("treino", "ROC-AUC (treino)", ".3f"), ("f1_macro", "F1-macro", ".3f"),
                    ("recall_nao_alf", "Recall (não alf.)", ".3f"), ("precisao_nao_alf", "Precisão (não alf.)", ".3f"),
                    ("auc_pr", "AUC-PR", ".3f")]
    html_modelos = tabela(tab_modelos, cols_modelos, destaque=lambda r: r["modelo"] == nome_campeao)

    linhas_val = [
        {"protocolo": "Teste por município (1.109 municípios nunca vistos)", **{k: v for k, v in campeao.items() if k != "modelo"}},
        {"protocolo": "Treino (mesmos municípios)", **{k: v for k, v in treino_teste["treino"].items() if k != "modelo"}},
        {"protocolo": "Split aleatório por aluno (municípios conhecidos)",
         **{k: v for k, v in comparacao["aleatorio_por_aluno"].items() if k != "modelo"}},
    ]
    if temporal:
        linhas_val.append({"protocolo": "Temporal: treino 2023 → teste 2024 (todas as UFs)",
                           **{k: v for k, v in temporal["teste_2024_todos"].items() if k != "modelo"}})
        linhas_val.append({"protocolo": f"Temporal: só UFs presentes em 2023 (sem {', '.join(temporal['ufs_sem_2023'])})",
                           **{k: v for k, v in temporal["teste_2024_ufs_com_2023"].items() if k != "modelo"}})
    tab_val = pd.DataFrame(linhas_val)
    n_teste = cv["protocolo"]["teste"]["municipios"]
    tab_val.loc[0, "protocolo"] = f"Teste por município ({fmt_int(n_teste)} municípios nunca vistos)"
    html_val = tabela(tab_val, [("protocolo", "Protocolo", str), ("roc_auc", "ROC-AUC", ".3f"),
                                ("f1_macro", "F1-macro", ".3f"), ("recall_nao_alf", "Recall (não alf.)", ".3f"),
                                ("acuracia", "Acurácia", ".3f")], destaque=lambda r: r["protocolo"].startswith("Teste por"))
    xgb_ref = comparacao.get("xgboost_referencia")
    texto_val = (f"<p class='texto'>O número que vale é o do <b>teste por município</b>: {fmt_int(n_teste)} municípios que o modelo "
                 f"nunca viu. A diferença para o split aleatório por aluno ({comparacao['diferenca_roc_auc']:+.3f} em ROC-AUC) "
                 f"mede quanto o modelo dependia de reconhecer o município.")
    if xgb_ref:
        texto_val += (f" O XGBoost ilustra o problema: {xgb_ref['roc_auc_teste_aleatorio']:.3f} no split aleatório, "
                      f"{xgb_ref['roc_auc_cv_por_municipio']:.3f} em municípios novos — a vantagem do boosting era memória "
                      f"de território, não sinal.")
    texto_val += "</p>"

    fin = tuning["finalistas"]
    itens_tuning = "".join(
        f"<li><b>{n}</b> — CV baseline {v['roc_auc_cv_baseline']:.4f}, tunado {v['roc_auc_cv_tunado']:.4f} "
        f"({'tuning mantido' if v['mantido'] == 'tunado' else 'baseline mantido'}); "
        f"melhores hiperparâmetros: <code>{', '.join(f'{k.replace('modelo__', '')}={val}' for k, val in v['melhores_hiperparametros'].items())}</code></li>"
        for n, v in fin.items())
    html_tuning = (f"<ul class='fontes'>{itens_tuning}</ul><p class='texto'>Critério: {tuning['criterio']}. "
                   f"Campeão: <b>{campeao['modelo']}</b>.</p>")

    z = [[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]]
    fcm = go.Figure(go.Heatmap(
        z=z, x=["previsto: não alf.", "previsto: alf."], y=["real: não alf.", "real: alf."],
        colorscale=[[0, "#eef3f7"], [1, "#8fb0cb"]], showscale=False,
        text=[[fmt_int(v) for v in r] for r in z], texttemplate="%{text}", textfont=dict(color=INK, size=16)))
    fcm.update_layout(yaxis_autorange="reversed")
    froc = go.Figure()
    froc.add_scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=f"AUC {roc['auc']:.2f}",
                     line=dict(color=ACCENT, width=2.5), fill="tozeroy", fillcolor="rgba(31,92,140,.10)")
    froc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#c9ccd1", dash="dash", width=1), showlegend=False)
    froc.update_layout(xaxis_title="taxa de falso positivo", yaxis_title="taxa de verdadeiro positivo", legend=dict(x=.55, y=.1))

    mp, me = threshold["metricas_corte_padrao"], threshold["metricas_corte_escolhido"]
    texto_corte = (f"<p class='texto'>Com o corte padrão (0,50) o modelo recupera {mp['recall_nao_alf']:.0%} dos alunos não "
                   f"alfabetizados. Para uma política de busca ativa, o corte de <b>{threshold['corte_escolhido']:.2f}</b> "
                   f"eleva o recall para <b>{me['recall_nao_alf']:.0%}</b> ao custo de precisão ({me['precisao_nao_alf']:.0%} vs "
                   f"{mp['precisao_nao_alf']:.0%}) e de sinalizar {threshold['prop_alunos_sinalizados_escolhido']:.0%} dos alunos "
                   f"em vez de {threshold['prop_alunos_sinalizados_padrao']:.0%}. O corte é uma decisão de gestão, não do modelo.</p>")

    # ---------------- interpretabilidade ----------------
    shap_var.index = [nome_amigavel(i) for i in shap_var.index]
    shap_imp.index = [nome_amigavel(i) for i in shap_imp.index]
    importancias.index = [nome_amigavel(i) for i in importancias.index]
    f_shap_var = barras_h(shap_var, "|SHAP| médio (soma)", margem_esq=230)
    f_shap = barras_h(shap_imp, "|SHAP| médio", margem_esq=230)
    tipo_imp = "|coeficiente| (features padronizadas)" if "Logística" in nome_campeao else "ganho"
    f_imp = barras_h(importancias, "importância", margem_esq=230)
    share_uf = shap_var.get("UF", 0) / shap_var.sum()
    contexto_top = [n for n in shap_var.index if n not in ("UF", "região", "Amazônia Legal", "capital da UF", "rede de ensino")][:3]
    infra_top = [n for n in shap_var.index if n.startswith("% de escolas") or n in
                 ("nº de escolas públicas", "alunos por docente", "matrículas por sala", "docentes por escola", "escolas por 10 mil habitantes")][:3]

    # ---------------- estratégia ----------------
    perfil = perfil.sort_values("taxa_alf")
    vuln, dev = perfil.iloc[0], perfil.iloc[-1]
    f_clu = go.Figure()
    rot = [f"{r['rotulo']} ({fmt_int(r['n_municipios'])})" for _, r in perfil.iterrows()]
    f_clu.add_bar(name="taxa de alfabetização", x=rot, y=perfil["taxa_alf"], marker_color=ACCENT)
    f_clu.add_bar(name="risco previsto (modelo)", x=rot, y=perfil["risco_previsto"], marker_color=NEG)
    f_clu.add_bar(name="IDHM", x=rot, y=perfil["idhm"], marker_color=MID)
    f_clu.update_layout(barmode="group", yaxis_title=None, legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0))
    tab_k3 = perfil_k3.sort_values("taxa_alf").reset_index(drop=True)
    html_k3 = tabela(tab_k3, [("rotulo", "Perfil (k=3)", str), ("n_municipios", "Municípios", fmt_int),
                              ("taxa_alf", "Taxa", ".1%"), ("risco_previsto", "Risco previsto", ".2f"), ("idhm", "IDHM", ".2f"),
                              ("prop_pobreza", "% pobres", ".0f"), ("esc_prop_rural", "% escolas rurais", ".0%"),
                              ("ufs_predominantes", "UFs predominantes", str)])

    cen = proj["cenarios"]
    nomes_cen = {"mantem_2024": "Mantém a taxa de 2024", "tendencia_uf": "Segue a tendência da UF",
                 "tendencia_encolhida": "Tendência própria, encolhida para a da UF (referência)"}
    tab_cen = pd.DataFrame([{"cenario": nomes_cen[k], **v} for k, v in cen.items()])
    html_cen = tabela(tab_cen, [("cenario", "Cenário", str), ("municipios_nao_atingem", "Não atingem", fmt_int),
                                ("pct_nao_atingem", "% dos avaliados", ".1f"), ("gap_medio_pp", "Gap médio (p.p.)", "+.1f")],
                      destaque=lambda r: "referência" in r["cenario"])
    atingem = 100 - proj["pct_nao_atingem"]
    f_meta = go.Figure(go.Pie(labels=["Não atingem", "Atingem"], values=[proj["pct_nao_atingem"], atingem],
                              hole=0.62, marker_colors=[NEG, ACCENT], sort=False, textfont=dict(color="#ffffff")))

    cols_top = [("nome_municipio", "Município", str), ("sigla_uf", "UF", str),
                (f"n_alunos_{ano_ref}", "Alunos", fmt_int), (f"presenca_{ano_ref}", "Participação", ".0%"),
                (f"taxa_presentes_{ano_ref}", "Alfabetizados (presentes)", ".1%"),
                (f"taxa_alf_{ano_ref}", "Alfabetizados (rótulo)", ".1%"),
                (f"risco_previsto_{ano_ref}", "Risco previsto", ".2f"), ("idhm", "IDHM", ".2f")]
    html_top = tabela(top20, cols_top)
    ne = int((top20["nome_regiao"] == "Nordeste").sum())
    ufs_top = ", ".join(f"{u} ({n})" for u, n in top20["sigla_uf"].value_counts().head(4).items())
    motivos = sem_dado["motivo"].value_counts()
    html_sem_dado = "<ul class='fontes'>" + "".join(f"<li>{fmt_int(n)} — {m}</li>" for m, n in motivos.items()) + "</ul>"

    for f in (f_reg, f_uf, f_cv, fcm, froc, f_clu, f_meta):
        estilizar(f)
    f_uf.update_layout(height=380, margin=dict(l=64, r=24, t=46, b=44))
    f_cv.update_layout(margin=dict(l=64, r=24, t=46, b=44))
    f_clu.update_layout(margin=dict(l=64, r=44, t=46, b=44))

    # ---------------- textos fixos ----------------
    contexto = (f"<p class='texto'>Construí um <b>modelo supervisionado</b> que prevê se um aluno do 2º ano do Ensino "
                f"Fundamental está <b>alfabetizado</b> ou <b>não alfabetizado</b> a partir de variáveis educacionais, "
                f"territoriais e socioeconômicas do seu município, e uma camada de análise que traduz o resultado em "
                f"<b>inteligência para a decisão pública</b>: quais fatores pesam, quais municípios estão em risco, quais "
                f"territórios se parecem e quem tende a não alcançar a meta de 2030.</p>"
                f"<p class='link-ext'><a href='{GITHUB}' target='_blank' rel='noopener'>Código no GitHub <span aria-hidden='true'>↗</span></a></p>")
    tc2 = ("<p class='texto'>Este projeto parte do data lake do <b>Tech Challenge 2</b>, uma pipeline em nuvem com "
           "<b>Arquitetura Medalhão</b> no S3: ingestão do BigQuery, camadas Bronze → Silver → Gold com AWS Glue, um caminho de "
           "streaming (SQS + Lambda) e um dashboard em CloudFront. O TC3 consome a <b>Silver de alunos</b> (grão individual, "
           "necessário para prever o aluno) e a <b>Gold de indicadores municipais</b> (taxas e metas 2030).</p>"
           "<p class='link-ext'><a href='https://d131u8q9uidloe.cloudfront.net' target='_blank' rel='noopener'>"
           "Abrir o dashboard do TC2 <span aria-hidden='true'>↗</span></a></p>")
    etapas = [("Base integrada", "Silver de alunos + Censo Escolar (rede pública) + Atlas ADH/IVS + IBGE, por município e ano"),
              ("EDA", "cobertura, presença na prova, variação anual, correlações e desigualdade; hipóteses H1–H6"),
              ("Pipeline", "engenharia de atributos, imputação, One-Hot e padronização dentro do Pipeline"),
              ("Validação por município", "treino e teste com municípios disjuntos; 5 folds; tuning dos dois melhores"),
              ("Avaliação", "teste único, corte de decisão, robustez (split aleatório e temporal), SHAP"),
              ("Aplicação", "clusters, ranking de risco com filtro de participação, cenários de meta 2030")]
    passos = "".join(f"<div class='step'><span class='n'>{i + 1:02d}</span><div class='txt'><b>{t}</b><span class='d'>{d}</span></div></div>"
                     for i, (t, d) in enumerate(etapas))
    flow = f"<div class='flow'>{passos}</div>"
    fontes = (f"<ul class='fontes'>"
              f"<li><b>SAEB Alfabetização (Silver do TC2)</b> — {fmt_int(agg['n_alunos'])} alunos, {anos}; alvo, presença, rede e UF.</li>"
              "<li><b>Censo Escolar (INEP)</b> — infraestrutura, docentes e matrículas das escolas públicas de anos iniciais, agregados por município e ano.</li>"
              "<li><b>Atlas do Desenvolvimento Humano (PNUD) e IVS (Ipea)</b> — IDHM e subíndices, renda, Gini, pobreza, saneamento, vulnerabilidade.</li>"
              "<li><b>IBGE</b> — população e PIB per capita; diretório com região, Amazônia Legal e capital.</li>"
              "</ul><p class='texto'>Tudo integrado por <code>id_municipio</code> (e ano, no Censo). Fora das features: tudo que é "
              "medido no exame (proficiência, presença) ou derivado do alvo.</p>")
    metodo = "<ol class='metodo'>" + "".join(f"<li>{s}</li>" for s in [
        "<b>Imputação de faltantes</b> — mediana nas numéricas, constante nas categóricas (SimpleImputer).",
        "<b>Transformação de variáveis</b> — One-Hot nas categóricas, padronização nas numéricas.",
        "<b>Engenharia de atributos</b> — alunos por docente, matrículas por sala, escolas por 10 mil habitantes, log de população e PIB, dentro do Pipeline.",
        "<b>Tratamento de data leakage</b> — colunas do exame barradas por lista (<code>COLUNAS_LEAKAGE</code>) com verificação em código; "
        "treino e teste separados <b>por município</b>, porque as features são municipais e alunos do mesmo município não são independentes.",
        "<b>Pré-processamento integrado ao modelo</b> — Pipeline com FunctionTransformer + ColumnTransformer + estimador, ajustado só no treino.",
        "<b>Treinamento e validação</b> — StratifiedGroupKFold(5) para comparar 4 algoritmos; RandomizedSearchCV nos dois melhores; baseline mantido quando vence.",
        "<b>Replicabilidade e generalização</b> — semente fixa; teste tocado uma única vez; robustez com split aleatório e validação temporal 2023 → 2024.",
        f"<b>Corte de decisão</b> — escolhido para recall de {threshold['recall_alvo_nao_alf']:.0%} na classe não alfabetizado, com o trade-off explícito.",
    ]) + "</ol>"

    ufs_sem_2023 = [u for u, d in agg["alunos_por_uf_ano"].items() if str(agg["anos"][0]) not in d]
    texto_cobertura = (f"<p class='texto'>{', '.join(ufs_sem_2023)} só entram em {ano_ref}, com "
                       f"{fmt_int(sum(agg['alunos_por_uf_ano'][u][ano_ref] for u in ufs_sem_2023))} alunos. Por isso <code>ano</code> "
                       f"ficou fora das features e a comparação anual só vale entre UFs com os dois anos.</p>")
    texto_presenca = (f"<p class='texto'>{agg['presenca']:.1%} dos alunos estiveram presentes e todo ausente conta como não "
                      f"alfabetizado no rótulo da fonte: {agg['taxa_nacional']:.1%} de alfabetizados, ou "
                      f"{agg['taxa_nacional_presentes']:.1%} só entre presentes. O modelo aprende o indicador que o gestor recebe; a "
                      f"camada municipal filtra participação antes de ranquear.</p>")
    a0, a1 = str(agg["anos"][0]), ano_ref
    var = {u: d[a1]["taxa"] - d[a0]["taxa"] for u, d in agg["por_uf_ano"].items() if a0 in d and a1 in d}
    pior = min(var, key=var.get)
    dp = agg["por_uf_ano"][pior]
    texto_variacao = (f"<p class='texto'>Entre as UFs com os dois anos, a mediana da variação é "
                      f"{pd.Series(var).median() * 100:+.1f} p.p. O extremo é {pior} ({var[pior] * 100:+.1f} p.p.), com presença "
                      f"estável ({dp[a0]['presenca']:.0%} → {dp[a1]['presenca']:.0%}) e queda entre presentes "
                      f"({dp[a0]['taxa_presentes']:.1%} → {dp[a1]['taxa_presentes']:.1%}): queda de resultado medido, não de participação. "
                      f"Uma diferença de um único ano não é tendência, e é por isso que a projeção de metas usa cenários.</p>")

    # ================= montagem =================
    p = [_HEAD, _hero(kpis, anos), _tabbar(), "<main id='main'>"]

    p += ["<section id='tab-tecnica' role='tabpanel' aria-label='Visão técnica'>"]
    p += [bloco("1", "O que foi construído"), "<div class='grade'>",
          card("Objetivo do projeto", "Da predição do aluno à inteligência para a gestão.", contexto, wide=True),
          card("Herança do Tech Challenge 2", "", tc2, wide=True),
          card("Etapas do projeto", "", flow, wide=True),
          card("Como a base foi integrada", "", fontes, wide=True),
          card("Como foi feito", "", metodo, wide=True), "</div>"]
    p += [bloco("2", "O que a análise exploratória mostrou"), "<div class='grade'>",
          card("Alfabetização por UF: rótulo da fonte vs só presentes", "Quanto da diferença entre estados é participação.", div(f_uf), wide=True),
          card("Cobertura por UF e ano", "", texto_cobertura + fig_relatorio("14_cobertura_uf_ano.png", "Alunos avaliados por UF e ano")),
          card("Variação 2023 → 2024 por UF", "", texto_variacao + fig_relatorio("15_variacao_uf.png", "Variação 2023-2024 por UF")),
          card("Presença na prova e sensibilidade do rótulo", "", texto_presenca),
          card("Alfabetização por decil de IDHM", "A curva é plana acima do 3º decil: o contexto socioeconômico só separa a base da distribuição.",
               fig_relatorio("06_idhm_decil.png", "Alfabetização por decil de IDHM")),
          card("Taxa por região", "", div(f_reg)),
          "</div>", "<p class='texto intro'>Resumo completo, com tabelas e as hipóteses H1–H6 que orientaram a modelagem, em "
          f"<a href='{GITHUB}/blob/main/reports/eda_resumo.md' target='_blank' rel='noopener'>reports/eda_resumo.md</a>.</p>"]
    p += [bloco("3", "Validação e modelo"), "<div class='grade'>",
          card("Comparação de modelos — validação cruzada por município (5 folds)",
               f"{nome_campeao} campeã: com municípios fora do fold, os modelos de árvore perdem a vantagem e o treino descola da validação.",
               html_modelos, wide=True),
          card("Treino vs validação", "Barras de erro = desvio entre folds.", div(f_cv)),
          card("Tuning dos finalistas", "", html_tuning),
          card("Protocolos de validação do campeão", "", html_val + texto_val, wide=True),
          card("Matriz de confusão (teste por município)", "", div(fcm)),
          card("Curva ROC", f"Área sob a curva = {roc['auc']:.2f}.", div(froc)),
          card("Corte de decisão para a classe não alfabetizado", "", texto_corte + fig_relatorio("18_threshold.png", "Precisão e recall por corte"), wide=True),
          "</div>"]
    p += [bloco("4", "Interpretabilidade"), "<div class='grade'>",
          card("Influência por variável (SHAP somado por variável original)",
               f"UF concentra {share_uf:.0%} da influência somada; depois vêm {', '.join(contexto_top)}.", div(f_shap_var)),
          card("Influência por feature (|SHAP| médio)", "One-hots de UF aparecem separadas.", div(f_shap)),
          card(f"Importância no modelo ({tipo_imp})", "", div(f_imp)),
          card("Resumo SHAP", "Cada ponto é um aluno; a cor é o valor da feature.", fig_relatorio("09_shap_summary.png", "SHAP summary plot")),
          "</div>"]
    p += [bloco("5", "Arquitetura"), "<div class='grade'>",
          card("Serviços e fluxo de dados",
               "Execução local em Python; dados, modelo e relatórios no S3; dashboard em bucket privado servido pelo CloudFront (OAC).",
               figura(FLOW_PNG, "Fluxograma da arquitetura: BigQuery e S3 para execução local, artefatos no S3, CloudFront"), wide=True),
          "</div>"]
    p += ["</section>"]

    p += ["<section id='tab-business' role='tabpanel' aria-label='Visão de negócio' hidden>"]
    p += [bloco("1", "Inteligência aplicada"), "<div class='perguntas'>",
          item("1 · Quais fatores mais impactam a alfabetização?",
               f"O lugar. A UF do aluno concentra {share_uf:.0%} da influência somada das 15 variáveis mais importantes, "
               f"{shap_var.iloc[0] / shap_var.iloc[1]:.1f} vezes a segunda colocada. Entre as variáveis de contexto, pesam "
               f"{', '.join(contexto_top)}. Infraestrutura escolar aparece, mas atrás: "
               f"{', '.join(infra_top) if infra_top else 'nenhum item entra entre as 15 variáveis mais influentes'}.",
               div(f_shap_var)),
          item("2 · Quais municípios apresentam maior risco educacional?",
               f"Os 20 municípios com o menor indicador de {ano_ref} (alunos presentes), entre {fmt_int(n_muni_ranking)} com pelo menos 50 alunos e "
               f"participação de 50% ou mais. {ne} dos 20 estão no Nordeste ({ufs_top}). O risco previsto pelo modelo — probabilidade média "
               f"de não alfabetizado, calculada só a partir do contexto do município — acompanha o indicador observado "
               f"(correlação {corr_risco:.2f}). Lista completa em reports/risco_municipios.csv.",
               html_top),
          item("3 · Quais regiões possuem padrões semelhantes?",
               f"O agrupamento separa dois perfis: <b>{vuln['rotulo']}</b> ({fmt_int(vuln['n_municipios'])} municípios; taxa {vuln['taxa_alf']:.0%}, IDHM "
               f"{vuln['idhm']:.2f}, {vuln['prop_pobreza']:.0f}% de pobres, {vuln['esc_prop_rural']:.0%} de escolas rurais; {vuln['ufs_predominantes']}) e "
               f"<b>{dev['rotulo']}</b> ({fmt_int(dev['n_municipios'])}; taxa {dev['taxa_alf']:.0%}, IDHM {dev['idhm']:.2f}, {dev['prop_pobreza']:.0f}% de pobres). "
               f"A silhouette é {clustering['silhouette_por_k']['2']:.2f}: a estrutura é fraca, um corte do gradiente de IDHM mais do que grupos nítidos; "
               f"com três perfis surge um grupo intermediário.",
               div(f_clu) + html_k3),
          item("4 · Como prever municípios que podem não atingir metas futuras?",
               f"Com cenários, não com uma reta. Entre {fmt_int(proj['municipios_avaliados'])} municípios com dados confiáveis, de "
               f"{proj['faixa_pct_nao_atingem'][0]:.0f}% a {proj['faixa_pct_nao_atingem'][1]:.0f}% não alcançam a meta de 2030 conforme o cenário; "
               f"na referência (tendência própria encolhida para a da UF, variação anual limitada a ±{proj['limite_velocidade_pp_ano']:.0f} p.p.) são "
               f"{fmt_int(proj['municipios_nao_atingem_meta_2030'])} ({proj['pct_nao_atingem']:.0f}%). A lista por município está em reports/metas_2030_municipios.csv.",
               div(f_meta) + html_cen),
          item("5 · Quais variáveis possuem maior influência nos modelos?",
               f"No modelo {nome_campeao}, as maiores magnitudes são de UFs específicas — o Ceará, positivo; RN, BA e SE, negativos — seguidas de "
               f"{', '.join([n for n in importancias.sort_values(ascending=False).index if not n.startswith('UF')][:3])}. "
               f"Um modelo linear regularizado venceu os de árvore em municípios novos: sinal simples, generalização melhor.",
               div(f_imp)),
          "</div>"]
    p += [bloco("2", "Onde agir primeiro"), "<div class='perguntas'>",
          item("Priorizar recursos nos territórios mais vulneráveis",
               f"Concentrar esforço nos {fmt_int(vuln['n_municipios'])} municípios do perfil {vuln['rotulo']} e nos 20 do topo do ranking, em vez de "
               f"distribuir de forma uniforme. Norte e Nordeste concentram as menores taxas.", div(f_reg)),
          item("Replicar o que funciona — a política do Ceará",
               f"O Ceará tem {agg['taxa_por_uf']['CE']:.0%} de alfabetizados com IDHM médio-baixo; é a UF que mais move o modelo para cima. "
               "Evidência de que ação pública consistente (o PAIC) supera o contexto socioeconômico.", div(f_uf)),
          item("Monitorar as metas e agir de forma preventiva",
               f"Acompanhar os {fmt_int(proj['municipios_nao_atingem_meta_2030'])} municípios fora da rota de 2030 no cenário de referência e agir antes "
               f"da próxima avaliação. Manter a taxa atual deixaria {proj['cenarios']['mantem_2024']['pct_nao_atingem']:.0f}% fora da meta.", div(f_meta)),
          item("Garantir a avaliação onde falta dado confiável",
               f"{fmt_int(len(sem_dado))} municípios ficaram fora do ranking por falta de dado confiável em {ano_ref}. Sem medição não há gestão: "
               "esses são os primeiros a resolver.", html_sem_dado),
          "</div>"]
    p += ["</section>"]

    p += ["</main>", _SCRIPT, _foot(campeao["modelo"])]
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(p))
    print(f"dashboard salvo -> {os.path.relpath(OUTPUT)}  ({os.path.getsize(OUTPUT) // 1024} KB)")


def _hero(kpis, anos):
    tiles = "".join(f"<div class='kpi'><span class='v'>{v}</span><span class='l'>{l}</span>"
                    f"<span class='h'>{h}</span></div>" for l, v, h in kpis)
    return f"""<a href="#main" class="skip">Pular para o conteúdo</a>
<header class="hero">
  <p class="eyebrow">Tech Challenge · Fase 3 · Pós-graduação AI Scientist — FIAP</p>
  <h1>Quem o Brasil está alfabetizando?</h1>
  <p class="dek">Um modelo supervisionado no grão do aluno, validado em municípios que ele nunca viu, e uma leitura
  estratégica por município sobre a base pública do 2º ano do Ensino Fundamental enriquecida com Censo Escolar,
  Atlas do Desenvolvimento Humano e IBGE.</p>
  <p class="assinatura">Theo Coleone de Camargo · dados {anos}</p>
</header>
<section class="kpis" aria-label="Indicadores-chave">{tiles}</section>"""


def _tabbar():
    return ("<div class='tabs' role='tablist' aria-label='Visões do projeto'>"
            "<button role='tab' id='aba-tecnica' aria-controls='tab-tecnica' data-tab='tecnica' aria-selected='true'>Visão técnica</button>"
            "<button role='tab' id='aba-business' aria-controls='tab-business' data-tab='business' aria-selected='false'>Visão de negócio</button>"
            "</div>")


_SCRIPT = """<script>
(function(){
  var tabs=[].slice.call(document.querySelectorAll('[role=tab]'));
  var panels={tecnica:document.getElementById('tab-tecnica'),business:document.getElementById('tab-business')};
  function activate(name){
    tabs.forEach(function(t){var on=t.dataset.tab===name;t.setAttribute('aria-selected',on);t.classList.toggle('on',on);});
    for(var k in panels){panels[k].hidden=(k!==name);}
    if(history.replaceState){history.replaceState(null,'','#'+name);}
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
  activate(location.hash==='#business'?'business':'tecnica');
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
<b>Metodologia.</b> Modelo campeão: {nome_campeao}, escolhido por validação cruzada com folds por município e avaliado uma
única vez em municípios que não entraram no treino. Pipeline Scikit-learn com engenharia de atributos, imputação,
codificação e padronização integradas; tuning por RandomizedSearchCV; interpretação por SHAP. As métricas no grão individual
são modestas por natureza — prever um aluno a partir do contexto do município tem teto baixo; o valor está na interpretação e
na leitura municipal. O alvo é o rótulo da fonte, em que alunos ausentes contam como não alfabetizados; a camada municipal
filtra participação antes de ranquear ou projetar.<br><br>
<b>Fontes.</b> Base dos Dados · INEP (SAEB Alfabetização, Censo Escolar) · IBGE · Atlas do Desenvolvimento Humano (PNUD) ·
IVS (Ipea).<br><b>Autor:</b> Theo Coleone de Camargo — Tech Challenge Fase 3, FIAP.
</footer>
</body></html>"""


if __name__ == "__main__":
    main()
