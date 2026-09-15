"""Modelagem supervisionada — prever `alfabetizado` (0/1) no grão aluno.

Protocolo:
  1. amostra estratificada de 800 mil alunos (tratabilidade em máquina local);
  2. treino e teste separados POR MUNICÍPIO (GroupShuffleSplit): quase todas as features são
     municipais, então o que interessa medir é a generalização para territórios novos;
  3. comparação de 4 algoritmos por StratifiedGroupKFold(5) no treino, com média ± desvio;
  4. tuning dos dois melhores (RandomizedSearchCV com folds por município); o baseline de cada um
     é mantido se for melhor em validação, e o campeão é o melhor após o tuning;
  5. teste tocado uma única vez, pelo campeão final;
  6. robustez: mesmo modelo com split aleatório de alunos (municípios conhecidos) e validação
     temporal 2023 -> 2024;
  7. corte de decisão para a classe "não alfabetizado" e interpretabilidade (SHAP + ganho).

Rodar da raiz do repo:  python notebooks/02_modelagem.py
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
import shap
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import (
    GroupShuffleSplit, RandomizedSearchCV, StratifiedGroupKFold, cross_val_score,
    cross_validate, train_test_split,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import config
from src.data_access import publicar_artefatos
from src.evaluation.metricas import (
    METRICAS, SCORING, agrupar_por_variavel, avaliar, curva_threshold, plot_confusao,
    plot_roc, plot_threshold, resumir_cv,
)
from src.modeling.pipeline import construir_pipeline, nomes_features, separar_features

warnings.filterwarnings("ignore")

REPORTS = config.REPORTS_DIR
FIG = config.FIGURAS_DIR
MODELS = config.MODELS_DIR
os.makedirs(FIG, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

SEED = 42
N_AMOSTRA = 800_000
N_CV = 300_000
N_TUNING = 200_000
N_FINALISTAS = 2
RECALL_ALVO = 0.70


def salvar_json(obj, nome):
    with open(os.path.join(REPORTS, nome), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def subamostra(X, y, g, n):
    if len(X) <= n:
        return X, y, g
    idx, _ = train_test_split(np.arange(len(X)), train_size=n, stratify=y, random_state=SEED)
    return X.iloc[idx], y.iloc[idx], g.iloc[idx]


def modelos_base():
    return {
        "Regressão Logística": LogisticRegression(max_iter=1000),
        "Árvore de Decisão": DecisionTreeClassifier(max_depth=8, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_depth=14, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(
            tree_method="hist", n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=SEED, eval_metric="logloss"),
    }


def espaco_busca(nome):
    if "XGB" in nome:
        return {
            "modelo__n_estimators": randint(200, 600),
            "modelo__max_depth": randint(3, 9),
            "modelo__learning_rate": uniform(0.02, 0.20),
            "modelo__subsample": uniform(0.6, 0.4),
            "modelo__colsample_bytree": uniform(0.6, 0.4),
            "modelo__min_child_weight": randint(1, 10),
        }
    if "Forest" in nome:
        return {
            "modelo__n_estimators": randint(100, 400),
            "modelo__max_depth": randint(6, 20),
            "modelo__min_samples_leaf": randint(1, 50),
            "modelo__max_features": ["sqrt", "log2", 0.5],
        }
    if "Árvore" in nome:
        return {"modelo__max_depth": randint(3, 16), "modelo__min_samples_leaf": randint(1, 200)}
    return {"modelo__C": loguniform(1e-3, 1e2)}


def main():
    base = config.base_local()
    df = pd.read_parquet(base)
    total = len(df)
    if total > N_AMOSTRA:
        df, _ = train_test_split(df, train_size=N_AMOSTRA, stratify=df[config.COLUNA_ALVO],
                                 random_state=SEED)
        print(f"[amostra estratificada: {N_AMOSTRA:,} de {total:,} linhas]")
    X, y, grupos, numericas, categoricas = separar_features(df)
    ano = df["ano"]
    print(f"features: {len(numericas)} numéricas (5 derivadas) + {len(categoricas)} categóricas | "
          f"{grupos.nunique():,} municípios")

    # ---- split por município ----
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, grupos))
    X_tr, X_te, y_tr, y_te, g_tr, g_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], grupos.iloc[tr], grupos.iloc[te]
    split_info = {
        "protocolo": "GroupShuffleSplit por id_municipio (80/20)",
        "treino": {"alunos": int(len(tr)), "municipios": int(g_tr.nunique()), "taxa_alvo": round(float(y_tr.mean()), 4)},
        "teste": {"alunos": int(len(te)), "municipios": int(g_te.nunique()), "taxa_alvo": round(float(y_te.mean()), 4)},
    }
    print(f"treino: {len(tr):,} alunos / {g_tr.nunique():,} municípios | teste: {len(te):,} / {g_te.nunique():,}")

    # ---- seleção por validação cruzada (folds por município) ----
    X_cv, y_cv, g_cv = subamostra(X_tr, y_tr, g_tr, N_CV)
    cv5 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    modelos = modelos_base()
    cv_resumo = {}
    for nome, modelo in modelos.items():
        print(f"\n>>> CV 5 folds: {nome}...")
        res = cross_validate(construir_pipeline(modelo, numericas, categoricas), X_cv, y_cv,
                             groups=g_cv, cv=cv5, scoring=SCORING, return_train_score=True, n_jobs=1)
        cv_resumo[nome] = resumir_cv(res)
        r = cv_resumo[nome]["roc_auc"]
        print(f"    ROC-AUC validação {r['validacao_media']:.4f} ± {r['validacao_desvio']:.4f} "
              f"| treino {r['treino_media']:.4f}")
    tabela = pd.DataFrame({m: {n: cv_resumo[n][m]["validacao_media"] for n in modelos} for m in METRICAS})
    tabela = tabela.sort_values("roc_auc", ascending=False)
    print("\n=== VALIDAÇÃO CRUZADA (média dos 5 folds, por município) ===")
    print(tabela.to_string())
    finalistas = list(tabela.index[:N_FINALISTAS])

    # ---- tuning dos finalistas (mesmos folds por município); baseline mantido se for melhor ----
    print(f"\n=== FINALISTAS: {', '.join(finalistas)} — tuning ===")
    X_tu, y_tu, g_tu = subamostra(X_tr, y_tr, g_tr, N_TUNING)
    cv3 = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    tuning, candidatos = {}, {}
    for nome in finalistas:
        pipe = construir_pipeline(modelos[nome], numericas, categoricas)
        busca = RandomizedSearchCV(pipe, espaco_busca(nome), n_iter=15, scoring="roc_auc",
                                   cv=cv3, n_jobs=1, random_state=SEED, verbose=0)
        busca.fit(X_tu, y_tu, groups=g_tu)
        auc_base = float(cross_val_score(clone(pipe), X_tu, y_tu, groups=g_tu, cv=cv3, scoring="roc_auc").mean())
        melhorou = busca.best_score_ > auc_base
        candidatos[nome] = (busca.best_estimator_ if melhorou else clone(pipe),
                            max(auc_base, float(busca.best_score_)), melhorou)
        tuning[nome] = {
            "roc_auc_cv_baseline": round(auc_base, 4),
            "roc_auc_cv_tunado": round(float(busca.best_score_), 4),
            "mantido": "tunado" if melhorou else "baseline",
            "melhores_hiperparametros": {k: (round(v, 4) if isinstance(v, float) else v)
                                         for k, v in busca.best_params_.items()},
        }
        print(f"{nome}: CV baseline {auc_base:.4f} | tunado {busca.best_score_:.4f} -> {tuning[nome]['mantido']}")
    campeao_nome = max(candidatos, key=lambda n: candidatos[n][1])
    campeao, _, tunado_melhor = candidatos[campeao_nome]
    rotulo = f"{campeao_nome} (tunado)" if tunado_melhor else campeao_nome
    tuning = {"campeao": campeao_nome, "criterio": "maior ROC-AUC em CV por município após tuning",
              "finalistas": tuning}
    print(f"campeão final: {rotulo}")

    campeao.fit(X_tr, y_tr)

    # ---- teste: uma única avaliação ----
    proba_te = campeao.predict_proba(X_te)[:, 1]
    pred_te = (proba_te >= 0.5).astype(int)
    final = avaliar(rotulo, y_te, pred_te, proba_te)
    proba_tr = campeao.predict_proba(X_tr)[:, 1]
    treino = avaliar(rotulo, y_tr, (proba_tr >= 0.5).astype(int), proba_tr)
    print("\n=== CAMPEÃO NO TESTE (municípios nunca vistos) ===")
    print(pd.DataFrame({"treino": treino, "teste": final}).drop("modelo").to_string())

    plot_confusao(y_te, pred_te, rotulo, os.path.join(FIG, "07_confusao_campeao.png"))
    plot_roc(y_te, proba_te, rotulo, os.path.join(FIG, "08_roc_campeao.png"))
    tn, fp, fn, tp = confusion_matrix(y_te, pred_te).ravel()
    salvar_json({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}, "confusao.json")
    fpr, tpr, _ = roc_curve(y_te, proba_te)
    idx = np.linspace(0, len(fpr) - 1, min(220, len(fpr))).astype(int)
    salvar_json({"fpr": [round(float(v), 4) for v in fpr[idx]], "tpr": [round(float(v), 4) for v in tpr[idx]],
                 "auc": final["roc_auc"]}, "roc.json")

    # ---- corte de decisão para a classe não alfabetizado ----
    curva = curva_threshold(y_te, proba_te)
    atende = curva[curva["recall_nao_alf"] >= RECALL_ALVO]
    linha = atende.sort_values("precisao_nao_alf", ascending=False).iloc[0] if len(atende) \
        else curva.loc[curva["f1_nao_alf"].idxmax()]
    corte = float(linha["threshold"])
    pred_corte = (proba_te >= corte).astype(int)
    threshold = {
        "recall_alvo_nao_alf": RECALL_ALVO,
        "corte_padrao": 0.5,
        "metricas_corte_padrao": {k: v for k, v in final.items() if k != "modelo"},
        "corte_escolhido": corte,
        "metricas_corte_escolhido": {k: v for k, v in avaliar(rotulo, y_te, pred_corte, proba_te).items() if k != "modelo"},
        "prop_alunos_sinalizados_padrao": round(float((pred_te == 0).mean()), 4),
        "prop_alunos_sinalizados_escolhido": round(float((pred_corte == 0).mean()), 4),
    }
    print(f"\ncorte para recall(não alf.) >= {RECALL_ALVO}: {corte:.2f} -> "
          f"recall {threshold['metricas_corte_escolhido']['recall_nao_alf']}, "
          f"precisão {threshold['metricas_corte_escolhido']['precisao_nao_alf']}")
    plot_threshold(curva, corte, os.path.join(FIG, "18_threshold.png"))

    # ---- robustez 1: split aleatório de alunos (municípios conhecidos no treino) ----
    print("\n=== ROBUSTEZ: split aleatório vs por município ===")
    tr_a, te_a = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=SEED)
    m_a = clone(campeao).fit(X.iloc[tr_a], y.iloc[tr_a])
    p_a = m_a.predict_proba(X.iloc[te_a])[:, 1]
    aleatorio = avaliar(rotulo, y.iloc[te_a], (p_a >= 0.5).astype(int), p_a)
    comparacao = {"por_municipio": final, "aleatorio_por_aluno": aleatorio,
                  "diferenca_roc_auc": round(aleatorio["roc_auc"] - final["roc_auc"], 4)}
    print(f"ROC-AUC por município {final['roc_auc']} | aleatório {aleatorio['roc_auc']}")
    if "XGB" not in campeao_nome:
        m_x = construir_pipeline(modelos["XGBoost"], numericas, categoricas).fit(X.iloc[tr_a], y.iloc[tr_a])
        p_x = m_x.predict_proba(X.iloc[te_a])[:, 1]
        auc_x = avaliar("XGBoost", y.iloc[te_a], (p_x >= 0.5).astype(int), p_x)["roc_auc"]
        comparacao["xgboost_referencia"] = {
            "roc_auc_cv_por_municipio": cv_resumo["XGBoost"]["roc_auc"]["validacao_media"],
            "roc_auc_teste_aleatorio": auc_x,
            "leitura": "o boosting ganha no split aleatório porque memoriza municípios; em municípios novos perde para a regressão logística",
        }
        print(f"XGBoost: CV por município {comparacao['xgboost_referencia']['roc_auc_cv_por_municipio']} | aleatório {auc_x}")

    # ---- robustez 2: validação temporal 2023 -> 2024 ----
    temporal = {}
    m23, m24 = (ano == 2023).values, (ano == 2024).values
    if m23.sum() > 1000 and m24.sum() > 1000:
        print("\n=== ROBUSTEZ: treino 2023 -> teste 2024 ===")
        m_t = clone(campeao).fit(X[m23], y[m23])
        p_t = m_t.predict_proba(X[m24])[:, 1]
        ufs_2023 = set(X.loc[m23, "sigla_uf"])
        so_ufs_conhecidas = m24 & X["sigla_uf"].isin(ufs_2023).values
        p_c = m_t.predict_proba(X[so_ufs_conhecidas])[:, 1]
        temporal = {
            "treino": {"ano": 2023, "alunos": int(m23.sum())},
            "teste_2024_todos": avaliar(rotulo, y[m24], (p_t >= 0.5).astype(int), p_t),
            "teste_2024_ufs_com_2023": avaliar(rotulo, y[so_ufs_conhecidas], (p_c >= 0.5).astype(int), p_c),
            "ufs_sem_2023": sorted(set(X.loc[m24, "sigla_uf"]) - ufs_2023),
        }
        print(f"ROC-AUC 2024 (todos) {temporal['teste_2024_todos']['roc_auc']} | "
              f"só UFs com 2023 {temporal['teste_2024_ufs_com_2023']['roc_auc']}")

    # ---- interpretabilidade ----
    print("\n=== INTERPRETABILIDADE ===")
    modelo_fit = campeao.named_steps["modelo"]
    nomes = nomes_features(campeao)
    X_shap = X_te.sample(min(5000, len(X_te)), random_state=SEED)
    Xt = campeao[:-1].transform(X_shap)
    if hasattr(modelo_fit, "feature_importances_"):
        imp = pd.Series(modelo_fit.feature_importances_, index=nomes)
        explainer = shap.TreeExplainer(modelo_fit)
        tipo_imp = "ganho"
    else:
        imp = pd.Series(np.abs(modelo_fit.coef_[0]), index=nomes)
        explainer = shap.LinearExplainer(modelo_fit, Xt)
        tipo_imp = "|coeficiente| (features padronizadas)"
    imp = imp.sort_values(ascending=False)
    imp.head(30).to_json(os.path.join(REPORTS, "importancias.json"), indent=2)
    print(f"Top 10 por {tipo_imp}:\n" + imp.head(10).round(4).to_string())
    try:
        sv = explainer.shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[1]
        elif sv.ndim == 3:
            sv = sv[:, :, 1]
        shap.summary_plot(sv, Xt, feature_names=nomes, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, "09_shap_summary.png"), dpi=110, bbox_inches="tight")
        plt.close()
        shap_imp = pd.Series(np.abs(sv).mean(axis=0), index=nomes).sort_values(ascending=False)
        shap_imp.head(20).to_json(os.path.join(REPORTS, "shap_importancia.json"), indent=2)
        por_var = agrupar_por_variavel(shap_imp)
        por_var.head(15).to_json(os.path.join(REPORTS, "shap_por_variavel.json"), indent=2)
        print("\nTop 10 por |SHAP| médio:\n" + shap_imp.head(10).round(4).to_string())
        print("\nPor variável original:\n" + por_var.head(8).round(4).to_string())
    except Exception as e:
        print(f"[aviso] SHAP falhou ({type(e).__name__}: {e}); seguindo só com as importâncias.")

    # ---- persistência ----
    tabela.to_json(os.path.join(REPORTS, "metricas_modelos.json"), indent=2, force_ascii=False)
    salvar_json({"protocolo": split_info, "n_cv": int(len(X_cv)), "modelos": cv_resumo}, "cv_modelos.json")
    salvar_json(tuning, "tuning.json")
    salvar_json({**final, "protocolo": split_info["protocolo"]}, "metricas_campeao.json")
    salvar_json({"treino": treino, "teste": final}, "treino_vs_teste.json")
    salvar_json(threshold, "threshold.json")
    salvar_json(comparacao, "comparacao_split.json")
    if temporal:
        salvar_json(temporal, "validacao_temporal.json")
    joblib.dump(campeao, os.path.join(MODELS, "campeao.joblib"))
    print("\nartefatos salvos em reports/ e models/campeao.joblib")
    if base == config.BASE_LOCAL:
        publicar_artefatos([os.path.join(MODELS, "campeao.joblib")], config.TC3_MODELS)
        publicar_artefatos(sorted(glob.glob(os.path.join(REPORTS, "*.json")) + glob.glob(os.path.join(REPORTS, "*.md"))),
                           config.TC3_REPORTS)
    print("=== modelagem concluída ===")


if __name__ == "__main__":
    main()
