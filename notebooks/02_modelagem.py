"""Modelagem supervisionada — prever `alfabetizado` (0/1) no grão aluno.

Fluxo: amostra estratificada (tratabilidade) -> split estratificado treino/teste ->
Pipeline (pré-processamento + modelo) para 4 algoritmos -> escolha do campeão por ROC-AUC
-> tuning (RandomizedSearchCV) -> avaliação no teste -> interpretabilidade (SHAP).

Rodar da raiz do repo:  python notebooks/02_modelagem.py
"""
import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.evaluation.metricas import avaliar, plot_confusao, plot_roc
from src.modeling.pipeline import construir_preprocessador, separar_features

warnings.filterwarnings("ignore")

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(AQUI, "..", "data", "base_analitica.parquet")
REPORTS = os.path.join(AQUI, "..", "reports")
FIG = os.path.join(REPORTS, "figuras")
MODELS = os.path.join(AQUI, "..", "models")
os.makedirs(FIG, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

SEED = 42
N_AMOSTRA = 800_000   # amostra estratificada de 3,87M — decisão de tratabilidade (registrada)


def main():
    df = pd.read_parquet(BASE)
    if len(df) > N_AMOSTRA:
        df, _ = train_test_split(df, train_size=N_AMOSTRA, stratify=df["alfabetizado"],
                                 random_state=SEED)
        print(f"[amostra estratificada: {N_AMOSTRA:,} de 3.867.999 linhas]")

    X, y, numericas, categoricas = separar_features(df)
    print(f"features: {len(numericas)} numéricas + {len(categoricas)} categóricas")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pre = construir_preprocessador(numericas, categoricas)

    modelos = {
        "Regressão Logística": LogisticRegression(max_iter=1000),
        "Árvore de Decisão": DecisionTreeClassifier(max_depth=8, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_depth=14, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(
            tree_method="hist", n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=SEED, eval_metric="logloss"),
    }

    resultados = []
    for nome, modelo in modelos.items():
        print(f"\n>>> treinando {nome}...")
        pipe = Pipeline([("pre", pre), ("modelo", modelo)])
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = pipe.predict(X_te)
        resultados.append(avaliar(nome, y_te, pred, proba))

    tabela = pd.DataFrame(resultados).set_index("modelo").sort_values("roc_auc", ascending=False)
    print("\n=== COMPARAÇÃO (ordenado por ROC-AUC) ===")
    print(tabela.to_string())

    # ---- Campeão + tuning ----
    campeao_nome = tabela.index[0]
    print(f"\n=== CAMPEÃO: {campeao_nome} — tuning (RandomizedSearchCV) ===")
    pipe_c = Pipeline([("pre", pre), ("modelo", modelos[campeao_nome])])

    if "XGB" in campeao_nome or "Forest" in campeao_nome:
        if "XGB" in campeao_nome:
            dist = {
                "modelo__n_estimators": randint(200, 600),
                "modelo__max_depth": randint(3, 9),
                "modelo__learning_rate": uniform(0.02, 0.20),
                "modelo__subsample": uniform(0.6, 0.4),
                "modelo__colsample_bytree": uniform(0.6, 0.4),
                "modelo__min_child_weight": randint(1, 10),
            }
        else:
            dist = {
                "modelo__n_estimators": randint(100, 400),
                "modelo__max_depth": randint(6, 20),
                "modelo__min_samples_leaf": randint(1, 50),
                "modelo__max_features": ["sqrt", "log2", 0.5],
            }
        # tuning num subconjunto p/ velocidade
        X_sub, _, y_sub, _ = train_test_split(X_tr, y_tr, train_size=200_000,
                                              stratify=y_tr, random_state=SEED)
        busca = RandomizedSearchCV(
            pipe_c, dist, n_iter=15, scoring="roc_auc",
            cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
            n_jobs=-1, random_state=SEED, verbose=0)
        busca.fit(X_sub, y_sub)
        print("melhores hiperparâmetros:", busca.best_params_)
        print("ROC-AUC (CV):", round(busca.best_score_, 4))
        campeao = busca.best_estimator_
        campeao.fit(X_tr, y_tr)   # refit na base de treino cheia
    else:
        campeao = pipe_c.fit(X_tr, y_tr)

    # ---- Avaliação final do campeão ----
    proba = campeao.predict_proba(X_te)[:, 1]
    pred = campeao.predict(X_te)
    final = avaliar(f"{campeao_nome} (tunado)", y_te, pred, proba)
    print("\n=== CAMPEÃO NO TESTE ===")
    for k, v in final.items():
        print(f"  {k}: {v}")
    plot_confusao(y_te, pred, campeao_nome, os.path.join(FIG, "07_confusao_campeao.png"))
    plot_roc(y_te, proba, campeao_nome, os.path.join(FIG, "08_roc_campeao.png"))

    # ---- Interpretabilidade (SHAP + importâncias) ----
    print("\n=== INTERPRETABILIDADE (SHAP) ===")
    pre_fit = campeao.named_steps["pre"]
    modelo_fit = campeao.named_steps["modelo"]
    nomes = list(pre_fit.get_feature_names_out())
    imp = None
    if hasattr(modelo_fit, "feature_importances_"):
        try:
            X_shap = X_te.sample(min(5000, len(X_te)), random_state=SEED)
            Xt = pre_fit.transform(X_shap)
            expl = shap.TreeExplainer(modelo_fit)
            sv = expl.shap_values(Xt)
            if isinstance(sv, list):        # RF binário devolve lista [classe0, classe1]
                sv = sv[1]
            shap.summary_plot(sv, Xt, feature_names=nomes, show=False, max_display=15)
            plt.tight_layout()
            plt.savefig(os.path.join(FIG, "09_shap_summary.png"), dpi=110, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"[aviso] SHAP falhou ({type(e).__name__}: {e}); seguindo com importâncias.")
        imp = pd.Series(modelo_fit.feature_importances_, index=nomes).sort_values(ascending=False)
        print("Top 15 variáveis (importância do modelo):")
        print(imp.head(15).round(4).to_string())
    else:
        print("Campeão não é baseado em árvore; SHAP/feature_importances_ não aplicável.")

    # ---- Persistência ----
    tabela.to_json(os.path.join(REPORTS, "metricas_modelos.json"), indent=2, force_ascii=False)
    with open(os.path.join(REPORTS, "metricas_campeao.json"), "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    joblib.dump(campeao, os.path.join(MODELS, "campeao.joblib"))
    if imp is not None:
        imp.head(30).to_json(os.path.join(REPORTS, "importancias.json"), indent=2)
    print(f"\nartefatos salvos em reports/ e models/campeao.joblib")
    print("=== modelagem concluída ===")


if __name__ == "__main__":
    main()
