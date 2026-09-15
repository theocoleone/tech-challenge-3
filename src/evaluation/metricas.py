"""Métricas, scorers e gráficos de avaliação da classificação binária.

Convenção: classe 0 = "não alfabetizado" é a classe de interesse do negócio (o aluno que
precisa de intervenção), então precisão e recall são reportados para ela.
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, ConfusionMatrixDisplay,
    f1_score, make_scorer, precision_score, recall_score,
    RocCurveDisplay, roc_auc_score,
)

METRICAS = ["acuracia", "precisao_nao_alf", "recall_nao_alf", "f1_macro", "roc_auc", "auc_pr"]

SCORING = {
    "acuracia": "accuracy",
    "precisao_nao_alf": make_scorer(precision_score, pos_label=0),
    "recall_nao_alf": make_scorer(recall_score, pos_label=0),
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc",
    "auc_pr": "average_precision",
}


def avaliar(nome, y_true, y_pred, y_proba) -> dict:
    """y_proba = probabilidade da classe 1 (alfabetizado)."""
    return {
        "modelo": nome,
        "acuracia": round(accuracy_score(y_true, y_pred), 4),
        "precisao_nao_alf": round(precision_score(y_true, y_pred, pos_label=0), 4),
        "recall_nao_alf": round(recall_score(y_true, y_pred, pos_label=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "auc_pr": round(average_precision_score(y_true, y_proba), 4),
    }


def resumir_cv(cv_results: dict) -> dict:
    """Média e desvio por métrica, em treino e validação, a partir de cross_validate."""
    resumo = {}
    for m in METRICAS:
        val, tr = cv_results[f"test_{m}"], cv_results[f"train_{m}"]
        resumo[m] = {
            "validacao_media": round(float(np.mean(val)), 4),
            "validacao_desvio": round(float(np.std(val)), 4),
            "treino_media": round(float(np.mean(tr)), 4),
        }
    resumo["tempo_fit_s"] = round(float(np.mean(cv_results["fit_time"])), 1)
    return resumo


def curva_threshold(y_true, y_proba, passos=99) -> pd.DataFrame:
    """Precisão, recall e F1 da classe 0 para cortes na probabilidade da classe 1."""
    y_true = np.asarray(y_true)
    linhas = []
    for t in np.linspace(0.01, 0.99, passos):
        pred = (y_proba >= t).astype(int)
        linhas.append({
            "threshold": round(float(t), 2),
            "precisao_nao_alf": precision_score(y_true, pred, pos_label=0, zero_division=0),
            "recall_nao_alf": recall_score(y_true, pred, pos_label=0),
            "f1_nao_alf": f1_score(y_true, pred, pos_label=0, zero_division=0),
            "prop_sinalizados": float((pred == 0).mean()),
        })
    return pd.DataFrame(linhas)


def agrupar_por_variavel(importancias: pd.Series) -> pd.Series:
    """Soma importâncias das colunas one-hot de volta à variável original."""
    def origem(nome):
        nome = re.sub(r"^(num|cat)__", "", nome)
        for cat in ("rede", "sigla_uf", "nome_regiao", "amazonia_legal", "capital_uf"):
            if nome.startswith(cat + "_"):
                return cat
        return nome
    return importancias.groupby(importancias.index.map(origem)).sum().sort_values(ascending=False)


def plot_confusao(y_true, y_pred, nome, caminho):
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["não alf.", "alfabetizado"], cmap="Blues"
    )
    plt.title(f"Matriz de confusão — {nome}")
    plt.tight_layout()
    plt.savefig(caminho, dpi=110, bbox_inches="tight")
    plt.close()


def plot_roc(y_true, y_proba, nome, caminho):
    RocCurveDisplay.from_predictions(y_true, y_proba, name=nome)
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.title(f"Curva ROC — {nome}")
    plt.tight_layout()
    plt.savefig(caminho, dpi=110, bbox_inches="tight")
    plt.close()


def plot_threshold(curva: pd.DataFrame, escolhido: float, caminho):
    plt.figure(figsize=(7, 4.5))
    plt.plot(curva["threshold"], curva["precisao_nao_alf"], label="precisão (não alf.)")
    plt.plot(curva["threshold"], curva["recall_nao_alf"], label="recall (não alf.)")
    plt.plot(curva["threshold"], curva["prop_sinalizados"], "--", color="gray", label="% alunos sinalizados")
    plt.axvline(0.5, color="#c9ccd1", lw=1)
    plt.axvline(escolhido, color="#a6462e", ls="--", lw=1.2, label=f"corte escolhido = {escolhido:.2f}")
    plt.xlabel("corte na probabilidade de alfabetizado"); plt.ylim(0, 1); plt.legend()
    plt.title("Trade-off do corte de decisão para a classe não alfabetizado")
    plt.tight_layout()
    plt.savefig(caminho, dpi=110, bbox_inches="tight")
    plt.close()
