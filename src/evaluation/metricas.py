"""Métricas e gráficos de avaliação para a classificação binária do TC3.

Convenção: classe 0 = "não alfabetizado" (a de interesse do negócio — é o aluno que
precisa de intervenção), então reportamos precisão/recall focados nessa classe.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, average_precision_score, ConfusionMatrixDisplay,
    confusion_matrix, f1_score, precision_score, recall_score,
    RocCurveDisplay, roc_auc_score,
)


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
