"""Agregados descritivos da base para o dashboard, que assim não precisa do parquet.

Rodar da raiz do repo:  python -m src.evaluation.exportar_agregados
"""
import json
import os

import pandas as pd

from src import config

SAIDA = os.path.join(config.REPORTS_DIR, "agregados_base.json")


def exportar_agregados(caminho_base: str | None = None, saida: str = SAIDA) -> dict:
    caminho_base = caminho_base or config.base_local()
    df = pd.read_parquet(
        caminho_base, columns=[config.COLUNA_ALVO, "id_municipio", "nome_regiao", "sigla_uf", "ano"]
    )
    y = df[config.COLUNA_ALVO].astype(int)

    def taxa_por(col):
        return {str(k): round(float(v), 4) for k, v in y.groupby(df[col]).mean().items()}

    agregados = {
        "fonte": os.path.relpath(caminho_base, config.RAIZ),
        "n_alunos": int(len(df)),
        "n_municipios": int(df["id_municipio"].nunique()),
        "taxa_nacional": round(float(y.mean()), 4),
        "taxa_por_ano": taxa_por("ano"),
        "taxa_por_regiao": taxa_por("nome_regiao"),
        "taxa_por_uf": taxa_por("sigla_uf"),
    }
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(agregados, f, indent=2, ensure_ascii=False)
    print(f"agregados -> {os.path.relpath(saida, config.RAIZ)}")
    return agregados


if __name__ == "__main__":
    exportar_agregados()
