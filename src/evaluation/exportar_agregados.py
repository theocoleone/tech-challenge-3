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
        caminho_base,
        columns=[config.COLUNA_ALVO, "id_municipio", "nome_regiao", "sigla_uf", "ano", "presenca"],
    )
    y = df[config.COLUNA_ALVO].astype(int)
    presentes = df["presenca"] == 1
    contagem = df.groupby(["sigla_uf", "ano"]).size()
    chave = [df["sigla_uf"], df["ano"]]
    por_uf_ano = pd.DataFrame({
        "taxa": y.groupby(chave).mean(),
        "presenca": presentes.groupby(chave).mean(),
        "taxa_presentes": y[presentes].groupby([df.loc[presentes, "sigla_uf"], df.loc[presentes, "ano"]]).mean(),
    })

    def taxa_por(col, mask=None):
        s = y if mask is None else y[mask]
        g = df[col] if mask is None else df.loc[mask, col]
        return {str(k): round(float(v), 4) for k, v in s.groupby(g).mean().items()}

    agregados = {
        "fonte": os.path.relpath(caminho_base, config.RAIZ),
        "n_alunos": int(len(df)),
        "n_municipios": int(df["id_municipio"].nunique()),
        "anos": [int(a) for a in sorted(df["ano"].unique())],
        "taxa_nacional": round(float(y.mean()), 4),
        "presenca": round(float(presentes.mean()), 4),
        "taxa_nacional_presentes": round(float(y[presentes].mean()), 4),
        "taxa_por_ano": taxa_por("ano"),
        "taxa_por_regiao": taxa_por("nome_regiao"),
        "taxa_por_uf": taxa_por("sigla_uf"),
        "taxa_presentes_por_uf": taxa_por("sigla_uf", presentes),
        "presenca_por_uf": {str(k): round(float(v), 4) for k, v in presentes.groupby(df["sigla_uf"]).mean().items()},
        "por_uf_ano": {
            str(uf): {str(int(a)): {k: round(float(v), 4) for k, v in linha.items()}
                      for a, linha in bloco.droplevel(0).iterrows()}
            for uf, bloco in por_uf_ano.groupby(level=0)
        },
        "alunos_por_uf_ano": {
            str(uf): {str(int(a)): int(n) for a, n in contagem.loc[uf].items()}
            for uf in contagem.index.get_level_values(0).unique()
        },
    }
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(agregados, f, indent=2, ensure_ascii=False)
    print(f"agregados -> {os.path.relpath(saida, config.RAIZ)}")
    return agregados


if __name__ == "__main__":
    exportar_agregados()
