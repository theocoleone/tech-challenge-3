# TECH CHALLENGE FASE 3 — Predição e Inteligência Analítica para Alfabetização no Brasil

**Autor:** Theo Coleone de Camargo
**Curso:** Pós-Graduação — AI Scientist — FIAP

**Dashboard:** _(CloudFront — a publicar)_
**Vídeo executivo:** _(a gravar, ≤5 min)_

> Continuação do Tech Challenge Fase 2: consome a camada **Gold** do data lake construído lá
> (Arquitetura Medalhão no S3) e desenvolve um modelo supervisionado de Machine Learning.

---

## 1. Contexto do problema
_A alfabetização na idade certa (2º ano do EF, corte de 743 na escala Saeb) e o Compromisso
Nacional Criança Alfabetizada. Por que antecipar risco importa para a gestão pública._ <!-- TODO -->

## 2. Objetivo analítico
Prever se um **aluno** é *alfabetizado* (1) ou *não alfabetizado* (0) — classificação binária
supervisionada — e traduzir o modelo em inteligência aplicável (fatores de impacto, municípios de
risco, regiões semelhantes, projeção de metas). <!-- TODO detalhar -->

## 3. Descrição da base utilizada
_Microdados por aluno (`silver/alunos`, ~3,87M linhas) + território (`diretorio_municipio`) +
enriquecimento externo (Atlas IDHM, IBGE, Censo Escolar). Alvo = `alfabetizado`. Tratamento de
data leakage (proficiência e agregados do exame excluídos)._ <!-- TODO -->

## 4. Etapas de modelagem
_EDA → pipeline sklearn (imputação, encoding, scaling) integrada ao modelo → split estratificado
→ validação (StratifiedKFold) → otimização (RandomizedSearchCV)._ <!-- TODO -->

## 5. Escolha do algoritmo
_Baselines interpretáveis (Regressão Logística, Árvore) + Random Forest / XGBoost. Justificativa._ <!-- TODO -->

## 6. Métricas de avaliação
_Matriz de confusão, precisão, recall, F1, ROC-AUC/AUC-PR. Foco em recall da classe "não
alfabetizado". Tratamento de desbalanceamento._ <!-- TODO -->

## 7. Interpretação dos resultados
_Feature Importance + SHAP (summary/waterfall/dependence). Quais variáveis mais influenciam._ <!-- TODO -->

## 8. Insights encontrados
_Fatores de impacto, municípios de maior risco, clusters de regiões semelhantes, projeção de metas 2030._ <!-- TODO -->

## 9. Limitações do projeto
_Defasagem temporal do Atlas (2010), rótulo de ausentes, só 2 anos de série, ausência ≠ causalidade._ <!-- TODO -->

## 10. Aplicação prática para políticas públicas
_Como o modelo apoia secretarias de educação e gestão pública na priorização de recursos._ <!-- TODO -->

## 11. Possíveis evoluções futuras
_Mais fontes externas, séries mais longas, modelo por escola, MLOps/retreinamento._ <!-- TODO -->

---

## Arquitetura e reprodução

```
tech-challenge-3/
├── data/            # dados locais (gitignored — vivem no S3)
├── notebooks/       # 01_eda · 02_pipeline_modelagem · 03_avaliacao_shap · 04_aplicacao_estrategica
├── src/
│   ├── preprocessing/  # ingestão S3+BigQuery, montagem da base, ColumnTransformer
│   ├── modeling/       # Pipelines, treino, tuning, persistência
│   ├── evaluation/     # métricas, SHAP, curvas
│   └── visualization/  # figuras + dashboard (Plotly)
├── dashboard/       # gerar_dashboard.py → index.html (publicado no S3 + CloudFront)
├── reports/         # métricas e relatório técnico
├── images/          # fluxograma dos serviços AWS + prints
├── requirements.txt
└── requirements-cloud.txt  # SageMaker (treino gerenciado)
```

**Setup local:**
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# acesso ao data lake via AWS profile fiap-tech-challenge (conta 286958704145, us-east-1)
```

**Nuvem:** treino no **SageMaker**; dados/artefatos/relatórios no **S3**; dashboard servido via **CloudFront**.
Fluxo: `BigQuery → S3 (features) → SageMaker (treino) → S3 (models/reports) → dashboard → S3 → CloudFront`.
