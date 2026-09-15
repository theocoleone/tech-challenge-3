# TECH CHALLENGE FASE 3 — Predição e Inteligência Analítica para Alfabetização no Brasil

**Autor:** Theo Coleone de Camargo
**Curso:** Pós-Graduação — AI Scientist — FIAP

**Dashboard:** https://dv2nlyojecknt.cloudfront.net
**Vídeo executivo:** _(a gravar)_

> Continuação do Tech Challenge Fase 2: consome o data lake construído lá (Arquitetura Medalhão no
> S3) e desenvolve um modelo supervisionado de Machine Learning validado em municípios que ele nunca
> viu, mais uma leitura estratégica por município.

---

## 1. Contexto do problema

A alfabetização na idade certa é um marco do desenvolvimento educacional. O **Compromisso Nacional
Criança Alfabetizada** estabelece que toda criança deve estar alfabetizada até o fim do 2º ano do
Ensino Fundamental, com meta em 2030. Uma criança é considerada alfabetizada a partir de **743 pontos
na escala Saeb** (INEP).

Saber os números atuais não basta: o gestor precisa **antecipar risco**, identificar territórios
vulneráveis e entender **quais fatores** pesam no resultado. É onde a Ciência de Dados transforma dado
público em inteligência para a decisão.

## 2. Objetivo analítico

1. **Modelo supervisionado (grão aluno):** classificação binária — prever se um aluno do 2º ano é
   *alfabetizado* (1) ou *não alfabetizado* (0) a partir de variáveis educacionais, territoriais e
   socioeconômicas do seu município.
2. **Aplicação estratégica (grão município):** responder às perguntas de negócio — quais fatores mais
   impactam, quais municípios têm maior risco, quais regiões se parecem e quem tende a não atingir a
   meta de 2030.

## 3. Descrição da base utilizada

- **Alunos (Silver do TC2, S3):** microdados do SAEB Alfabetização 2023–2024, **3.867.999 alunos** em
  **5.548 municípios**. Alvo = coluna `alfabetizado` (0/1) da fonte; também carrego `presenca`, só
  para definir população e checar sensibilidade (nunca como feature).
- **Censo Escolar (INEP, via BigQuery / Base dos Dados):** infraestrutura, docentes, matrículas e salas
  das **escolas públicas** de anos iniciais, **agregados por município e ano** (o `id_escola` do SAEB é
  anonimizado e não casa com o código INEP).
- **Atlas do Desenvolvimento Humano (PNUD) e IVS (Ipea), 2010:** IDHM e subíndices, renda, Gini,
  pobreza, saneamento, analfabetismo adulto, vulnerabilidade social.
- **IBGE:** população e PIB per capita (2023); diretório com região, Amazônia Legal e capital.
- **Gold do TC2 (S3):** `indicador_municipio` (taxa por município e ano, metas 2030), usada só na
  camada estratégica.

Base analítica final: **49 colunas no grão aluno**, join por `id_municipio` (e `ano`, no Censo), 100%
de cobertura. Dicionário completo em [`docs/dicionario_dados.md`](docs/dicionario_dados.md); a
análise exploratória com tabelas e hipóteses está em [`reports/eda_resumo.md`](reports/eda_resumo.md).

O que a EDA mostrou e mudou a modelagem:

- **Todo aluno ausente está rotulado como não alfabetizado.** Presença de 86,8%; taxa de 51,3% com o
  rótulo da fonte, **59,1% só entre presentes**. A Gold do TC2 usa a taxa entre presentes (diferença
  média de 1,1 p.p.), que é a definição do indicador oficial.
- **AC, DF e SP só existem em 2024** na Silver (484 mil alunos). A comparação anual e a variável `ano`
  ficam confundidas com a composição de UFs.
- **Variações anuais extremas por mecanismos distintos:** RS caiu 14,8 p.p. com presença estável e a
  taxa entre presentes indo de 64,7% para 45,7%; SC caiu 4,7 p.p. por queda de presença (78,6% →
  70,1%) com resultado entre presentes estável.
- **O contexto socioeconômico pesa pouco no indivíduo:** correlações |r| < 0,15 e taxa plana acima do
  3º decil de IDHM. UF separa muito mais (CE 82% vs RN 30%).
- **342 municípios têm menos de 50 alunos** avaliados nos dois anos; taxas instáveis.

## 4. Etapas de modelagem

1. **Pipeline Scikit-learn** (`src/modeling/pipeline.py`): `FunctionTransformer` (engenharia de
   atributos) → `ColumnTransformer` (mediana + padronização nas numéricas; constante + One-Hot nas
   categóricas) → estimador. Tudo ajustado só no treino.
2. **Engenharia de atributos** (`src/modeling/features.py`): alunos por docente, matrículas por sala,
   escolas por 10 mil habitantes, log de população e de PIB per capita.
3. **Tratamento de data leakage:** colunas medidas no exame ou derivadas do alvo ficam em
   `COLUNAS_LEAKAGE` (`src/config.py`) e `separar_features` falha se alguma sobrar; `ano` fica fora.
   E, como quase todas as features são municipais, **treino e teste são separados por município**
   (`GroupShuffleSplit`), senão o modelo é avaliado em territórios que já conhece.
4. **Seleção por validação cruzada:** `StratifiedGroupKFold(5)` com folds por município, quatro
   algoritmos (Regressão Logística, Árvore, Random Forest, XGBoost), média ± desvio.
5. **Tuning dos dois melhores** com `RandomizedSearchCV` (15 iterações, 3 folds por município); o
   baseline de cada um é mantido se vencer.
6. **Teste tocado uma única vez** pelo campeão final, em 1.109 municípios nunca vistos.
7. **Robustez:** o mesmo modelo com split aleatório de alunos (municípios conhecidos) e validação
   temporal (treino 2023 → teste 2024).
8. **Corte de decisão** para a classe não alfabetizado e **interpretabilidade** (SHAP e importâncias).

Para tratabilidade, o treino usa uma **amostra estratificada de 800 mil alunos** (todos os
municípios continuam representados); a validação cruzada roda em 300 mil e o tuning em 200 mil.

## 5. Escolha do algoritmo

Validação cruzada por município (ROC-AUC, média de 5 folds ± desvio):

| Modelo | Validação | Treino |
|---|---|---|
| **Regressão Logística** | **0,630 ± 0,013** | 0,642 |
| Random Forest | 0,625 ± 0,012 | 0,688 |
| XGBoost | 0,623 ± 0,017 | 0,689 |
| Árvore de Decisão | 0,599 ± 0,019 | 0,633 |

**Campeã: Regressão Logística** (L2, `C = 0,0115` após tuning; 0,6278 vs 0,6272 do baseline em CV).
Random Forest tunada ficou em 0,6262. O resultado inverte a intuição inicial: os modelos de árvore
chegavam a **0,67** no split aleatório de alunos porque **memorizavam municípios** (treino 0,69 vs
validação 0,62); em municípios novos, o modelo linear regularizado generaliza igual ou melhor, com
metade da distância entre treino e validação.

## 6. Métricas de avaliação

Classe 0 = *não alfabetizado* (a de interesse para intervenção). Campeã no **teste por município**
(138.576 alunos de 1.109 municípios fora do treino):

| Métrica | Teste | Treino |
|---|---|---|
| ROC-AUC | **0,633** | 0,642 |
| Acurácia | 0,595 | 0,601 |
| F1-macro | 0,594 | 0,601 |
| Recall (não alfabetizado) | 0,568 | 0,585 |
| Precisão (não alfabetizado) | 0,584 | 0,592 |
| AUC-PR | 0,645 | 0,653 |

Robustez do mesmo modelo: **0,643** com split aleatório por aluno (a diferença de 0,009 mede quanto
o modelo dependia de reconhecer o município; no XGBoost essa diferença é 0,05); **0,619** treinando
em 2023 e testando em 2024 (0,645 sem AC, DF e SP, que não existem em 2023).

**Corte de decisão:** com 0,50 o modelo recupera 57% dos não alfabetizados; com **0,55** recupera
**71%** ao custo de precisão (56% vs 58%) e de sinalizar 62% dos alunos em vez de 47%. O corte é uma
escolha de gestão (busca ativa vs custo de triagem), não do modelo.

Alvo balanceado (51/49). Figuras: `reports/figuras/07_confusao_campeao.png`, `08_roc_campeao.png`,
`18_threshold.png`; artefatos em `reports/cv_modelos.json`, `tuning.json`, `metricas_campeao.json`,
`treino_vs_teste.json`, `comparacao_split.json`, `validacao_temporal.json`, `threshold.json`.

## 7. Interpretação dos resultados

Por **SHAP** (somado por variável original, `reports/shap_por_variavel.json`): **UF** concentra 33%
da influência das 15 variáveis mais importantes, 3,3 vezes a segunda colocada. Depois vêm **porte do
município** (log da população e nº de escolas), **IDHM renda**, **% de pobres**, **analfabetismo
adulto**, **IVS capital humano** e **região**. Infraestrutura escolar aparece atrás (% de escolas com
quadra, matrículas por escola).

Nos coeficientes, as maiores magnitudes são de UFs: **Ceará** (positivo, o maior de todos), **RN, BA e
SE** (negativos), GO e PR (positivos). Sinais individuais de variáveis correlacionadas (população e
nº de escolas, IDHM e pobreza) devem ser lidos em conjunto: em um modelo linear regularizado eles se
compensam. A leitura robusta é a de magnitude — geografia primeiro, porte e renda depois, escola por
último.

## 8. Insights encontrados

- **Desigualdade regional:** Norte 42% → Centro-Oeste 55%; UFs de 30% (RN) a 82% (CE).
- **Ceará é um caso de política pública, não de contexto:** IDHM médio-baixo, 82% de alfabetizados,
  o maior coeficiente positivo do modelo. Evidência do peso da ação consistente (PAIC) sobre o
  contexto socioeconômico.
- **Memória de território não é sinal:** a vantagem do boosting desaparecia ao trocar o split
  aleatório pelo split por município. Sem essa validação, o projeto teria escolhido o modelo errado.
- **Dois perfis de município** (clustering, silhouette 0,31 — estrutura fraca, um corte do gradiente
  de IDHM): *vulnerável* (2.368 municípios; taxa 51%, IDHM 0,59, 41% de pobres, 70% de escolas rurais;
  BA, MG, PI, MA) e *desenvolvido* (3.180; taxa 60%, IDHM 0,71, 10% de pobres). Com k = 3 surge um
  grupo intermediário (1.840 municípios; MG, RS, PR, GO).
- **Risco municipal:** os 20 municípios de menor indicador em 2024 (entre 4.393 com ≥ 50 alunos e
  participação ≥ 50%) têm de 11% a 19% de alfabetizados entre presentes; 13 estão no Nordeste e 5
  no Tocantins (Pedrinhas-SE, Dirceu Arcoverde-PI, Filadélfia-TO, Arataca-BA...). O risco previsto
  pelo modelo tem correlação de −0,74 com o indicador observado.
- **Metas 2030 em risco:** entre 4.271 municípios com dado confiável, **59,6%** não atingem a meta no
  cenário de referência (tendência própria encolhida para a da UF, variação anual limitada a ±5 p.p.);
  81,0% se mantiverem a taxa de 2024 e 66,1% seguindo a tendência da UF.
- **1.155 municípios ficaram fora do ranking** por falta de dado confiável em 2024 (1.114 com menos
  de 50 alunos, 29 sem avaliação, 12 com participação abaixo de 50%).

## 9. Limitações do projeto

- **Teto de sinal no grão aluno:** sem atributos do aluno ou da escola, prever o indivíduo a partir do
  contexto municipal rende ROC-AUC ~0,63. O valor está na interpretação e na leitura municipal.
- **Rótulo:** ausentes contam como não alfabetizados (13% da base). O modelo aprende o indicador que o
  gestor recebe; a sensibilidade está documentada e a camada municipal filtra participação.
- **Cobertura da Silver:** AC, DF e SP sem 2023 (a Gold do TC2 tem o indicador municipal de 2023 para
  SP, então o gap está nos microdados). Sem painel aluno a aluno: `id_aluno` é anonimizado por edição.
- **`id_escola` anonimizado:** Censo Escolar entra agregado por município; a principal perda de
  features do projeto.
- **Defasagem:** Atlas ADH e IVS são de 2010.
- **Série curta:** dois anos. A projeção de metas é um conjunto de cenários limitados, não uma
  previsão; a diferença de um único ano não é tendência (RS e SC mostram isso).
- **Clusters fracos:** silhouette 0,31; a tipologia é indicativa.

## 10. Aplicação prática para políticas públicas

- **Priorizar** os municípios do perfil vulnerável e o topo do ranking de risco, em vez de distribuir
  recursos de forma uniforme.
- **Replicar o que funciona:** estudar e adaptar o modelo cearense nos estados da base do ranking.
- **Monitorar as metas** com cenários: os 2.547 municípios fora da rota de 2030 no cenário de
  referência são a lista de acompanhamento (`reports/metas_2030_municipios.csv`).
- **Garantir a avaliação** onde falta dado confiável: sem medição não há gestão.
- **Usar o corte de decisão como política:** recall de 70% para busca ativa, corte padrão para triagem.

## 11. Possíveis evoluções futuras

- Crosswalk `id_escola` → INEP para features reais por escola.
- Mais edições da avaliação para modelagem temporal adequada e validação temporal completa.
- Target encoding de município fold-safe, avaliado com o mesmo protocolo por grupo.
- Novas fontes: FUNDEB, PNAD Contínua, Cadastro Único, Censo 2022 (substituir o ADH 2010).
- Treino gerenciado (SageMaker) com o mesmo `Pipeline`, monitoramento de drift e retreinamento anual.

## 12. Decisões analíticas

| Decisão | Alternativa descartada | Por quê |
|---|---|---|
| Treinar no grão aluno com a Silver | Treinar na Gold (grão município) | O guideline pede prever o aluno; a Gold tem ~5,5 mil linhas e responde outra pergunta |
| Rótulo `alfabetizado` da fonte | Recalcular de `proficiencia ≥ 743` | Exigiria carregar a variável que vaza o alvo; o rótulo é o indicador que o gestor recebe |
| `presenca` só como filtro | Usar como feature / ignorar | Filtro de população não vaza; feature vaza |
| Censo Escolar por município, só rede pública | Join por escola / todas as redes | `id_escola` anonimizado; alunos avaliados são da rede pública |
| Split e CV por município | Split aleatório por aluno | Features municipais: alunos do mesmo município não são independentes |
| `ano` fora das features | Manter | Confundido com UF (AC/DF/SP só em 2024) e inútil para prever um ano novo |
| Tuning dos dois melhores, baseline mantido se vencer | Tunar só o primeiro | Diferença pequena entre finalistas; evita escolher por ruído |
| Regressão Logística como campeã | XGBoost (melhor no split aleatório) | Melhor em municípios novos, metade do gap treino/validação |
| Amostra de 800 mil alunos | Base completa | Sinal individual satura; tuning viável localmente |
| Ranking pela taxa entre presentes, com filtros | Taxa com ausentes, sem filtro | Mesma definição da Gold; município com 0% e 410 alunos era não participação |
| Cenários de meta com limite de ±5 p.p./ano | Extrapolação linear de um ano | A reta produzia gaps de ±600 p.p.; um ano não é tendência |

---

## Arquitetura e reprodução

![Arquitetura — BigQuery e S3 para execução local, artefatos no S3, dashboard no CloudFront](images/arquitetura_tc3.png)

```
tech-challenge-3/
├── data/            # base completa (gitignored, vive no S3) + amostra.parquet versionada
├── docs/            # dicionario_dados.md
├── notebooks/       # 01_eda · 02_modelagem · 03_aplicacao_estrategica
├── src/
│   ├── config.py       # caminhos, bucket, alvo, COLUNAS_LEAKAGE
│   ├── data_access.py  # S3 (Parquet particionado), BigQuery, publicação de artefatos
│   ├── preprocessing/  # ingestão S3 + BigQuery, montagem da base, amostra
│   ├── modeling/       # features.py (atributos derivados) + pipeline.py (ColumnTransformer)
│   ├── evaluation/     # métricas, scorers, gráficos, agregados para o dashboard
│   └── visualization/
├── models/          # campeao.joblib (pipeline completo, 13 KB)
├── dashboard/       # gerar_dashboard.py → index.html; deploy.sh + configs S3/CloudFront
├── reports/         # métricas (JSON/CSV), eda_resumo.md, figuras
├── images/          # fluxograma (drawio + png)
└── requirements.txt
```

**Setup e execução (da raiz do repo):**
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Acesso ao data lake (opcional — só para montar a base completa):
#   AWS: profile fiap-tech-challenge (conta 286958704145, us-east-1)
#   GCP: gcloud auth application-default login
#        export GCP_BILLING_PROJECT=<projeto que fatura as queries no BigQuery>
python -m src.preprocessing.montar_base       # S3 + BigQuery -> data/base_analitica.parquet (+ S3 tc3/features)

python notebooks/01_eda.py                    # EDA: figuras, reports/eda_resumo.md, agregados do dashboard
python notebooks/02_modelagem.py              # CV por município, tuning, teste, robustez, SHAP (+ S3 tc3/models, tc3/reports)
python notebooks/03_aplicacao_estrategica.py  # clusters, ranking com filtro de participação, cenários 2030 (Gold no S3)
python dashboard/gerar_dashboard.py           # index.html a partir de reports/ e images/
bash dashboard/deploy.sh                      # publica no S3 + CloudFront
```

Sem credenciais, os scripts usam `data/amostra.parquet` (149 municípios inteiros sorteados por UF,
61.726 alunos): EDA, modelagem e dashboard rodam ponta a ponta, com números diferentes dos da base
completa. A projeção de metas em `03` lê a Gold no S3 e é pulada sem acesso. `TC3_BASE=<caminho>`
força um parquet específico.

**Nuvem:** execução local em Python; dados de entrada, base analítica (`tc3/features`), modelo
(`tc3/models`) e relatórios (`tc3/reports`) no **S3**; dashboard em bucket privado servido pelo
**CloudFront** com OAC (`dashboard/cloudfront-config.json`, `bucket-policy.json`, `deploy.sh`).
