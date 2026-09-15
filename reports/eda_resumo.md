# Resumo da análise exploratória

Base: `data/base_analitica.parquet` — 3,867,999 alunos, 49 colunas, 5,548 municípios, anos 2023, 2024.

## 1. Alvo

Taxa de alfabetização geral: **51.3%** (alvo balanceado: 48.7% de não alfabetizados).

| ano | taxa_alfabetizacao | n_alunos |
|---|---|---|
| 2023 | 0.5021 | 1,747,439 |
| 2024 | 0.5221 | 2,120,560 |

## 2. Cobertura por UF e ano

UFs presentes só em 2024: **AC, DF, SP** (484,130 alunos). A comparação 2023→2024 e a variável `ano` ficam confundidas com a composição de UFs.

| sigla_uf | 2023 | 2024 |
|---|---|---|
| AC | 0 | 12,655 |
| AL | 35,503 | 33,668 |
| AM | 63,055 | 58,608 |
| AP | 12,349 | 12,580 |
| BA | 137,724 | 115,576 |
| CE | 100,349 | 89,841 |
| DF | 0 | 27,698 |
| ES | 50,482 | 47,538 |
| GO | 83,741 | 77,166 |
| MA | 74,958 | 69,284 |
| MG | 210,107 | 203,450 |
| MS | 42,199 | 41,327 |
| MT | 54,526 | 50,965 |
| PA | 104,972 | 99,977 |
| PB | 38,215 | 35,118 |
| PE | 88,271 | 80,359 |
| PI | 35,415 | 32,792 |
| PR | 138,532 | 138,914 |
| RJ | 159,997 | 151,201 |
| RN | 31,748 | 29,399 |
| RO | 22,834 | 22,673 |
| RS | 121,148 | 108,262 |
| SC | 102,621 | 98,550 |
| SE | 18,634 | 19,424 |
| SP | 0 | 443,777 |
| TO | 20,059 | 19,758 |

## 3. Presença na prova e sensibilidade do rótulo

- 86.8% dos alunos estiveram presentes; **todo aluno ausente está rotulado como não alfabetizado** (taxa entre ausentes = 0%).

- Taxa com o rótulo da fonte: **51.3%**; só entre presentes: **59.1%**. A diferença (+7.8%) é o efeito do rótulo, não de aprendizagem.

- Município-anos com participação abaixo de 50%: 25 de 10,392; abaixo de 80%: 901.

| sigla_uf | taxa | presenca | taxa_presentes |
|---|---|---|---|
| RN | 0.304 | 0.781 | 0.389 |
| BA | 0.319 | 0.870 | 0.367 |
| SE | 0.334 | 0.963 | 0.346 |
| TO | 0.397 | 0.845 | 0.469 |
| AP | 0.398 | 0.894 | 0.445 |
| PA | 0.401 | 0.818 | 0.490 |
| AM | 0.401 | 0.778 | 0.516 |
| AC | 0.420 | 0.809 | 0.519 |
| AL | 0.434 | 0.931 | 0.467 |
| RJ | 0.449 | 0.821 | 0.546 |
| RS | 0.457 | 0.821 | 0.556 |
| MS | 0.464 | 0.893 | 0.520 |
| SC | 0.470 | 0.744 | 0.631 |
| DF | 0.474 | 0.798 | 0.594 |
| PB | 0.476 | 0.883 | 0.540 |
| MT | 0.512 | 0.879 | 0.582 |
| SP | 0.522 | 0.891 | 0.586 |
| MA | 0.526 | 0.900 | 0.585 |
| PI | 0.533 | 0.942 | 0.565 |
| PE | 0.560 | 0.933 | 0.600 |
| RO | 0.566 | 0.882 | 0.641 |
| MG | 0.594 | 0.892 | 0.666 |
| PR | 0.623 | 0.862 | 0.722 |
| GO | 0.625 | 0.894 | 0.698 |
| ES | 0.626 | 0.891 | 0.702 |
| CE | 0.821 | 0.966 | 0.850 |

## 4. Variação 2023 → 2024 por UF

Mediana da variação: +3.9 p.p.; extremos RS (-14.8 p.p.) e MG (+10.0 p.p.). Separar o efeito da presença do efeito da proficiência muda a leitura: em RS a presença foi de 81% para 83% e a taxa **entre presentes** de 64.7% para 45.7%. Quando a taxa entre presentes cai junto, a queda é de resultado medido; quando só a presença cai, é participação. Em qualquer caso, a diferença de um único ano não é uma tendência.

| sigla_uf | taxa_2023 | taxa_2024 | variacao_pp | presenca_2023 | presenca_2024 | taxa_presentes_2023 | taxa_presentes_2024 |
|---|---|---|---|---|---|---|---|
| RS | 0.527 | 0.379 | -14.823 | 0.814 | 0.829 | 0.647 | 0.457 |
| SC | 0.493 | 0.446 | -4.670 | 0.786 | 0.701 | 0.627 | 0.637 |
| PR | 0.635 | 0.611 | -2.408 | 0.862 | 0.862 | 0.736 | 0.708 |
| RO | 0.575 | 0.557 | -1.819 | 0.881 | 0.883 | 0.653 | 0.630 |
| AM | 0.406 | 0.396 | -0.921 | 0.761 | 0.796 | 0.533 | 0.498 |
| BA | 0.316 | 0.324 | 0.806 | 0.845 | 0.900 | 0.374 | 0.360 |
| PA | 0.396 | 0.406 | 0.935 | 0.806 | 0.832 | 0.492 | 0.488 |
| RN | 0.298 | 0.311 | 1.221 | 0.785 | 0.777 | 0.380 | 0.400 |
| CE | 0.807 | 0.837 | 3.069 | 0.952 | 0.981 | 0.847 | 0.853 |
| RJ | 0.433 | 0.465 | 3.107 | 0.812 | 0.831 | 0.534 | 0.559 |
| PE | 0.544 | 0.577 | 3.331 | 0.919 | 0.948 | 0.592 | 0.609 |
| MA | 0.507 | 0.547 | 3.939 | 0.891 | 0.909 | 0.570 | 0.602 |
| SE | 0.313 | 0.353 | 4.045 | 1.000 | 0.928 | 0.313 | 0.381 |
| ES | 0.604 | 0.648 | 4.389 | 0.884 | 0.899 | 0.684 | 0.721 |
| PB | 0.454 | 0.500 | 4.534 | 0.882 | 0.883 | 0.515 | 0.566 |
| AL | 0.411 | 0.459 | 4.823 | 0.924 | 0.938 | 0.445 | 0.489 |
| AP | 0.373 | 0.421 | 4.823 | 0.897 | 0.891 | 0.416 | 0.473 |
| MT | 0.486 | 0.539 | 5.336 | 0.875 | 0.885 | 0.556 | 0.610 |
| TO | 0.367 | 0.427 | 6.020 | 0.839 | 0.851 | 0.437 | 0.501 |
| PI | 0.496 | 0.572 | 7.521 | 0.934 | 0.951 | 0.532 | 0.601 |
| GO | 0.583 | 0.670 | 8.696 | 0.870 | 0.921 | 0.670 | 0.728 |
| MS | 0.420 | 0.509 | 8.871 | 0.885 | 0.900 | 0.474 | 0.565 |
| MG | 0.545 | 0.645 | 9.953 | 0.896 | 0.889 | 0.609 | 0.726 |

## 5. Rede de ensino e porte dos municípios

| rede | n | taxa |
|---|---|---|
| 2 | 435,398 | 0.532 |
| 3 | 3,432,576 | 0.511 |
| 4 | 25 | 0.640 |

Alunos por município: mediana 234, máximo 107,729; **342 municípios têm menos de 50 alunos** avaliados nos dois anos — taxas instáveis, por isso o ranking de risco exige mínimo de alunos.

Duplicidade de `id_aluno` dentro do mesmo ano: 0. Entre anos há 1,515,671 colisões de id (39% das linhas): o identificador é anonimizado por edição, então não existe painel aluno-a-aluno entre 2023 e 2024.

## 6. Valores faltantes

|  | proporcao_nula |
|---|---|
| renda_pc | 0.0004 |
| taxa_agua_encanada | 0.0004 |
| taxa_analfabetismo_15_mais | 0.0004 |
| indice_gini | 0.0004 |
| taxa_energia_eletrica | 0.0004 |
| idhm_r | 0.0004 |
| idhm_l | 0.0004 |
| idhm_e | 0.0004 |
| idhm | 0.0004 |
| prop_pobreza | 0.0004 |
| taxa_coleta_lixo | 0.0004 |
| expectativa_vida | 0.0004 |
| ivs | 0.0004 |
| ivs_infraestrutura_urbana | 0.0004 |
| ivs_capital_humano | 0.0004 |
| ivs_renda_trabalho | 0.0004 |
| taxa_freq_liquida_fundamental | 0.0004 |
| esc_prop_banda_larga | 0.0000 |

Imputação por mediana no pipeline cobre os casos residuais; nenhuma coluna exige tratamento especial.

## 7. Correlação (Pearson) das numéricas com o alvo

Correlações individuais são fracas (|r| < 0,15): o contexto municipal explica pouco da variação entre alunos, o que antecipa o teto de desempenho do modelo.

|  | r |
|---|---|
| ivs_infraestrutura_urbana | -0.065 |
| indice_gini | -0.052 |
| ivs | -0.046 |
| ivs_capital_humano | -0.034 |
| esc_n_escolas | -0.028 |
| prop_pobreza | -0.017 |
| ivs_renda_trabalho | -0.015 |
| esc_prop_rural | -0.015 |
| esc_prop_laboratorio_informatica | 0.032 |
| esc_prop_biblioteca | 0.036 |
| esc_prop_energia_rede_publica | 0.045 |
| esc_prop_patio_coberto | 0.046 |
| esc_prop_area_verde | 0.049 |
| taxa_energia_eletrica | 0.049 |
| esc_prop_internet | 0.056 |
| esc_prop_quadra_esportes | 0.062 |

## 8. Desigualdade territorial

| nome_regiao | taxa |
|---|---|
| Norte | 0.418 |
| Nordeste | 0.506 |
| Sul | 0.525 |
| Sudeste | 0.536 |
| Centro-Oeste | 0.547 |

UF: de **RN (30.4%)** a **CE (82.1%)**. O CE descola do resto do país (+19.5% sobre o segundo, ES).

Por decil de IDHM a taxa vai de 45.7% (1º decil) a 50.3% (10º), com máximo de 54.1%: o gradiente socioeconômico existe só na base da distribuição e a curva fica plana a partir do 3º decil. Sem o CE (IDHM médio-baixo e a maior taxa do país) o degrau inicial fica mais nítido.

| idhm_decil | idhm_medio | taxa | taxa_presentes | taxa_sem_CE |
|---|---|---|---|---|
| 0.0 | 0.556 | 0.457 | 0.511 | 0.439 |
| 1.0 | 0.611 | 0.541 | 0.595 | 0.471 |
| 2.0 | 0.656 | 0.531 | 0.599 | 0.497 |
| 3.0 | 0.690 | 0.527 | 0.596 | 0.513 |
| 4.0 | 0.714 | 0.502 | 0.583 | 0.493 |
| 5.0 | 0.732 | 0.514 | 0.610 | 0.514 |
| 6.0 | 0.747 | 0.537 | 0.622 | 0.518 |
| 7.0 | 0.764 | 0.507 | 0.592 | 0.507 |
| 8.0 | 0.789 | 0.515 | 0.607 | 0.515 |
| 9.0 | 0.812 | 0.503 | 0.605 | 0.503 |

## 9. Hipóteses e decisões de modelagem derivadas da EDA

- **H1 — A geografia domina o sinal.** UF e região separam mais as taxas do que qualquer indicador socioeconômico (CE 82% vs RN 30%; região 42%–55%). Decisão: UF, região, Amazônia Legal e capital entram como categóricas; modelos de árvore para capturar interações.

- **H2 — O contexto socioeconômico pesa pouco no indivíduo e só na base da distribuição.** Correlações |r| < 0,15 e taxa plana acima do 3º decil de IDHM. Decisão: esperar ROC-AUC modesto no grão aluno, ler o valor do modelo na interpretação e no grão município.

- **H3 — O rótulo carrega a ausência.** 13% dos alunos são ausentes e contam como não alfabetizados. Decisão: manter o rótulo da fonte no supervisionado (é o indicador que o gestor recebe), reportar a sensibilidade sem ausentes e, na camada municipal, filtrar município-anos com participação baixa antes de ranquear ou projetar.

- **H4 — Cobertura desigual entre anos.** AC, DF, SP só aparecem em 2024. Decisão: `ano` fica fora das features (confundido com UF e inútil para prever 2025) e comparações anuais só entre UFs com os dois anos.

- **H5 — Variações anuais extremas em algumas UFs, por mecanismos diferentes** (queda de presença em umas, queda de resultado entre presentes em outras). Decisão: a projeção de metas não extrapola a diferença de um único ano sem limites; usa cenários, limita a taxa a [0, 100] e encolhe a tendência municipal para a da UF.

- **H6 — Features são municipais, alunos não são independentes dentro do município.** Decisão: separar treino e teste por município (GroupShuffleSplit/StratifiedGroupKFold) para medir generalização a territórios novos, e reportar também o split aleatório como comparação.
