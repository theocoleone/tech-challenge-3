# Dicionário de dados — `data/base_analitica.parquet`

Grão: um aluno do 2º ano do Ensino Fundamental avaliado no SAEB Alfabetização (2023 ou 2024).
Chaves de junção: `id_municipio` (código IBGE de 7 dígitos) e `ano`.

## Identificadores e alvo — SAEB Alfabetização (Silver do TC2)

| Coluna | Tipo | Descrição |
|---|---|---|
| `id_aluno` | string | Identificador anonimizado do aluno, atribuído por edição (o mesmo código pode reaparecer em outro ano para outro aluno). Não é feature. |
| `id_municipio` | string | Código IBGE do município da escola. Chave de junção; não é feature. |
| `nome_municipio` | string | Nome do município. Só para relatórios; não é feature. |
| `ano` | int | Ano da avaliação (2023 ou 2024). Reconstruído da partição `ano=` no S3. AC, DF e SP só têm 2024. |
| `rede` | string | Dependência administrativa, código SAEB: `2` estadual, `3` municipal, `4` privada (25 registros). Feature categórica. |
| `sigla_uf` | string | UF da escola. Feature categórica. |
| `alfabetizado` | int | **Alvo.** 1 = alfabetizado, 0 = não alfabetizado, conforme rótulo da fonte. Todo aluno ausente consta como 0. |
| `presenca` | int | Presença na avaliação (1/0); 86,8% presentes. Só define população e análises de sensibilidade; nunca é feature. |

A Silver também traz `id_escola` (anonimizado, sem correspondência com o código INEP), que não é carregado.

## Contexto escolar — Censo Escolar INEP, agregado por (`id_municipio`, `ano`), prefixo `esc_`

Só escolas públicas (rede federal, estadual ou municipal) com anos iniciais do fundamental, as
mesmas redes dos alunos avaliados. Proporções variam de 0 a 1.

| Coluna | Descrição |
|---|---|
| `esc_n_escolas` | Número de escolas de anos iniciais no município no ano. |
| `esc_prop_rural` | Proporção de escolas em localização rural. |
| `esc_prop_agua_rede_publica` | Proporção com água da rede pública. |
| `esc_prop_energia_rede_publica` | Proporção com energia da rede pública. |
| `esc_prop_esgoto_rede_publica` | Proporção com esgoto da rede pública. |
| `esc_prop_internet` | Proporção com acesso à internet. |
| `esc_prop_banda_larga` | Proporção com banda larga. |
| `esc_prop_biblioteca` | Proporção com biblioteca. |
| `esc_prop_laboratorio_informatica` | Proporção com laboratório de informática. |
| `esc_prop_laboratorio_ciencias` | Proporção com laboratório de ciências. |
| `esc_prop_quadra_esportes` | Proporção com quadra de esportes. |
| `esc_prop_refeitorio` | Proporção com refeitório. |
| `esc_prop_alimentacao` | Proporção que oferece alimentação escolar. |
| `esc_prop_area_verde` | Proporção com área verde. |
| `esc_prop_patio_coberto` | Proporção com pátio coberto. |
| `esc_media_quantidade_docente_fundamental_anos_iniciais` | Média de docentes dos anos iniciais por escola. |
| `esc_media_quantidade_matricula_fundamental_anos_iniciais` | Média de matrículas dos anos iniciais por escola. |
| `esc_media_quantidade_sala_utilizada` | Média de salas de aula utilizadas por escola. |

## Contexto socioeconômico — por `id_municipio`, estático

Atlas do Desenvolvimento Humano (PNUD/Ipea/FJP), Censo 2010:

| Coluna | Descrição |
|---|---|
| `idhm` | Índice de Desenvolvimento Humano Municipal (0–1). |
| `idhm_e` | Subíndice educação. |
| `idhm_l` | Subíndice longevidade. |
| `idhm_r` | Subíndice renda. |
| `renda_pc` | Renda domiciliar per capita média (R$ de ago/2010). |
| `indice_gini` | Índice de Gini da renda domiciliar per capita (0–1). |
| `taxa_analfabetismo_15_mais` | % da população de 15 anos ou mais analfabeta. |
| `taxa_freq_liquida_fundamental` | % de 6 a 14 anos frequentando o fundamental. |
| `taxa_agua_encanada` | % da população em domicílios com água encanada. |
| `taxa_energia_eletrica` | % em domicílios com energia elétrica. |
| `taxa_coleta_lixo` | % da população urbana em domicílios com coleta de lixo. |
| `prop_pobreza` | % de pobres (renda per capita até R$ 140, ago/2010). |
| `expectativa_vida` | Esperança de vida ao nascer (anos). |

Índice de Vulnerabilidade Social (Ipea), 2010, média das UDHs do município (totais de raça, sexo e
localização). Varia de 0 a 1; maior = mais vulnerável.

| Coluna | Descrição |
|---|---|
| `ivs` | IVS geral. |
| `ivs_infraestrutura_urbana` | Dimensão infraestrutura urbana. |
| `ivs_capital_humano` | Dimensão capital humano. |
| `ivs_renda_trabalho` | Dimensão renda e trabalho. |

IBGE, 2023:

| Coluna | Descrição |
|---|---|
| `populacao` | População estimada. |
| `pib_per_capita` | PIB municipal / população (R$ correntes). |

Diretório de municípios (Base dos Dados / IBGE):

| Coluna | Descrição |
|---|---|
| `nome_regiao` | Grande região (Norte, Nordeste, Centro-Oeste, Sudeste, Sul). Feature categórica. |
| `nome_mesorregiao` | Mesorregião. Alta cardinalidade; fora do modelo. |
| `amazonia_legal` | 1 se o município integra a Amazônia Legal. Feature categórica. |
| `capital_uf` | 1 se é capital da UF. Feature categórica. |

## Colunas da Silver que não entram na base

Medidas no exame ou derivadas do alvo (vazamento; `COLUNAS_LEAKAGE` em `src/config.py`):
`proficiencia`, `preenchimento_caderno` e as descrições `*_desc`. Também ficam fora `caderno`
(administrativo), `serie` (constante "2") e `peso_aluno` (peso amostral). As colunas agregadas da
Gold listadas em `COLUNAS_LEAKAGE` (`taxa_alfabetizacao`, `media_portugues`, `gap_para_meta_2030`,
`ranking_na_uf` etc.) nunca são juntadas à base de treino.

## Camada Gold usada na aplicação estratégica

`gold/indicador_municipio`, por município e ano (partição `ano=`): `nome`, `sigla_uf`,
`taxa_alfabetizacao` (0–100), `media_portugues`, `meta_alfabetizacao_2024/2026/2028/2030` (0–100),
`gap_para_meta_2030` e `ranking_na_uf`. Só é lida no `notebooks/03_aplicacao_estrategica.py`; a
participação por município-ano é calculada a partir de `presenca` na base analítica.
