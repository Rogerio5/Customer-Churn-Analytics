# Validação — Customer Churn Analytics v1.1

**Data:** 22/09/2026  
**Versão do projeto:** 1.1.0  
**Ambiente:** Windows  
**Python:** 3.12.1

---

## Objetivo

Registrar a validação técnica da versão 1.1 do Customer Churn Analytics, incluindo seleção de modelo por validação cruzada, teste final preservado, previsão de novos clientes, explicabilidade individual, segmentação de risco, fila de retenção, Streamlit e Power BI.

## Resumo

| Item | Resultado |
|---|---:|
| Base completa | 7.043 |
| Desenvolvimento | 5.634 |
| Teste final | 1.409 |
| Validação cruzada | 5 folds |
| Modelo selecionado | Logistic Regression |
| ROC-AUC médio CV | 0,8436 |
| ROC-AUC teste final | 0,8481 |
| Pytest | 152 passed |
| Ruff | PASS |
| Compilação | PASS |

## Estratégia de validação

O teste final foi separado antes da seleção do algoritmo.

```text
test_size = 0.20
holdout_random_state = 20260920
```

A seleção utiliza somente os 5.634 registros de desenvolvimento:

```text
StratifiedKFold
folds = 5
shuffle = True
random_state = 42
selection_metric = mean_cv_roc_auc
```

Modelos avaliados:

- Logistic Regression
- XGBoost
- Random Forest

## Resultados da validação cruzada

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0,8040 | 0,6587 | 0,5472 | 0,5968 | 0,8436 |
| XGBoost | 0,7987 | 0,6540 | 0,5151 | 0,5757 | 0,8398 |
| Random Forest | 0,7891 | 0,6280 | 0,5030 | 0,5583 | 0,8189 |

A Logistic Regression apresentou o maior ROC-AUC médio entre os modelos avaliados.

```text
roc_auc_mean = 0.8435524400645331
roc_auc_std  = 0.006608776953633585
```

## Avaliação no teste final

A Logistic Regression foi avaliada nos 1.409 clientes preservados para teste final.

| Métrica | Resultado |
|---|---:|
| Accuracy | 0,7984 |
| Precision | 0,6433 |
| Recall | 0,5401 |
| F1 | 0,5872 |
| ROC-AUC | 0,8481 |

```text
accuracy  = 0.7984386089425124
precision = 0.643312101910828
recall    = 0.5401069518716578
f1        = 0.5872093023255814
roc_auc   = 0.8481451342065153
```

### Matriz de confusão

| Resultado | Clientes |
|---|---:|
| Verdadeiro negativo | 923 |
| Falso positivo | 112 |
| Falso negativo | 172 |
| Verdadeiro positivo | 202 |

## Segmentação de risco

| Risco | Clientes | Participação |
|---|---:|---:|
| LOW | 876 | 62,17% |
| MEDIUM | 339 | 24,06% |
| HIGH | 194 | 13,77% |
| Total | 1.409 | 100% |

Regras:

```text
LOW    -> probabilidade < 0.30       -> MONITOR
MEDIUM -> 0.30 <= probabilidade < 0.60 -> ENGAGE
HIGH   -> probabilidade >= 0.60      -> URGENT
```

As faixas são regras operacionais e não garantem cancelamento.

## Probabilidade prevista x churn observado

| Risco | Probabilidade média | Churn observado |
|---|---:|---:|
| HIGH | 69,48% | 70,10% |
| MEDIUM | 45,41% | 45,13% |
| LOW | 9,62% | 9,70% |

A comparação é descritiva e não substitui uma avaliação formal de calibração.

## Modelo operacional

Artefato:

```text
artifacts/churn_model.joblib
```

Algoritmo:

```text
Logistic Regression
```

Regeneração:

```powershell
python scripts/train_churn_model.py
```

O artefato contém o pipeline de pré-processamento e previsão.

## Previsão e explicabilidade

A versão 1.1 permite prever um novo cliente no Streamlit e retorna:

- probabilidade estimada;
- nível de risco;
- prioridade de retenção;
- fatores que aumentaram a estimativa;
- fatores que reduziram a estimativa;
- orientação operacional.

A explicação individual utiliza as contribuições da própria Logistic Regression:

```text
valor transformado x coeficiente = contribuição
intercepto + soma das contribuições = decision score
```

Os testes verificam a reconstrução do `decision_score`. A explicação descreve o comportamento do modelo e não comprova causalidade.

## Streamlit

Páginas principais:

- Visão Geral;
- Modelos;
- Fila de Retenção;
- Prever Novo Cliente;
- Sobre.

## Power BI

Arquivo:

```text
dashboards/powerbi/Customer-Churn-Analytics-V1.1.pbix
```

Páginas:

- Visão Executiva;
- Análise de Churn;
- Risco e Retenção;
- Perfil dos Clientes.

A presença do arquivo foi verificada. A validação automatizada deste documento refere-se ao código Python e aos testes do projeto.

## SQL

Execução:

```powershell
python scripts/run_churn_sql_analytics.py
```

Relatórios:

```text
reports/
```

## Testes automatizados

A suíte anterior possuía 134 testes. A versão 1.1 adicionou 18 testes relacionados às novas funcionalidades.

Resultado:

```text
152 passed
```

Os novos testes cobrem humanização de variáveis, fatores positivos e negativos, tabelas de contribuição, orientação operacional, explicação da Logistic Regression, reconstrução do decision score, carregamento do modelo persistido, probabilidade entre 0 e 1, distribuição de risco, limites LOW/MEDIUM/HIGH e prioridades MONITOR/ENGAGE/URGENT.

## Qualidade e compilação

```powershell
python -m ruff check .
python -m pytest -q
python -m compileall app.py scripts src/customer_churn tests -q
```

Resultados:

```text
RUFF=PASS
PYTEST=152_PASSED
COMPILE=PASS
```

## Rastreabilidade

```text
reports/churn_cv_model_comparison.csv
reports/churn_final_test_evaluation.csv
reports/churn_model_selection_v1_1.json
artifacts/churn_model.joblib
dashboards/powerbi/Customer-Churn-Analytics-V1.1.pbix
```

## Limitações

A base é pública e representa um cenário específico de telecomunicações. As associações encontradas não comprovam causalidade. Os limites de 30% e 60% são regras operacionais. O projeto ainda não utiliza dados reais de custo de retenção, margem ou retorno financeiro de campanhas. Um uso em produção exigiria monitoramento contínuo do desempenho e da distribuição dos dados.

## Resultado final

```text
CUSTOMER_CHURN_ANALYTICS_V1_1=PASS
PROJECT_VERSION=1.1.0
PYTHON=3.12.1
TOTAL_ROWS=7043
DEVELOPMENT_ROWS=5634
FINAL_TEST_ROWS=1409
CV_FOLDS=5
SELECTION_METRIC=MEAN_CV_ROC_AUC
SELECTED_MODEL=LOGISTIC_REGRESSION
CV_ROC_AUC=0.8436
FINAL_TEST_ROC_AUC=0.8481
RISK_LOW=876
RISK_MEDIUM=339
RISK_HIGH=194
PYTEST=152_PASSED
RUFF=PASS
COMPILE=PASS
```

A versão **Customer Churn Analytics v1.1** está tecnicamente validada para publicação.
