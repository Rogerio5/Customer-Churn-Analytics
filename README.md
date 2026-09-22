# Customer Churn Analytics

**Análise de cancelamento, previsão de risco e priorização de retenção de clientes.**

O **Customer Churn Analytics** é um projeto completo de análise de dados e aprendizado de máquina desenvolvido para compreender padrões de cancelamento de clientes, estimar o risco de churn e transformar previsões em informações úteis para ações de retenção.

A solução integra **Python, Machine Learning, explicabilidade, Streamlit, SQL e Power BI**.

> **Pergunta de negócio:** quais clientes apresentam maior risco de cancelar o serviço e como organizar a priorização das ações de retenção?

**Versão 1.1** · **Python 3.12.1** · **Streamlit** · **Power BI** · **Licença MIT**

[Visão geral](#visão-geral) ·
[Problema](#problema-de-negócio) ·
[Resultados](#resultados-dos-modelos) ·
[Previsão](#previsão-de-novos-clientes) ·
[Power BI](#power-bi) ·
[Como executar](#como-executar) ·
[Validação](#validação-da-versão-11)

---

## Visão geral

O projeto utiliza a base **IBM Telco Customer Churn**, composta por **7.043 clientes**, para investigar características associadas ao cancelamento do serviço.

A versão 1.1 organiza o processo de modelagem separando o desenvolvimento da avaliação final.

```text
7.043 clientes
      │
      ├── 5.634 clientes
      │        ↓
      │  Desenvolvimento
      │        ↓
      │  Validação cruzada estratificada
      │        ↓
      │  Comparação de 3 modelos
      │        ↓
      │  Seleção do modelo
      │
      └── 1.409 clientes
               ↓
          Teste final
               ↓
          Avaliação do modelo
```

Após a seleção do algoritmo, um modelo operacional é treinado para permitir a previsão de novos clientes dentro da aplicação Streamlit.

O projeto vai além da criação de um modelo preditivo. Ele conecta:

```text
Dados
  ↓
Análise
  ↓
Modelagem
  ↓
Avaliação
  ↓
Explicação
  ↓
Segmentação de risco
  ↓
Priorização de retenção
  ↓
Streamlit + Power BI
```

---

## Problema de negócio

Uma operação de retenção não consegue abordar todos os clientes com a mesma prioridade.

Por isso, é necessário identificar quais clientes apresentam maior risco estimado de cancelamento e organizar uma fila de atendimento baseada em dados.

O projeto busca apoiar perguntas como:

- quais perfis apresentam maior ocorrência histórica de churn;
- quais clientes apresentam maior probabilidade estimada de cancelamento;
- quais clientes devem ser analisados primeiro;
- quais características contribuíram para uma determinada previsão;
- como transformar o resultado do modelo em informação compreensível;
- como apresentar os resultados para acompanhamento operacional e gerencial.

O projeto não afirma que determinada característica cause churn e não garante que um cliente classificado como alto risco irá cancelar.

---

## Principais indicadores

| Indicador | Resultado |
|---|---:|
| Clientes na base | **7.043** |
| Clientes com churn | **1.869** |
| Taxa histórica de churn | **26,54%** |
| Conjunto de desenvolvimento | **5.634** |
| Conjunto de teste final | **1.409** |
| Modelos comparados | **3** |
| Validação cruzada | **5 folds** |
| Modelo selecionado | **Logistic Regression** |
| ROC-AUC médio na validação cruzada | **0,8436** |
| ROC-AUC no teste final | **0,8481** |
| Testes automatizados | **152 aprovados** |
| Qualidade de código | **Ruff aprovado** |

---

## Fluxo do projeto

```text
IBM Telco Customer Churn
          │
          ▼
Validação dos dados
          │
          ▼
Limpeza e preparação
          │
          ▼
Análise exploratória
          │
          ▼
Análise estatística
          │
          ▼
Preparação das variáveis
          │
          ▼
Separação do teste final
          │
          ▼
Validação cruzada no desenvolvimento
          │
          ▼
┌─────────────────────────────────────┐
│ Logistic Regression                 │
│ Random Forest                       │
│ XGBoost                             │
└─────────────────────────────────────┘
          │
          ▼
Comparação pelo ROC-AUC médio
          │
          ▼
Seleção da Logistic Regression
          │
          ▼
Avaliação no teste final
          │
          ▼
Segmentação de risco
          │
     ┌────┴─────┐
     ▼          ▼
 Streamlit   Power BI
     │
     ▼
Previsão individual
     │
     ▼
Explicação da previsão
```

---

## Dados utilizados

A base contém informações relacionadas ao perfil, relacionamento e serviços utilizados pelos clientes.

Entre as informações disponíveis estão:

- gênero;
- cliente idoso;
- parceiro;
- dependentes;
- tempo como cliente;
- serviço telefônico;
- múltiplas linhas;
- serviço de internet;
- segurança online;
- backup online;
- proteção de dispositivo;
- suporte técnico;
- streaming;
- contrato;
- fatura digital;
- forma de pagamento;
- cobrança mensal;
- cobrança acumulada;
- churn.

A variável alvo do projeto é:

```text
Churn
```

### Variáveis numéricas

```text
SeniorCitizen
tenure
MonthlyCharges
TotalCharges
```

### Variáveis categóricas

```text
gender
Partner
Dependents
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
```

---

## Análise exploratória

A análise histórica mostrou diferenças importantes entre alguns grupos de clientes.

### Churn por tipo de contrato

| Contrato | Taxa de churn |
|---|---:|
| Month-to-month | **42,71%** |
| One year | 11,27% |
| Two year | 2,83% |

Clientes com contrato mensal apresentaram a maior taxa histórica de churn entre os tipos de contrato analisados.

---

### Churn por serviço de internet

| Serviço | Taxa de churn |
|---|---:|
| Fiber optic | **41,89%** |
| DSL | 18,96% |
| No | 7,40% |

---

### Churn por forma de pagamento

| Forma de pagamento | Taxa de churn |
|---|---:|
| Electronic check | **45,29%** |
| Mailed check | 19,11% |
| Bank transfer (automatic) | 16,71% |
| Credit card (automatic) | 15,24% |

---

### Churn por tempo como cliente

| Tempo de relacionamento | Taxa de churn |
|---|---:|
| 0 a 12 meses | **47,44%** |
| 13 a 24 meses | 28,71% |
| 25 a 48 meses | 20,39% |
| 49 a 72 meses | 9,51% |

Na base analisada, a taxa de churn diminui conforme aumenta o tempo de relacionamento.

---

### Outros padrões observados

| Característica | Taxa de churn |
|---|---:|
| Sem segurança online | **41,77%** |
| Com segurança online | 14,61% |
| Sem suporte técnico | **41,64%** |
| Com suporte técnico | 15,17% |
| Fatura sem papel | **33,57%** |
| Sem fatura sem papel | 16,33% |
| Sem dependentes | **31,28%** |
| Com dependentes | 15,45% |
| Sem parceiro | **32,96%** |
| Com parceiro | 19,66% |

Esses resultados representam **associações observadas na base** e não demonstram causalidade.

---

## Análise estatística

O projeto também utiliza testes estatísticos para complementar a análise visual.

Foram aplicados:

```text
Qui-quadrado
Cramér's V
Mann–Whitney U
Tamanho de efeito
```

Essa etapa ajuda a avaliar as diferenças e associações encontradas durante a exploração dos dados.

Os testes estatísticos não são utilizados para afirmar causalidade.

---

## Preparação dos dados

O pipeline utiliza tratamento separado para variáveis numéricas e categóricas.

### Variáveis numéricas

```text
StandardScaler
```

### Variáveis categóricas

```text
OneHotEncoder
```

O pré-processamento faz parte do pipeline do modelo.

Isso ajuda a garantir que o mesmo processo utilizado durante o treinamento também seja aplicado durante a previsão de novos clientes.

---

## Estratégia de validação da versão 1.1

A versão 1.1 utiliza uma separação entre:

```text
desenvolvimento
e
teste final
```

Primeiro, **20% da base é separado como conjunto de teste final**.

Esse conjunto não participa da escolha do algoritmo.

```text
Base completa:       7.043 clientes
Desenvolvimento:     5.634 clientes
Teste final:         1.409 clientes
```

A separação do holdout utiliza:

```text
test_size = 0.20
random_state = 20260920
```

Dentro do conjunto de desenvolvimento é realizada uma validação cruzada estratificada:

```text
StratifiedKFold
5 folds
shuffle = True
random_state = 42
```

A métrica utilizada para selecionar o algoritmo é:

```text
mean_cv_roc_auc
```

Depois da seleção, o modelo escolhido é avaliado no conjunto de teste final preservado.

---

## Modelos avaliados

Foram comparados três algoritmos:

```text
Logistic Regression
Random Forest
XGBoost
```

A comparação é realizada utilizando exatamente o mesmo conjunto de desenvolvimento e a mesma estratégia de validação cruzada.

---

## Resultados dos modelos

### Validação cruzada

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0,8040** | **0,6587** | **0,5472** | **0,5968** | **0,8436** |
| XGBoost | 0,7987 | 0,6540 | 0,5151 | 0,5757 | 0,8398 |
| Random Forest | 0,7891 | 0,6280 | 0,5030 | 0,5583 | 0,8189 |

A **Logistic Regression apresentou o maior ROC-AUC médio na validação cruzada entre os três modelos avaliados**.

O resultado completo da Logistic Regression foi:

```text
ROC-AUC médio = 0,8436
Desvio padrão = 0,0066
```

Por esse critério, ela foi selecionada para a avaliação final.

---

## Avaliação no teste final

Após a seleção, a Logistic Regression foi avaliada nos **1.409 clientes reservados para teste final**.

| Métrica | Resultado |
|---|---:|
| Accuracy | **0,7984** |
| Precision | **0,6433** |
| Recall | **0,5401** |
| F1 | **0,5872** |
| ROC-AUC | **0,8481** |

### Matriz de confusão

| Resultado | Clientes |
|---|---:|
| Verdadeiro negativo | **923** |
| Falso positivo | **112** |
| Falso negativo | **172** |
| Verdadeiro positivo | **202** |

O conjunto de teste final não participou da seleção do algoritmo.

---

## Segmentação de risco

Os **1.409 clientes do conjunto de teste** foram organizados em três níveis de risco.

| Risco | Clientes | Participação |
|---|---:|---:|
| LOW | **876** | **62,17%** |
| MEDIUM | **339** | **24,06%** |
| HIGH | **194** | **13,77%** |
| **Total** | **1.409** | **100%** |

### Limites utilizados

```text
Probabilidade < 30%
→ LOW

30% ≤ Probabilidade < 60%
→ MEDIUM

Probabilidade ≥ 60%
→ HIGH
```

Essas faixas são regras operacionais adotadas no projeto.

---

## Prioridade de retenção

Cada nível de risco recebe uma prioridade operacional.

| Risco | Prioridade | Interpretação |
|---|---|---|
| LOW | MONITOR | acompanhamento normal |
| MEDIUM | ENGAGE | acompanhar e avaliar ações de relacionamento |
| HIGH | URGENT | analisar primeiro |

O objetivo é transformar uma probabilidade produzida pelo modelo em uma informação mais fácil de utilizar.

A classificação não significa que o cliente obrigatoriamente cancelará o serviço.

---

## Previsão e comportamento observado

No conjunto de teste, também foi comparada a probabilidade média prevista com a taxa de churn observada dentro de cada faixa.

| Risco | Probabilidade média prevista | Churn observado |
|---|---:|---:|
| HIGH | **69,48%** | **70,10%** |
| MEDIUM | **45,41%** | **45,13%** |
| LOW | **9,62%** | **9,70%** |

As médias ficaram próximas das taxas observadas nas três faixas analisadas.

Essa comparação é descritiva e **não substitui uma avaliação formal de calibração das probabilidades**.

---

## Previsão de novos clientes

A versão 1.1 adiciona uma funcionalidade para calcular o risco de churn de um novo cliente diretamente na aplicação Streamlit.

O usuário informa as características do cliente no formulário.

Entre os dados utilizados estão:

```text
Gênero
Cliente idoso
Parceiro
Dependentes
Tempo como cliente
Serviço telefônico
Múltiplas linhas
Serviço de internet
Segurança online
Backup online
Proteção do dispositivo
Suporte técnico
Streaming de TV
Streaming de filmes
Tipo de contrato
Fatura digital
Forma de pagamento
Cobrança mensal
Total acumulado
```

Após o envio, o sistema apresenta:

```text
Probabilidade estimada de churn
Nível de risco
Prioridade de retenção
Fatores que aumentaram a estimativa
Fatores que reduziram a estimativa
Orientação operacional
```

---

## Modelo operacional

O modelo utilizado para previsão de novos clientes é persistido em:

```text
artifacts/churn_model.joblib
```

O artefato contém o pipeline necessário para realizar o pré-processamento e a previsão.

Para gerar novamente o modelo operacional:

```powershell
python scripts/train_churn_model.py
```

Após a seleção e avaliação do algoritmo, o artefato operacional é treinado com os dados disponíveis para utilização na aplicação.

---

## Explicação individual da previsão

A previsão individual utiliza uma explicação compatível com a **Logistic Regression**, o mesmo algoritmo utilizado para gerar a probabilidade apresentada ao usuário.

Para cada variável, é calculada sua contribuição para o score da previsão.

De forma simplificada:

```text
Valor transformado
       ×
Coeficiente aprendido
       =
Contribuição da variável
```

A aplicação organiza os resultados em:

```text
Sinais que aumentaram o risco estimado

Sinais que reduziram o risco estimado
```

Também é possível consultar informações técnicas mais detalhadas sobre:

```text
valor transformado
coeficiente
contribuição
intercepto
decision score
```

A explicação mostra como o modelo chegou ao resultado.

Ela não significa que determinada característica causou o comportamento do cliente.

---

## Explicabilidade global

O projeto também mantém análises de explicabilidade global utilizando **SHAP** nas análises de referência.

Dessa forma, existem duas camadas diferentes de interpretação:

```text
Visão global
→ SHAP

Previsão individual
→ contribuições da Logistic Regression
```

Isso evita utilizar a explicação de um algoritmo diferente para justificar uma previsão individual.

---

## Streamlit

A aplicação Streamlit organiza o projeto em diferentes páginas.

### Visão Geral

Apresenta informações como:

- indicadores gerais da base;
- comportamento histórico do churn;
- análise exploratória;
- resultados estatísticos;
- informações gerais do projeto.

---

### Modelos

Apresenta:

- estratégia de validação;
- comparação dos três modelos;
- resultados da validação cruzada;
- modelo selecionado;
- métricas do teste final;
- explicação das principais métricas.

---

### Fila de Retenção

A fila utiliza os clientes do conjunto de teste e permite:

- visualizar clientes por risco;
- filtrar por nível de risco;
- filtrar por contrato;
- pesquisar pelo identificador do cliente;
- ordenar pela probabilidade estimada;
- visualizar informações do perfil;
- exportar os clientes filtrados em CSV.

A lista começa pelos clientes com maior probabilidade estimada dentro dos filtros selecionados.

---

### Prever Novo Cliente

Permite:

- preencher os dados de um novo cliente;
- calcular a probabilidade de churn;
- classificar o nível de risco;
- definir a prioridade operacional;
- visualizar os principais fatores associados à previsão;
- consultar detalhes técnicos da explicação.

---

### Sobre

Apresenta informações sobre:

- objetivo do projeto;
- escopo;
- tecnologias;
- interpretação das previsões;
- limitações;
- uso responsável dos resultados.

---

## Power BI

A versão 1.1 inclui um dashboard completo desenvolvido em **Power BI**.

Arquivo:

```text
dashboards/powerbi/Customer-Churn-Analytics-V1.1.pbix
```

O dashboard está dividido em quatro páginas.

---

### 1. Visão Executiva

Apresenta os principais indicadores:

```text
Total de clientes
Clientes com churn
Taxa de churn
Alto risco
Médio risco
Baixo risco
```

Também apresenta churn por:

```text
Tipo de contrato
Serviço de internet
Forma de pagamento
```

---

### 2. Análise de Churn

Apresenta a taxa histórica de churn por:

```text
Tempo como cliente
Segurança online
Suporte técnico
Fatura sem papel
Dependentes
Parceiro
```

---

### 3. Risco e Retenção

Apresenta:

```text
Distribuição dos clientes por risco
Churn observado por nível de risco
Prioridade de retenção
Probabilidade média prevista
Risco por tipo de contrato
Risco por serviço de internet
```

---

### 4. Perfil dos Clientes

Apresenta informações sobre:

```text
Gênero
Dependentes
Parceiro
Cliente idoso
Tempo como cliente
Tipo de contrato
```

---

## Perfil da base

### Gênero

| Grupo | Participação |
|---|---:|
| Male | 50,48% |
| Female | 49,52% |

### Dependentes

| Grupo | Participação |
|---|---:|
| No | 70,04% |
| Yes | 29,96% |

### Parceiro

| Grupo | Participação |
|---|---:|
| No | 51,70% |
| Yes | 48,30% |

### Cliente idoso

| Grupo | Participação |
|---|---:|
| Não | 83,79% |
| Sim | 16,21% |

### Tempo como cliente

| Faixa | Clientes |
|---|---:|
| 0 a 12 meses | 2.186 |
| 13 a 24 meses | 1.024 |
| 25 a 48 meses | 1.594 |
| 49 a 72 meses | 2.239 |

### Tipo de contrato

| Contrato | Clientes |
|---|---:|
| Month-to-month | 3.875 |
| Two year | 1.695 |
| One year | 1.473 |

---

## SQL

O projeto também utiliza **SQLite** para consultas analíticas.

Para executar as análises:

```powershell
python scripts/run_churn_sql_analytics.py
```

Os relatórios produzidos são armazenados em:

```text
reports/
```

As consultas complementam as análises realizadas em Python e Power BI.

---

## Arquitetura da solução

```text
              IBM Telco Customer Churn
                         │
                         ▼
                 Validação dos dados
                         │
                         ▼
                Limpeza e preparação
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
      Análise exploratória     Análise estatística
             │                       │
             └───────────┬───────────┘
                         ▼
                 Preparação dos dados
                         │
                         ▼
                Separação do holdout
                         │
                         ▼
                Validação cruzada
                         │
       ┌─────────────────┼─────────────────┐
       ▼                 ▼                 ▼
Logistic Regression  Random Forest      XGBoost
       │                 │                 │
       └─────────────────┼─────────────────┘
                         ▼
                 Seleção do modelo
                         │
                         ▼
                 Teste final preservado
                         │
                         ▼
                Segmentação de risco
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
         Streamlit                Power BI
             │
             ▼
      Previsão individual
             │
             ▼
     Explicação da previsão
```

---

## Tecnologias

| Área | Tecnologias |
|---|---|
| Linguagem | Python |
| Manipulação de dados | Pandas, NumPy |
| Estatística | SciPy |
| Machine Learning | scikit-learn, XGBoost |
| Explicabilidade | SHAP e coeficientes da Logistic Regression |
| Aplicação | Streamlit |
| Consultas | SQLite |
| Business Intelligence | Power BI, DAX |
| Testes | Pytest |
| Qualidade | Ruff |
| Persistência do modelo | Joblib |
| Versionamento | Git, GitHub |

---

## Como executar

### Pré-requisitos

```text
Git
Python 3.11 ou superior
```

A validação final da versão 1.1 foi realizada utilizando:

```text
Python 3.12.1
```

---

### 1. Clonar o repositório

```powershell
git clone https://github.com/Rogerio5/Customer-Churn-Analytics.git
cd Customer-Churn-Analytics
```

---

### 2. Criar o ambiente virtual

```powershell
python -m venv .venv
```

---

### 3. Ativar o ambiente no Windows

```powershell
.\.venv\Scripts\Activate.ps1
```

---

### 4. Instalar as dependências

```powershell
python -m pip install -e ".[dev]"
```

---

### 5. Executar a aplicação

```powershell
python -m streamlit run app.py
```

O endereço local normalmente será:

```text
http://localhost:8501
```

---

## Comandos principais

### Selecionar o modelo com validação cruzada

```powershell
python scripts/select_churn_model_cv.py
```

Esse script compara os modelos utilizando a base de desenvolvimento e registra os resultados da validação cruzada.

---

### Construir a inteligência de retenção

```powershell
python scripts/build_churn_retention_v1_1.py
```

Esse processo gera os dados utilizados para:

```text
probabilidade de churn
nível de risco
prioridade de retenção
Power BI
fila de retenção
```

---

### Treinar o modelo operacional

```powershell
python scripts/train_churn_model.py
```

O modelo é salvo em:

```text
artifacts/churn_model.joblib
```

---

### Executar análises SQL

```powershell
python scripts/run_churn_sql_analytics.py
```

---

### Executar todos os testes

```powershell
python -m pytest -q
```

---

### Verificar qualidade do código

```powershell
python -m ruff check .
```

---

### Verificar compilação

```powershell
python -m compileall app.py scripts src/customer_churn tests -q
```

---

## Estrutura do projeto

```text
Customer-Churn-Analytics/
│
├── app.py
│
├── artifacts/
│   └── churn_model.joblib
│
├── dashboards/
│   └── powerbi/
│       ├── Customer-Churn-Analytics-V1.1.pbix
│       ├── data/
│       └── docs/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│
├── reports/
│   ├── churn_cv_model_comparison.csv
│   ├── churn_final_test_evaluation.csv
│   ├── churn_model_selection_v1_1.json
│   └── churn_sql_by_risk.csv
│
├── scripts/
│   ├── build_churn_retention_v1_1.py
│   ├── compare_churn_models.py
│   ├── run_churn_sql_analytics.py
│   ├── select_churn_model_cv.py
│   └── train_churn_model.py
│
├── src/
│   └── customer_churn/
│       ├── analytics/
│       ├── dashboard/
│       │   ├── about_view.py
│       │   ├── churn_view.py
│       │   ├── models_view.py
│       │   ├── predict_view.py
│       │   └── retention_view.py
│       ├── data/
│       ├── features/
│       └── models/
│           └── logistic_explainability.py
│
├── tests/
│   ├── test_standalone.py
│   └── test_v1_1_prediction_and_retention.py
│
├── pyproject.toml
├── requirements-tested.txt
├── LICENSE
└── README.md
```

---

## Validação da versão 1.1

A validação final foi realizada em **22/09/2026**.

### Ambiente

```text
Python 3.12.1
```

### Compilação

```text
PASS
```

### Ruff

```text
All checks passed!
```

### Pytest

```text
152 passed
```

### Resultado final

```text
V1_1_FINAL_VALIDATION=PASS
```

---

## Testes adicionados na versão 1.1

A versão 1.1 adicionou **18 testes específicos** relacionados às novas funcionalidades.

Entre os pontos verificados estão:

```text
Humanização das variáveis
Organização dos fatores positivos
Organização dos fatores negativos
Construção das tabelas de contribuição
Textos de orientação operacional
Explicação da Logistic Regression
Reconstrução do decision score
Validação de uma única linha por explicação
Carregamento do modelo persistido
Probabilidade entre 0 e 1
Confirmação da Logistic Regression
Distribuição da fila de retenção
Limites de LOW
Limites de MEDIUM
Limites de HIGH
Mapeamento LOW → MONITOR
Mapeamento MEDIUM → ENGAGE
Mapeamento HIGH → URGENT
```

A suíte completa passou de **134 para 152 testes**.

---

## Rastreabilidade dos resultados

Os principais resultados utilizados no projeto são registrados em arquivos próprios.

### Comparação por validação cruzada

```text
reports/churn_cv_model_comparison.csv
```

### Avaliação no teste final

```text
reports/churn_final_test_evaluation.csv
```

### Modelo selecionado e configuração

```text
reports/churn_model_selection_v1_1.json
```

Esses arquivos permitem conferir os números apresentados no projeto.

---

## Entregas

O projeto possui diferentes formas de apresentação dos resultados.

| Entrega | Objetivo |
|---|---|
| Python | análise, preparação e modelagem |
| Machine Learning | estimativa de risco de churn |
| Streamlit | exploração e previsão individual |
| Power BI | acompanhamento visual e gerencial |
| SQL | consultas analíticas |
| Explicabilidade | compreensão das previsões |
| Fila de retenção | priorização operacional |
| Relatório | documentação estruturada do trabalho |

---

## Limitações

Os resultados devem ser interpretados dentro do contexto do projeto.

A base utilizada é pública e representa um cenário específico de telecomunicações.

As relações observadas entre características e churn representam associações e não comprovam causalidade.

Os limites de:

```text
30%
60%
```

utilizados para definir os níveis de risco são regras operacionais de referência.

O projeto ainda não utiliza informações reais sobre:

```text
custo de retenção
receita do cliente
margem
custo de contato
valor de campanha
retorno financeiro
```

A proximidade entre a probabilidade média prevista e o churn observado nas faixas de risco não substitui uma avaliação formal de calibração.

Em um ambiente de produção, o modelo também precisaria ser acompanhado ao longo do tempo para identificar possíveis mudanças no comportamento dos dados.

---

## Próximas evoluções

- [ ] Configurar integração contínua com GitHub Actions.
- [ ] Adicionar imagens reais do Streamlit ao README.
- [ ] Adicionar imagens reais do Power BI ao README.
- [ ] Criar uma capa visual utilizando telas reais do projeto.
- [ ] Publicar uma demonstração online do Streamlit.
- [ ] Avaliar formalmente a calibração das probabilidades.
- [ ] Analisar custo de falsos positivos e falsos negativos.
- [ ] Avaliar diferentes limites de classificação de risco.
- [ ] Adicionar monitoramento de desempenho do modelo.
- [ ] Adicionar monitoramento de mudanças nos dados.
- [ ] Estudar integração com sistemas de relacionamento com clientes.
- [ ] Avaliar experimentos de retenção e acompanhamento de resultados.

---

## Uso responsável dos resultados

Uma probabilidade elevada de churn significa que, segundo os padrões aprendidos pelo modelo, aquele cliente apresenta maior risco estimado.

Isso não significa:

```text
que o cliente certamente irá cancelar
```

Da mesma forma, uma variável que aumentou a previsão não deve ser interpretada automaticamente como a causa do cancelamento.

O objetivo da solução é:

```text
apoiar a tomada de decisão
```

e não substituir a análise humana.

---

## Autor

**Rogério Augusto Sabino**

Projeto desenvolvido como aplicação prática de:

**Análise de Dados · Machine Learning · Explicabilidade · SQL · Streamlit · Power BI**

---

## Licença

Este projeto é distribuído sob a licença [MIT](LICENSE).