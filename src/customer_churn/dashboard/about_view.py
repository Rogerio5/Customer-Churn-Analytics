import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "telco_customer_churn_clean.csv"
)

BI_PATH = (
    ROOT
    / "data"
    / "processed"
    / "churn_customer_intelligence_bi.csv"
)

FINAL_PATH = (
    ROOT
    / "reports"
    / "churn_final_test_evaluation.csv"
)

SELECTION_PATH = (
    ROOT
    / "reports"
    / "churn_model_selection_v1_1.json"
)


@st.cache_data
def load_about_data():
    data = pd.read_csv(DATA_PATH)
    bi = pd.read_csv(BI_PATH)
    final = pd.read_csv(FINAL_PATH)

    with SELECTION_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        selection = json.load(file)

    return data, bi, final, selection


def render_simple_view(
    data,
    bi,
    final,
    selection,
):
    customers = len(data)

    churned = int(
        (data["Churn"] == "Yes").sum()
    )

    churn_rate = (
        churned
        / customers
        * 100
    )

    low = int(
        (bi["risk_level"] == "LOW").sum()
    )

    medium = int(
        (bi["risk_level"] == "MEDIUM").sum()
    )

    high = int(
        (bi["risk_level"] == "HIGH").sum()
    )

    final_row = final.iloc[0]

    st.subheader(
        "Entenda o projeto"
    )

    st.caption(
        "Esta aba explica a solução sem exigir "
        "conhecimento de Machine Learning."
    )

    with st.container(border=True):
        st.markdown(
            """
### 🎯 Qual problema queremos resolver?

Quando um cliente deixa de utilizar um serviço ou cancela
seu contrato, temos um caso de **churn**.

Para uma empresa, identificar clientes que apresentam maior
risco de cancelamento pode ajudar a organizar o trabalho
da equipe de retenção.

Este projeto utiliza dados históricos e Machine Learning para
responder à pergunta:

> **Quais clientes apresentam maior risco estimado de churn?**

O sistema não afirma que um cliente irá cancelar com certeza.

Ele gera uma **estimativa de risco** para ajudar na priorização
e análise dos clientes.
"""
        )

    st.markdown(
        "### O que significa churn?"
    )

    left, right = st.columns(2)

    with left, st.container(border=True):
        st.markdown(
            "### ✅ Não churn"
        )

        st.write(
            "O cliente continua utilizando "
            "o serviço."
        )

    with right, st.container(border=True):
        st.markdown(
            "### ⚠️ Churn"
        )

        st.write(
            "O cliente deixou de utilizar "
            "ou cancelou o serviço."
        )

    st.divider()

    st.subheader(
        "O que existe na base de dados?"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Clientes",
        f"{customers:,}",
    )

    c2.metric(
        "Clientes com churn",
        f"{churned:,}",
    )

    c3.metric(
        "Taxa de churn",
        f"{churn_rate:.2f}%",
    )

    c4.metric(
        "Teste final",
        f'{selection["final_test_rows"]:,}',
    )

    st.caption(
        "A taxa de churn acima descreve este dataset. "
        "Ela não deve ser generalizada automaticamente "
        "para outras empresas."
    )

    st.divider()

    st.subheader(
        "Como a solução funciona?"
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1, st.container(border=True):
        st.markdown("### 1️⃣ Dados")
        st.write(
            "Recebemos informações do cliente, "
            "contrato, serviços e cobranças."
        )

    with s2, st.container(border=True):
        st.markdown("### 2️⃣ Modelo")
        st.write(
            "O modelo procura padrões aprendidos "
            "nos dados históricos."
        )

    with s3, st.container(border=True):
        st.markdown("### 3️⃣ Risco")
        st.write(
            "É calculada uma probabilidade "
            "estimada de churn."
        )

    with s4, st.container(border=True):
        st.markdown("### 4️⃣ Prioridade")
        st.write(
            "O cliente recebe uma classificação "
            "para apoiar a retenção."
        )

    st.code(
        """
Dados do cliente
      |
      v
Machine Learning
      |
      v
Probabilidade estimada de churn
      |
      v
LOW / MEDIUM / HIGH
      |
      v
MONITOR / ENGAGE / URGENT
""",
        language=None,
    )

    st.divider()

    st.subheader(
        "Como interpretar o risco?"
    )

    r1, r2, r3 = st.columns(3)

    with r1, st.container(border=True):
        st.markdown(
            "### 🟢 LOW"
        )
        st.markdown(
            "**Abaixo de 30%**"
        )
        st.write(
            "Faixa de menor risco estimado."
        )
        st.markdown(
            "**Prioridade: MONITOR**"
        )

    with r2, st.container(border=True):
        st.markdown(
            "### 🟠 MEDIUM"
        )
        st.markdown(
            "**30% até menos de 60%**"
        )
        st.write(
            "Faixa intermediária de risco."
        )
        st.markdown(
            "**Prioridade: ENGAGE**"
        )

    with r3, st.container(border=True):
        st.markdown(
            "### 🔴 HIGH"
        )
        st.markdown(
            "**60% ou mais**"
        )
        st.write(
            "Faixa de maior risco estimado."
        )
        st.markdown(
            "**Prioridade: URGENT**"
        )

    st.caption(
        "As faixas LOW, MEDIUM e HIGH são regras "
        "operacionais do projeto. Elas não significam "
        "certeza de cancelamento."
    )

    st.divider()

    st.subheader(
        "Fila atual de retenção"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "🟢 Baixo risco",
        f"{low:,}",
    )

    c2.metric(
        "🟠 Risco moderado",
        f"{medium:,}",
    )

    c3.metric(
        "🔴 Alto risco",
        f"{high:,}",
    )

    st.divider()

    st.subheader(
        "Qual modelo foi escolhido?"
    )

    selected_model = selection[
        "selected_model"
    ]

    final_auc = float(
        final_row["roc_auc"]
    )

    with st.container(border=True):
        st.markdown(
            f"""
### 🤖 {selected_model}

Foram comparados três modelos:

- Logistic Regression;
- Random Forest;
- XGBoost.

O modelo selecionado foi **{selected_model}**.

Ele foi escolhido usando o resultado médio da
cross-validation.

Depois da seleção, o modelo foi avaliado em um conjunto
de teste final separado.

### ROC-AUC final: {final_auc:.4f}

Esse valor indica a capacidade do modelo de ordenar clientes
de maior e menor risco.

**Não significa que o modelo acerta 84,81% das previsões.**
"""
        )

    st.divider()

    st.subheader(
        "Como navegar no projeto?"
    )

    page1, page2 = st.columns(2)

    with page1:
        with st.container(border=True):
            st.markdown(
                "### 🏠 Visão Geral"
            )
            st.write(
                "Resumo dos clientes, churn, "
                "risco e principais padrões."
            )

        with st.container(border=True):
            st.markdown(
                "### 🔮 Prever Novo Cliente"
            )
            st.write(
                "Permite preencher os dados de um "
                "cliente e gerar uma previsão individual."
            )

    with page2:
        with st.container(border=True):
            st.markdown(
                "### 👥 Fila de Retenção"
            )
            st.write(
                "Organiza os clientes por risco "
                "e prioridade operacional."
            )

        with st.container(border=True):
            st.markdown(
                "### 📊 Modelos e Explicabilidade"
            )
            st.write(
                "Apresenta métricas, cross-validation, "
                "erros, acertos e explicabilidade."
            )


def render_technical_view(
    data,
    final,
    selection,
):
    final_row = final.iloc[0]

    selected_model = selection[
        "selected_model"
    ]

    st.subheader(
        "Metodologia de Ciência de Dados"
    )

    st.caption(
        "Esta aba apresenta os detalhes técnicos "
        "da construção e avaliação do projeto."
    )

    st.markdown(
        "### 1. Dataset"
    )

    with st.container(border=True):
        st.markdown(
            f"""
O dataset contém **{len(data):,} clientes**.

Entre as informações utilizadas estão:

- tempo como cliente;
- cobrança mensal;
- cobrança total;
- tipo de contrato;
- serviço de internet;
- suporte técnico;
- segurança online;
- forma de pagamento;
- serviços contratados;
- características cadastrais.

A variável alvo é **Churn**:

- `Yes` → cliente teve churn;
- `No` → cliente não teve churn.
"""
        )

    st.markdown(
        "### 2. Preparação dos dados"
    )

    with st.container(border=True):
        st.markdown(
            """
O projeto utiliza um pipeline de pré-processamento.

**Variáveis numéricas**

São tratadas com:

`StandardScaler`

Exemplos:

- tenure;
- MonthlyCharges;
- TotalCharges.

**Variáveis categóricas**

São transformadas usando:

`OneHotEncoder`

Exemplos:

- Contract;
- InternetService;
- PaymentMethod;
- TechSupport;
- OnlineSecurity.

O pré-processamento faz parte do mesmo pipeline do modelo,
ajudando a manter consistência entre treinamento e previsão.
"""
        )

    st.markdown(
        "### 3. Separação dos dados"
    )

    with st.container(border=True):
        st.markdown(
            f"""
A metodologia V1.1 utiliza:

**80% para desenvolvimento**

- {selection["development_rows"]:,} clientes.

**20% para teste final**

- {selection["final_test_rows"]:,} clientes.

A divisão foi feita de forma **estratificada pela variável
Churn**.

Isso ajuda a preservar a proporção das classes.

O conjunto de teste final não participa da seleção entre
os modelos.
"""
        )

    st.markdown(
        "### 4. Cross-Validation"
    )

    with st.container(border=True):
        st.markdown(
            f"""
Foi utilizada:

### {selection["cv"]["folds"]}-Fold Stratified Cross-Validation

Os dados de desenvolvimento são divididos em cinco partes.

Em cada rodada:

1. quatro partes são usadas para treinamento;
2. uma parte é usada para validação;
3. o processo é repetido;
4. cada parte participa da validação uma vez.

A estratificação ajuda a manter a proporção de churn
em cada divisão.
"""
        )

    st.markdown(
        "### 5. Modelos avaliados"
    )

    m1, m2, m3 = st.columns(3)

    with m1, st.container(border=True):
        st.markdown(
            "### Logistic Regression"
        )
        st.write(
            "Modelo probabilístico linear, "
            "interpretável e utilizado na versão final."
        )

    with m2, st.container(border=True):
        st.markdown(
            "### Random Forest"
        )
        st.write(
            "Combina várias árvores de decisão "
            "para produzir a classificação."
        )

    with m3, st.container(border=True):
        st.markdown(
            "### XGBoost"
        )
        st.write(
            "Utiliza árvores construídas de forma "
            "sequencial para corrigir erros anteriores."
        )

    st.markdown(
        "### 6. Como o modelo foi selecionado?"
    )

    with st.container(border=True):
        st.markdown(
            f"""
A métrica principal utilizada para seleção foi:

### ROC-AUC médio da Cross-Validation

Modelo selecionado:

### ✅ {selected_model}

ROC-AUC médio da cross-validation:

### {selection["selected_cv_roc_auc"]:.4f}

A seleção ocorreu antes da avaliação no conjunto de teste final.
"""
        )

    st.markdown(
        "### 7. Resultado no teste final"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Accuracy",
        f'{final_row["accuracy"]:.4f}',
    )

    c2.metric(
        "Precision",
        f'{final_row["precision"]:.4f}',
    )

    c3.metric(
        "Recall",
        f'{final_row["recall"]:.4f}',
    )

    c4.metric(
        "F1",
        f'{final_row["f1"]:.4f}',
    )

    c5.metric(
        "ROC-AUC",
        f'{final_row["roc_auc"]:.4f}',
    )

    st.caption(
        "Essas métricas correspondem ao conjunto "
        "de teste final da V1.1."
    )

    st.markdown(
        "### 8. Explicabilidade"
    )

    with st.container(border=True):
        st.markdown(
            """
O projeto utiliza duas abordagens de explicabilidade.

**Logistic Regression**

É o modelo selecionado.

Os coeficientes ajudam a entender quais características
empurram o score para maior ou menor risco.

Na previsão individual também são mostradas as contribuições
das variáveis daquele cliente.

**XGBoost + SHAP**

O XGBoost foi mantido como análise complementar.

O SHAP permite estudar a importância das variáveis nas
previsões do modelo baseado em árvores.

**Importância preditiva não significa causalidade.**
"""
        )

    st.markdown(
        "### 9. Threshold"
    )

    with st.container(border=True):
        st.markdown(
            """
Para transformar a probabilidade em uma classe de churn,
o modelo utiliza atualmente:

### threshold = 0,50

Isso significa:

- abaixo de 50% → classe não churn;
- 50% ou mais → classe churn.

Alterar esse limite pode modificar:

- Precision;
- Recall;
- F1;
- falsos positivos;
- falsos negativos.

O melhor threshold pode depender do custo de negócio.
"""
        )

    st.markdown(
        "### 10. Limitações"
    )

    with st.container(border=True):
        st.markdown(
            """
O projeto possui limitações que precisam ser consideradas:

- utiliza um dataset de demonstração;
- resultados não devem ser generalizados automaticamente;
- as probabilidades ainda podem ser avaliadas quanto à calibração;
- o threshold pode ser otimizado conforme o negócio;
- associação não significa causalidade;
- comportamento dos clientes pode mudar ao longo do tempo;
- um modelo real em produção precisaria de monitoramento;
- drift de dados e desempenho precisariam ser acompanhados.

Reconhecer limitações faz parte de um projeto responsável
de Ciência de Dados.
"""
        )

    st.markdown(
        "### 11. Tecnologias"
    )

    with st.container(border=True):
        st.markdown(
            """
**Análise e dados**

`Python` · `Pandas` · `SQL`

**Machine Learning**

`scikit-learn` · `XGBoost`

**Explicabilidade**

`Logistic Regression` · `SHAP`

**Aplicação**

`Streamlit`

**Qualidade**

`pytest` · `Ruff`

**Conceitos aplicados**

EDA · Estatística · Feature Engineering · Pipeline ·
Cross-Validation · Holdout · Classificação ·
Explicabilidade · Retenção.
"""
        )

    st.markdown(
        "### 12. O que este projeto demonstra?"
    )

    with st.container(border=True):
        st.markdown(
            """
Este projeto demonstra um fluxo de Ciência de Dados que vai
além de simplesmente treinar um algoritmo.

Ele reúne:

- entendimento do problema de negócio;
- análise exploratória;
- tratamento de dados;
- SQL;
- estatística;
- preparação de features;
- Machine Learning;
- comparação entre modelos;
- cross-validation;
- avaliação final;
- explicabilidade;
- previsão de novos clientes;
- segmentação de risco;
- fila de retenção;
- dashboard;
- testes automatizados;
- reprodutibilidade.

O objetivo é transformar dados e Machine Learning em uma
solução que também possa ser entendida e utilizada por pessoas
de negócio.
"""
        )


def render_about_page():
    (
        data,
        bi,
        final,
        selection,
    ) = load_about_data()

    st.title(
        "Sobre o Projeto"
    )

    st.caption(
        "Conheça o problema de negócio, o funcionamento "
        "da solução e a metodologia de Ciência de Dados."
    )

    simple_tab, technical_tab = st.tabs(
        [
            "👤 Entenda o projeto",
            "🧪 Metodologia",
        ]
    )

    with simple_tab:
        render_simple_view(
            data=data,
            bi=bi,
            final=final,
            selection=selection,
        )

    with technical_tab:
        render_technical_view(
            data=data,
            final=final,
            selection=selection,
        )

    st.divider()

    st.caption(
        "Customer Churn Analytics · V1.1"
    )


if __name__ == "__main__":
    render_about_page()