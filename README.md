# crm_opportunity_loss_prediction

## Visão Geral do Projeto

Este projeto, atualmente **em desenvolvimento**, tem como objetivo principal criar uma solução de Machine Learning para prever a probabilidade de uma oportunidade de venda ser perdida. Tudo isso será apresentado através de uma **interface que simula um sistema de CRM (Customer Relationship Management) com foco em vendas**.

Em ambientes de vendas B2B, especialmente aqueles com alto valor agregado e ciclos longos, a capacidade de prever o desfecho de uma negociação é crucial. A perda de uma única oportunidade pode representar um impacto financeiro significativo. Este projeto visa demonstrar como a inteligência artificial pode ser aplicada para identificar proativamente quais oportunidades estão em risco de não serem convertidas, permitindo que as equipes de vendas e a gerência intervenham estrategicamente.

Este repositório demonstrará o ciclo completo de um projeto de Ciência de Dados, desde o entendimento do negócio e preparação dos dados, até a modelagem e a simulação de deploy de um modelo preditivo em uma arquitetura moderna.

---

## 1. Business Understanding (Entendimento do Negócio)

### 1.1. Problema de Negócio

Em vendas B2B, com suas complexidades e negociações de alto valor, a **perda de oportunidades de vendas** é um problema central que afeta diretamente a receita e a eficiência da equipe. Muitas vezes, o esforço e os recursos são investidos em propostas que, no final, não se concretizam. Nosso objetivo é mitigar essa perda.

### 1.2. Objetivo do Negócio

O objetivo é **desenvolver um modelo preditivo que, integrado a um CRM simulado, identifique proativamente oportunidades de vendas com alta probabilidade de serem perdidas**. Isso permitirá que:

* A equipe de vendas receba **alertas precoces** sobre oportunidades em risco diretamente na interface simulada do CRM.
* Sejam implementadas **ações preventivas** (ex: ofertas especiais, reuniões estratégicas, realocação de vendedores) para tentar reverter a situação.
* Haja uma **otimização do tempo e recursos** da equipe, focando nos casos mais promissores ou naqueles que necessitam de intervenção crítica.

### 1.3. Impacto dos Erros de Predição

Em nosso cenário de vendas B2B de alto valor:

* **Falso Negativo (Mais Crítico):** O modelo falha em prever uma perda que de fato acontece. O custo aqui é a **perda total da venda**, um impacto financeiro considerável para a empresa. Nossa prioridade será **minimizar falsos negativos** para garantir que a maioria das oportunidades em risco seja sinalizada.
* **Falso Positivo (Menos Crítico):** O modelo prevê uma perda, mas a oportunidade é ganha. Embora possa gerar custos (ex: descontos desnecessários, esforço extra da equipe, campanhas direcionadas), esses custos são considerados menores do que a perda completa de uma venda de alto valor.

---

## 2. Data Understanding (Entendimento dos Dados)
*(Esta seção será preenchida após a exploração inicial dos dados do Kaggle. Aqui você descreverá os datasets, as colunas principais e a identificação da variável alvo.)*

---

## 3. Data Preparation (Preparação dos Dados)
*(Esta seção detalhará as etapas de limpeza, tratamento de valores ausentes, engenharia de features e união dos datasets.)*

---

## 4. Modeling (Modelagem)
*(Aqui serão apresentados os modelos de Machine Learning testados, o processo de treinamento e otimização.)*

---

## 5. Evaluation (Avaliação)
*(Discussão das métricas de avaliação do modelo, com foco em Precision, Recall, F1-Score e AUC-ROC, e a Matriz de Confusão.)*

---

## 6. Arquitetura da Solução
*(Esta seção descreverá a arquitetura com FastAPI para o modelo e Streamlit para a interface de simulação de CRM. **Pode incluir um pequeno diagrama.**)*
### 6.1. API de Predição (FastAPI)
*(Descreva a funcionalidade da API: endpoint de predição, como ela recebe os dados e retorna a probabilidade.)*
### 6.2. Interface de Usuário (Streamlit - Simulação de CRM)
*(Descreva a funcionalidade da interface: como o usuário interage, preenche os dados da oportunidade simulada e visualiza o resultado da predição. **Pode incluir screenshots da interface.**)*

---

## 7. Como Rodar o Projeto
*(Instruções detalhadas para configurar o ambiente, instalar dependências e rodar a aplicação localmente.)*

---

## 8. Tecnologias Utilizadas
* Python
* Pandas
* Scikit-learn
* FastAPI
* Streamlit
* Docker (futuramente, para deploy)
* Git / GitHub

---