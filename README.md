# crm_opportunity_loss_prediction

## Visão Geral do Projeto

Este projeto, atualmente **em desenvolvimento**, visa criar uma solução de Machine Learning para prever a probabilidade de uma oportunidade de venda ser perdida em um cenário de **vendas B2B complexas e de alto valor agregado**.

Em ambientes de vendas B2B, a perda de uma única oportunidade pode representar um impacto financeiro significativo. Identificar proativamente quais oportunidades estão em risco de não serem convertidas permite que as equipes de vendas e a gerência intervenham estrategicamente, realocando esforços e recursos para maximizar as taxas de sucesso.

Este repositório demonstrará o ciclo completo de um projeto de Ciência de Dados, desde o entendimento do negócio e preparação dos dados, até a modelagem e a simulação de deploy de um modelo preditivo.

---

## 1. Business Understanding (Entendimento do Negócio)

### 1.1. Problema de Negócio

Em vendas B2B, especialmente aquelas com alto valor agregado e ciclos longos, a capacidade de prever o desfecho de uma negociação é crucial. O principal problema a ser abordado é a **perda de oportunidades de vendas**, que impacta diretamente a receita e a eficiência da equipe de vendas ao gastar tempo em propostas que não se concretizarão.

### 1.2. Objetivo do Negócio

O objetivo é **desenvolver um sistema preditivo que identifique proativamente oportunidades de vendas com alta probabilidade de serem perdidas**. Isso permitirá que:

* A equipe de vendas receba **alertas precoces** sobre oportunidades em risco.
* Sejam implementadas **ações preventivas** (ex: ofertas especiais, reuniões estratégicas, realocação de vendedores, etc.) para tentar reverter a situação.
* Haja uma **otimização do tempo e recursos** da equipe, focando nos casos mais promissores ou naqueles que necessitam de intervenção crítica.

### 1.3. Impacto dos Erros de Predição

Em nosso cenário de vendas B2B de alto valor:

* **Falso Negativo (Mais Crítico):** O modelo falha em prever uma perda que de fato acontece. O custo aqui é a **perda total da venda**, um impacto financeiro considerável para a empresa. Nossa prioridade será **minimizar falsos negativos** para garantir que a maioria das oportunidades em risco seja sinalizada.
* **Falso Positivo (Menos Crítico):** O modelo prevê uma perda, mas a oportunidade é ganha. Embora possa gerar custos (ex: descontos desnecessários, esforço extra da equipe, campanhas direcionadas), esses custos são considerados menores do que a perda completa de uma venda de alto valor.

---

## 2. Data Understanding (Entendimento dos Dados)

### 2.1. Visão Geral dos Dados e Variável Alvo

O projeto utiliza um conjunto de dados fictício de **oportunidades de vendas B2B**, obtido do Kaggle, para simular um cenário de negócio real. Os dados são fornecidos em quatro tabelas principais:
* `sales_pipeline`: Contém informações detalhadas sobre cada oportunidade, como ID, agente de vendas, produto, conta, estágio do negócio, datas de engajamento e fechamento, valor de fechamento e o status (`target`).
* `accounts`: Detalhes sobre as contas dos clientes, incluindo setor, ano de fundação, faturamento, número de funcionários e localização.
* `products`: Informações sobre os produtos envolvidos nas oportunidades, como série e preço de venda.
* `sales_teams`: Detalhes sobre a estrutura da equipe de vendas, incluindo agentes, gerentes e escritórios regionais.

Todas essas tabelas foram unificadas em um único DataFrame (`df_eda_consolidated`) para facilitar a análise exploratória. A **variável alvo** para o nosso modelo preditivo é a coluna `target`, onde `0` indica uma oportunidade **ganha** e `1` indica uma oportunidade **perdida**.

### 2.2. Principais Insights da Análise Exploratória de Dados (EDA)

A análise exploratória revelou diversos padrões e fatores que podem estar associados à perda de oportunidades de vendas:

#### 2.2.1. Análise por Características do Cliente

Avaliamos como as características dos clientes impactam o desfecho das oportunidades:

* **Setor:** Identificamos que **Finanças**, **Emprego** e **Telecomunicações** são os setores com as maiores taxas de perda de oportunidades. Em contraste, setores como o Governamental e de Educação apresentaram as menores taxas de perda.

![Gráfico da Taxa de Perda por Setor](images/taxa_perda_setor.png).

* **Porte da Empresa (Faturamento e Número de Funcionários):** Contrário à intuição inicial, a análise mostrou que oportunidades com empresas de **faturamento e número de funcionários muito grandes** (quintil superior), assim como as **muito pequenas** (quintil inferior), tendem a apresentar taxas de perda ligeiramente mais elevadas do que as de porte médio. Isso sugere que ambos os extremos do espectro de clientes podem ter complexidades de venda distintas.

![Taxa de Perda por Quintil de Faturamento e Número de Funcionários](images/taxa_perda_quintil_func_fat.png).

#### 2.2.2. Análise por Características do Produto

Nesta seção, exploramos como as características dos produtos se relacionam com a perda de oportunidades:

* **Produto e Série:** A taxa de perda varia entre os produtos, com o **"MG Advanced"** apresentando a maior taxa de perda (~30.45%), sugerindo que pode ser o mais desafiador de vender. As séries de produtos principais (**GTX** e **MG**) mostraram taxas de perda quase idênticas, indicando que a série em si não é um diferencial significativo na probabilidade de perda, mas sim o produto específico dentro da série.

![Taxa de Perda por Produto](images/taxa_perda_produto.png).

* **Preço de Venda (`sales_price`):** A análise do preço de venda não revelou uma correlação linear forte com a taxa de perda. Oportunidades com preços muito altos ou muito baixos não se mostraram consistentemente mais ou menos propensas a serem perdidas, indicando que o preço, isoladamente, pode não ser o fator decisivo para o desfecho da venda.

![Distribuição Preço Venda Op Perdidas vs Não perdidas](images/box_plot_preco_venda.png).

#### 2.2.3. Análise por Desempenho da Equipe de Vendas

Avaliou-se o impacto do desempenho da equipe de vendas nas taxas de perda:

* **Agente de Vendas (`sales_agent`):** Há uma **variação significativa no desempenho individual** dos agentes. Agentes como **Donn Cantrell (~42.55%)** e **Garret Kinder (~39.02%)** apresentaram taxas de perda notavelmente mais altas, indicando possíveis áreas para treinamento ou revisão de estratégias. Em contrapartida, agentes como **Wilburn Farren (~21.82%)** e **Hayden Neloms (~22.28%)** demonstraram performance superior, cujas melhores práticas poderiam ser replicadas.

![Taxa de Perda por Agente](images/taxa_perda_agentes.png).

* **Gerente (`manager`):** A maioria dos gerentes apresentou taxas de perda similares (entre 27% e 28%), mas **Rocco Neubert** se destacou com uma taxa visivelmente maior (~31.80%). Isso sugere que a equipe sob sua gerência pode enfrentar desafios específicos ou que suas abordagens necessitam de reavaliação.

![Taxa de Perda por Gerente](images/taxa_perda_gerentes.png).

* **Escritório Regional (`regional_office`):** A região **East (Leste)** apresentou a maior taxa de perda (~29.99%), enquanto a região **West (Oeste)** teve a menor (~27.06%). Essa disparidade regional pode indicar a influência de fatores locais como concorrência ou características de mercado.

![Taxa de Perda por Escritorio](images/taxa_perda_regiao.png).

#### 2.2.4. Análise Temporal e de Valor

Esta seção investigou a dinâmica do tempo e do valor das oportunidades:

* **Duração da Oportunidade (`opportunity_duration_days`):** Oportunidades que foram **perdidas** tenderam a ter um ciclo de vendas **mais curto** (média de ~41.48 dias) em comparação com as oportunidades **ganhas** (média de ~51.78 dias). Isso pode sugerir que negociações que se estendem um pouco mais ou que demandam um engajamento mais prolongado têm maior probabilidade de sucesso.

![Distribuição Duração Op Perdidas vs Não perdidas](images/box_plot_duracao.png).

* **Valor da Oportunidade (`close_value`):** O dataset registra um `close_value` de `0.0` para todas as oportunidades perdidas. Para as oportunidades **ganhas**, o valor médio é de aproximadamente **$1.581,40**. A distribuição dos valores ganhos mostra uma concentração em valores menores, mas com uma "cauda longa" de vendas de alto valor que elevam a média geral.

![Distribuição Valor Op Ganhas](images/distribuicao_valor_op_ganhas.png).

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
*(Esta seção descreverá a arquitetura com FastAPI para o modelo e Streamlit para a interface de simulação de CRM.)*

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