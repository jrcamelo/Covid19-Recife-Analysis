# Análise de Dados Públicos da COVID-19 em Recife Utilizando Aprendizagem de Máquina

This repository contains the code for a graduation thesis project in the field of Artificial Intelligence.  
The project is about the analysis of public data about COVID-19 in Recife, Brazil.  

Machine learning methods used were: k-NN, Decision Trees, Random Forests, Gradient Boosting, XGBoost, and LightGBM.  
The implementation of the project was done in Python 3, mainly using the libraries pandas and sklearn.  

You can find the implementation at [/code](code), and the LaTeX document at [document.pdf](document.pdf).

---

## RESUMO
O combate à COVID-19 se tornou um grande desafio da saúde mundial, sendo documentados mais de 600 mil casos em Recife, Pernambuco, até setembro de 2021, onde mais de 7 mil destes resultaram no óbito do paciente. Por meio da ampla coleta de dados demográficos e sintomáticos realizados pela Prefeitura do Recife, este trabalho comparou a efetividade de diferentes métodos de classificação por aprendizagem de máquina na análise de fatores de risco para casos graves da doença e óbito do paciente. Dados relacionados à vacinação foram utilizados de modo a identificar o progresso da vacinação na ocorrência de cada caso. O modelo _XGBoost_ alcançou uma acurácia média de 92% na previsão de casos graves, e 95% na predição de óbitos. Foram investigados também cenários sem dados sintomáticos, representando pacientes pré-clínicos, e filtragens por progresso da vacinação, para identificar mudanças nos fatores de risco. Interpretações dos modelos gerados foram discutidos, percebendo-se idade elevada, doenças cardiovasculares, hipertensão, diabetes e obesidade como maiores riscos para casos graves e óbitos, sendo a presença da vacinação um fator decisivo na diminuição da severidade dos casos.

**Palavras-chaves** : Aprendizagem de máquina. COVID-19. Fatores de risco. Vacinação.

## ABSTRACT

Combating COVID-19 has become a major global health challenge, with more than 600,000 cases having been documented in Recife, Pernambuco, until September 2021, where more than 7,000 of these resulted in the patient’s death. Through the extensive collection of demographic and symptomatic data carried out by the City of Recife, this work compared the effectiveness of different classification methods by machine learning in the analysis of risk factors for severe cases of the disease and patient’s death. Data related to vaccination were used to identify the progress of vaccination in the occurrence of each case. The XGBoost model achieved an average accuracy of 92% in predicting severe cases, and 95% in predicting death. Scenarios without symptomatic data, representing pre-clinical patients, and screening for vaccination progress were also investigated to identify changes in risk factors. Interpretations of the generated models were discussed, noting old age, cardiovascular diseases, hypertension, diabetes and obesity as greater risks for severe cases and deaths, with the presence of vaccination being a decisive factor in reducing the severity of cases.

**Keywords** : COVID-19. Machine learning. Risk factors. Vaccination.

## SUMÁRIO
- 1 INTRODUÇÃO 
   - 1.1 MOTIVAÇÃO
   - 1.2 OBJETIVOS
      - 1.2.1 Gerais
      - 1.2.2 Específicos
- 2 CONTEXTO
   - 2.1 APRENDIZADO DE MÁQUINA NA SAÚDE
   - 2.2 ALGORITMOS DE CLASSIFICAÇÃO
      - 2.2.1 k-Nearest Neighbors
      - 2.2.2 Decision Tree
      - 2.2.3 Random Forest
      - 2.2.4 Gradient Boosting
         - 2.2.4.1 XGBoost
         - 2.2.4.2 Light Gradient Boosting
   - 2.3 MÉTRICAS DE DESEMPENHO
      - 2.3.1 Matriz de Confusão
      - 2.3.2 Acurácia
      - 2.3.3 Precisão
      - 2.3.4 Valor Preditivo Negativo
      - 2.3.5 Precisão Macro
      - 2.3.6 Sensibilidade
      - 2.3.7 Especificidade
      - 2.3.8 Sensibilidade Macro
      - 2.3.9 F1-Score
      - 2.3.10 F1-Score Macro
      - 2.3.11 AUC ROC
   - 2.4 TRABALHOS RELACIONADOS
      - 2.4.1 Trabalhos Internacionais
      - 2.4.2 Trabalhos no Brasil
- 3 METODOLOGIA
   - 3.1 FERRAMENTAS UTILIZADAS
   - 3.2 CONJUNTOS DE DADOS
      - 3.2.1 Casos Leves
      - 3.2.2 Casos Graves
      - 3.2.3 Vacinômetro
   - 3.3 PRÉ-PROCESSAMENTO DE DADOS
      - 3.3.1 Formatação
      - 3.3.2 Filtragem
         - 3.3.2.1 Colunas dos Casos Leves e Graves
         - 3.3.2.2 Colunas da Vacinação
      - 3.3.3 Interpretação
         - 3.3.3.1 Sintomas
         - 3.3.3.2 Doenças Preexistentes
         - 3.3.3.3 Interpretando a Vacinação
      - 3.3.4 Categorização
      - 3.3.5 Normalização
      - 3.3.6 Conjunto de Dados Final
   - 3.4 EXPERIMENTO
      - 3.4.1 Etapa de Treinamento
      - 3.4.2 Etapa de Teste
      - 3.4.3 Cálculo de Métricas de Desempenho
      - 3.4.4 Otimização de Parâmetros
      - 3.4.5 Geração de Gráficos
      - 3.4.6 Subconjuntos
         - 3.4.6.1 Omitir dados sintomáticos
         - 3.4.6.2 Possibilidade de óbito
         - 3.4.6.3 Progresso da vacinação
- 4 RESULTADOS
   - 4.1 ANÁLISE DO CONJUNTO DE DADOS
   - 4.2 COMPARAÇÃO ENTRE ALGORITMOS
   - 4.3 COMPARAÇÃO ENTRE ALGORITMOS EM SUBCONJUNTOS
      - 4.3.1 Omitindo Dados Sintomáticos
      - 4.3.2 Prevendo Óbitos
      - 4.3.3 Filtrando por Progresso de Vacinação
         - 4.3.3.1 Casos antes da vacina
         - 4.3.3.2 Casos com 30% da população vacinada
   - 4.4 GRÁFICOS SHAP
      - 4.4.1 Conjunto de dados balanceado - XGBoost
      - 4.4.2 Subconjunto de dados sem colunas de sintomas - Gradient Boosting
      - 4.4.3 Subconjunto de dados de vacinação - Light Gradient Boosting
      - 4.4.4 Subconjunto de dados de óbitos - XGBoost
- 5 CONCLUSÃO
   - 5.1 TRABALHOS FUTUROS
- REFERÊNCIAS 
- APÊNDICE A – MAPA DE CALOR DE CORRELAÇÃO DA VACINAÇÃO 
- APÊNDICE B – MAPA DE CALOR DE CORRELAÇÃO ANTES DA VACINAÇÃO 
- APÊNDICE C – MAPA DE CALOR DE CORRELAÇÃO COM 30% DE VACINAÇÃO 