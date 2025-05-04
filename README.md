# Repositório do Projeto – Artigo IEEE Latin America Transactions (ID: 9469)

Este repositório contém os dados e códigos utilizados no desenvolvimento do artigo submetido ao IEEE Latin America Transactions, intitulado “Local Volt-Var Control Applied in an Islanded Microgrid Using Supervised Learning Techniques" (ID de submissão: 9469).

Descrição

O projeto investiga o uso de modelos de aprendizado de máquina para ajuste do controle Volt-Var (CVV) em inversores inteligentes operando localmente em Pontos de Acoplamento Comum (PAC), com o objetivo de melhorar a regulação de tensão em microrredes operando de forma ilhada. Este repositório oferece todos os recursos necessários para replicar os experimentos descritos no artigo.

Conteúdo do Repositório

•	Dataset_CVV_v.3.xlsx
Arquivo contendo o conjunto de dados utilizado para o treinamento e validação dos modelos. O dataset inclui variáveis relacionadas à tensão (magnitude e ângulo), além de rótulos com as classificações de estado da tensão.

•	CVV_MR_2025.py
Script em Python com a implementação dos modelos de aprendizado de máquina utilizados no artigo. O código inclui pré-processamento, definição da arquitetura da RNA, treinamento e avaliação dos modelos.

Requisitos

Para executar o script, recomenda-se a utilização de um ambiente Python 3.11.5 com as seguintes bibliotecas instaladas:

•	pandas
•	numpy
•	scikit-learn
•	imblearn
•	seaborn
•	matplotlib
•	tensorflow 

Certifique-se de revisar o script para confirmar a versão exata das bibliotecas utilizadas, conforme a configuração do seu ambiente de desenvolvimento.

Instruções de Reprodução
1.	Clonar este repositório;
2.	Abra o arquivo CVV_MR_2025.py e ajuste os caminhos de leitura dos arquivos, se necessário;
3.	Execute o script para treinar os modelos e gerar os resultados descritos no artigo.

Observação: Execute um script por vez. O código apresenta os três modelos de ML na mesma descrição.
#
#
#
#
#
#English Version:

#Project Repository – IEEE Latin America Transactions Article (ID: 9469)

This repository contains the data and code used in the development of the article submitted to IEEE Latin America Transactions, entitled “Local Volt-Var Control Applied in an Islanded Microgrid Using Supervised Learning Techniques” (Submission ID: 9469).

Description
The project investigates the use of machine learning models to adjust the Volt-Var Control (VVC) in smart inverters operating locally at Points of Common Coupling (PCC), aiming to improve voltage regulation in islanded microgrids. This repository provides all necessary resources to replicate the experiments presented in the article.

Repository Contents
•	Dataset_CVV_v.3.xlsx
Excel file containing the dataset used for training and validating the models. The dataset includes variables related to voltage (magnitude and angle), along with labels indicating the voltage condition classifications.

•	CVV_MR_2025.py
Python script implementing the machine learning models used in the study. The code includes preprocessing, definition of the MLP architecture, training, and evaluation of the models.

Requirements
To run the script, it is recommended to use a Python 3.11.5 environment with the following libraries installed:

•	pandas
•	numpy
•	scikit-learn
•	imblearn
•	seaborn
•	matplotlib
•	tensorflow

Make sure to review the script to confirm the exact version of the libraries required, depending on your development environment setup.

Reproduction Instructions
1.	Clone this repository;
2.	Open the CVV_MR_2025.py file and adjust the file paths if necessary;
3.	Run the script to train the models and reproduce the results presented in the article.
   
Note: Run one script section at a time. The code includes three ML models in the same script.
