# (10) - Modelo MLP-1 - 24/11/2024 (Final)

# # Importando bibliotecas
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# from imblearn.over_sampling import SMOTE
# from sklearn.utils import shuffle
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Parâmetros ajustáveis
# n_simulacoes = 100  # Número de simulações
# usar_minmax = True  # Escolha do normalizador: True para MinMaxScaler, False para StandardScaler
# embaralhar_dados = True  # Embaralhar os dados antes do split
#
# # Leia o arquivo Excel
# df = pd.read_excel('Dataset_CVV_v.3.xlsx')
#
# # Remova espaços em branco dos nomes das colunas
# df.columns = df.columns.str.strip()
#
# # Definindo as variáveis preditoras e o rótulo
# X = df[['V1', 'V2', 'V3', 'V1 ang', 'V2 ang', 'V3 ang']]
# y = df['Saida']
#
# # Embaralhar os dados se necessário
# if embaralhar_dados:
#     X, y = shuffle(X, y, random_state=None)
#
# # Função para rodar uma simulação
# def rodar_simulacao():
#     # Dividindo o dataset em treino e teste
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, stratify=y, random_state=None
#     )
#
#     # Balanceando as classes com SMOTE
#     smote = SMOTE(random_state=None)
#     X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
#
#     # Normalizando os dados
#     scaler = MinMaxScaler() if usar_minmax else StandardScaler()
#     X_train_bal = scaler.fit_transform(X_train_bal)
#     X_test = scaler.transform(X_test)
#
#     # Criando e treinando o modelo MLP
#     mlp = MLPClassifier(hidden_layer_sizes=(12), activation='tanh', max_iter=5000, early_stopping=False, random_state=None)
#     mlp.fit(X_train_bal, y_train_bal)
#
#     # Retorna o número de épocas usadas para convergência, além de métricas e matriz de confusão
#     return mlp.n_iter_, accuracy_score(y_test, mlp.predict(X_test)), f1_score(y_test, mlp.predict(X_test), average='weighted'), confusion_matrix(y_test, mlp.predict(X_test))
#
# # Executando as simulações e armazenando os resultados
# n_iteracoes = []  # Lista para armazenar o número de épocas para cada simulação
# acuracias = []
# f1_scores = []
# todas_matrizes = []
#
# for _ in range(n_simulacoes):
#     n_iter, acuracia, f1, matriz = rodar_simulacao()
#     n_iteracoes.append(n_iter)  # Armazena o número de épocas para convergência
#     acuracias.append(acuracia)
#     f1_scores.append(f1)
#     todas_matrizes.append(matriz)
#
# # Cálculo da média e desvio padrão das métricas
# media_acuracia = np.mean(acuracias)
# desvio_acuracia = np.std(acuracias)
# media_f1 = np.mean(f1_scores)
#
# # Cálculo da média do número de épocas de convergência
# media_epocas = np.mean(n_iteracoes)
#
# # Exibindo os resultados
# print(f"\nAcurácia Média: {media_acuracia:.4f}")
# print(f"Desvio Padrão da Acurácia: {desvio_acuracia:.4f}")
# print(f"F1-Score Médio: {media_f1:.4f}")
# print(f"Média do Número de Épocas para Convergência: {media_epocas:.0f}")
#
# # Calculando a matriz de confusão média
# matriz_confusao_media = np.mean(todas_matrizes, axis=0).astype(int)
#
#
# # Plotando a matriz de confusão média
# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     matriz_confusao_media,
#     annot=True,
#     fmt='d',
#     cmap='Blues',
#     cbar=False,
#     xticklabels=np.unique(y),
#     yticklabels=np.unique(y)
# )
# plt.xlabel('Previsão')
# plt.ylabel('Real')
# plt.title('Matriz de Confusão Média - MLP-01')
# plt.tight_layout()
#
# # Exibindo a matriz de confusão
# plt.show()


############################################################################################################
############################################################################################################


# (11) - Modelo SVM com métricas de avaliação completas: 24/11/2024 (Final)

#
# # Importando as bibliotecas necessárias
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# from imblearn.over_sampling import SMOTE
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.svm import SVC
#
# # Leia o arquivo Excel
# df = pd.read_excel('Dataset_CVV_v.3.xlsx')
#
# # Remova espaços em branco dos nomes das colunas
# df.columns = df.columns.str.strip()
#
# # Definindo as variáveis preditoras e o rótulo
# X = df[['V1', 'V2', 'V3', 'V1 ang', 'V2 ang', 'V3 ang']]
# y = df['Saida']
#
# # Parâmetro ajustável: Número de simulações
# n_simulacoes = 100
#
# # Listas para armazenar métricas
# acuracias = []
# f1_scores = []
# todas_matrizes = []
#
# # Função para rodar uma simulação do modelo SVM
# def rodar_simulacao_svm():
#     # Dividindo o dataset em treino e teste
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, stratify=y, random_state=None
#     )
#
#     # Balanceando as classes com SMOTE
#     smote = SMOTE(random_state=None)
#     X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
#
#     # Normalizando os dados
#     scaler = StandardScaler()
#     X_train_bal = scaler.fit_transform(X_train_bal)
#     X_test = scaler.transform(X_test)
#
#     # Criando e treinando o modelo SVM
#     svm = SVC(kernel='rbf', C=10, gamma=0.1, random_state=None)
#     svm.fit(X_train_bal, y_train_bal)
#
#     # Fazendo previsões
#     y_pred = svm.predict(X_test)
#
#     # Calculando métricas
#     acuracia = accuracy_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred, average='weighted')
#     matriz = confusion_matrix(y_test, y_pred)
#
#     return acuracia, f1, matriz
#
# # Executando as simulações
# for _ in range(n_simulacoes):
#     acuracia, f1, matriz = rodar_simulacao_svm()
#     acuracias.append(acuracia)
#     f1_scores.append(f1)
#     todas_matrizes.append(matriz)
#
# # Cálculo das métricas médias
# media_acuracia = np.mean(acuracias)
# desvio_acuracia = np.std(acuracias)
# media_f1 = np.mean(f1_scores)
#
# # Calculando a matriz de confusão média
# matriz_confusao_media = np.mean(todas_matrizes, axis=0).astype(int)
#
# # Exibindo os resultados
# print(f"\nAcurácia Média (SVM): {media_acuracia:.4f}")
# print(f"Desvio Padrão da Acurácia (SVM): {desvio_acuracia:.4f}")
# print(f"F1-Score Médio (SVM): {media_f1:.4f}")
#
# # Plotando a matriz de confusão média
# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     matriz_confusao_media,
#     annot=True,
#     fmt='d',
#     cmap='Blues',
#     cbar=False,
#     xticklabels=np.unique(y),
#     yticklabels=np.unique(y)
# )
# plt.xlabel('Previsão')
# plt.ylabel('Real')
# plt.title('Matriz de Confusão Média - SVM-01')
# plt.tight_layout()
# plt.show()


############################################################################################################
############################################################################################################

# (12) - Modelo Random Forest com métricas de avaliação completas: 24/11/2024 (Final)

# # Importando as bibliotecas necessárias
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# from imblearn.over_sampling import SMOTE
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.ensemble import RandomForestClassifier
#
# # Leia o arquivo Excel
# df = pd.read_excel('Dataset_CVV_v.3.xlsx')
#
# # Remova espaços em branco dos nomes das colunas
# df.columns = df.columns.str.strip()
#
# # Definindo as variáveis preditoras e o rótulo
# X = df[['V1', 'V2', 'V3', 'V1 ang', 'V2 ang', 'V3 ang']]
# y = df['Saida']
#
# # Parâmetro ajustável: Número de simulações
# n_simulacoes = 100
#
# # Listas para armazenar métricas
# acuracias = []
# f1_scores = []
# todas_matrizes = []
#
# # Função para rodar uma simulação do modelo Random Forest
# def rodar_simulacao_rf():
#     # Dividindo o dataset em treino e teste
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, stratify=y, random_state=None
#     )
#
#     # Balanceando as classes com SMOTE
#     smote = SMOTE(random_state=None)
#     X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
#
#     # Normalizando os dados
#     scaler = StandardScaler()
#     X_train_bal = scaler.fit_transform(X_train_bal)
#     X_test = scaler.transform(X_test)
#
#     # Criando e treinando o modelo Random Forest
#     rf = RandomForestClassifier(
#         n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=None
#     )
#     rf.fit(X_train_bal, y_train_bal)
#
#     # Fazendo previsões
#     y_pred = rf.predict(X_test)
#
#     # Calculando métricas
#     acuracia = accuracy_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred, average='weighted')
#     matriz = confusion_matrix(y_test, y_pred)
#
#     return acuracia, f1, matriz
#
# # Executando as simulações
# for _ in range(n_simulacoes):
#     acuracia, f1, matriz = rodar_simulacao_rf()
#     acuracias.append(acuracia)
#     f1_scores.append(f1)
#     todas_matrizes.append(matriz)
#
# # Cálculo das métricas médias
# media_acuracia = np.mean(acuracias)
# desvio_acuracia = np.std(acuracias)
# media_f1 = np.mean(f1_scores)
#
# # Calculando a matriz de confusão média
# matriz_confusao_media = np.mean(todas_matrizes, axis=0).astype(int)
#
# # Exibindo os resultados
# print(f"\nAcurácia Média (RF): {media_acuracia:.4f}")
# print(f"Desvio Padrão da Acurácia (RF): {desvio_acuracia:.4f}")
# print(f"F1-Score Médio (RF): {media_f1:.4f}")
#
# # Plotando a matriz de confusão média
# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     matriz_confusao_media,
#     annot=True,
#     fmt='d',
#     cmap='Blues',
#     cbar=False,
#     xticklabels=np.unique(y),
#     yticklabels=np.unique(y)
# )
# plt.xlabel('Previsão')
# plt.ylabel('Real')
# plt.title('Matriz de Confusão Média - RF-01')
# plt.tight_layout()
# plt.show()