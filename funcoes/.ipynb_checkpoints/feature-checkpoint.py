from sklearn.feature_selection import chi2
import pandas as pd
import numpy as np

def binary_features(data:pd.DataFrame):
    '''
    Função que recebe um DataFrame do pandas e verifica quais as colunas binárias.
    Realiza essa verificação consultando quantos valores únicos a coluna possui,
    qual o valor máximo e mínimo.
    OBS : os valores únicos precisam ser 1 e 0.
    
    Parâmetros:
    -----------
    data : DataFrame do pandas que será conferida quais as colunas binárias, tipo : pd.DataFrame
    
    Retorno:
    --------
    features : colunas binárias do dataframe, tipo : list
    '''
    #Criando a lista vazia para armazenar as colunas binarias
    features = []
    
    #Percorrendo as colunas do dataframe
    for feature in data.columns:
        
        #Verificando se a coluna possui apenas 2 valores únicos
        if len(data[feature].unique()) == 2:
            
            if max(data[feature])==1 and min(data[feature])==0:
                #Salvando o nome da coluna na lista
                features.append(feature)
    #Retorna as colunas           
    return features

def compute_chi2(X:pd.DataFrame, y:pd.Series):
    '''
    Recebe um dataframe do pandas com as colunas variáveis preditoras e uma Series do pandas da variável de reposta,
    calcula, através da função sklearn.model_selection.chi2, os p valores para cada coluna da hipótese nula 
    de que as variáveis preditoras são independentes da variável de reposta
    
    Parâmetros:
    -----------
    X : dataframe com as variáveis preditoras,tipo : pd.DataFrame,
        OBS: pode conter apenas uma coluna mas precisa ser do tipo pd.DataFrame
    y : Series do pandas da variável de resposta, tipo : pd.Series
    
    Retorno:
    -------
    p_values : dicionário contendo os p valores de cada coluna analisada, tipo : dict
    '''
    #Criando o dicionário vazio para armazenar os p valores
    p_values = {}
        
    #Calulando as estatísticas de teste e p valores através da função sklearn.chi2
    chi2_results = chi2(X, y)
    
    #Criando o dicionário com os p_valores
    for i,col in enumerate(X):
        p_values[col] = chi2_results[1][i]
        
    #Retorna os p valores
    return p_values

def compute_high_corr(data:pd.DataFrame, threshold:float=0.95):
    '''
    Função que recebe um dataframe do pandas, calcula a matriz de correlação, seleciona o triângulo superior da matriz, 
    percorre as colunas verificando quais colunas apresentam uma correlação maior que o valor limite passado, em relação às linhas, 
    que nesse caso representam as colunas anteriores. 
    
    Parâmetros:
    -----------
    data : DataFrame do pandas com as variáveis a serem analisadas
    threshold : valor limite da correlação entre duas variáveis, ou seja, 
                se duas colunas que possuem uma correlação absoluta maior que esse valor uma das duas será eliminada,
                tipo : str, padrão : 0.95
    Retorno:
    --------
    cols_drop : colunas a serem eliminadas com base nas correlações, tipo : list
    
    '''
    #Calculando a matriz de correlação absoluta
    matrix_corr = data.corr().abs()
    
    #Filtrando apenas a matriz superior
    matrix_corr_upper = matrix_corr.where(np.triu(np.ones(matrix_corr.shape), k=1).astype(np.bool))
  
    #Selecionando as colunas que estão apresentam uma correlação maior que o valor de corte
    cols_drop = [col for col in matrix_corr_upper.columns if any(matrix_corr_upper[col] >= threshold)]
    
    #Retorna as colunas a serem eliminadas
    return cols_drop