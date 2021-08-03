from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import chi2 as sklearn_chi2, SelectKBest

from statsmodels.stats.weightstats import CompareMeans, DescrStatsW
from scipy.stats import levene

import pandas as pd
import numpy as np

def binary_features(data:pd.DataFrame):
    '''
    Função que recebe um DataFrame do pandas e verifica quais as colunas binárias.Realiza essa verificação consultando quantos valores únicos a coluna possui
    e tenta transformar-la em números inteiros
    
    Parâmetros:
    -----------
    data : DataFrame do pandas que será 
    
    Retorno:
    --------
    features : colunas binárias do dataframe
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
                
    return features


class chi2_drop(BaseEstimator, TransformerMixin):
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.cols_drop = []
            
    def fit(self, X ,y):
        
        cols_binary = binary_features(X)
        p_values = compute_chi2(X[cols_binary], y)
        for col in cols_binary:
            if p_values[col] > self.alpha or np.isnan(p_values[col]):
                self.cols_drop.append(col)
        return self
                
    def transform(self, X, y=None):
        X = X.drop(self.cols_drop, axis=1, errors='ignore')
        return X

def compute_chi2(X, y):
    
    p_values = {}
        
    #Criando uma lista vazia para armazenar as colunas onde as médias são consideradas iguais com uma significância igual ao p_valor passado no __init__
    chi2 = sklearn_chi2(X, y)
        
    for i,col in enumerate(X):
        p_values[col] = chi2[1][i]
    return p_values

class high_corr_drop(BaseEstimator, TransformerMixin):
    '''
    Classe personalizada do tipo TransformerMixin do sklearn. Tem o objetivo de eliminar todas as colunas que possuem uma alta correlação com colunas anteriores.
    
    Construtor:
    -----------
        Parâmetros:
        -----------
        threshold : valor limite da correlação entre duas variáveis, ou seja, se duas colunas que possuem uma correlação absoluta maior que esse valor uma das duas será eliminada,
                    tipo : str, padrão : 0.95
    
    Métodos:
    --------
        fit(): Calcula as colunas a serem eliminadas com base no threshold passado no construtor e salva em um aributo cols_high_corr
        ------
            Parâmetros:
            -----------
            X : variáveis independentes, tipo : pd.DataFrame
            y : variável dependente/alvo, tipo : pd.DataFrame, padrão : None
            
            Retorno:
            --------
            self : retorna o próprio objeto
        
        transform(): Recebe o dataframe com as variáveis dependentes e elimina as colunas calculadas no método fit()
        ------------
            Parâmetros:
            -----------
            X : variáveis independentes, tipo : pd.DataFrame
            y : variável dependente/alvo, tipo : pd.DataFrame, padrão : None
            
            Retorno:
            --------
            X : Retorna o dataframe com as variáveis dependentes após eliminar as colunas calculadas no método fit()
    Atributos:
    ----------
    thd : threshold definido no construtor
    cols_high_corr : colunas a eliminar
    '''
    def __init__(self, threshold=0.95):
        #
        self.thd = threshold
    
    def fit(self, X, y=None):
        self.cols_high_corr = compute_high_corr(data=X, threshold=self.thd)
        return self
    
    def transform(self, X, y=None):
        return X.drop(self.cols_high_corr, axis=1)
        
def compute_high_corr(data:pd.DataFrame, threshold:float=0.95):
    '''
    Função que recebe um dataframe do pandas, calcula a matriz de correlação, seleciona o triângulo superior da matriz e percorre as colunas verificando quais colunas apresentam
    uma correlação maior que o valor limite passado, em relação às linhas, que nesse caso representam as colunas anteriores. 
    
    Parâmetros:
    -----------
    data : DataFrame do pandas com as variáveis a serem analisadas
    threshold : valor limite da correlação entre duas variáveis, ou seja, se duas colunas que possuem uma correlação absoluta maior que esse valor uma das duas será eliminada,
                    tipo : str, padrão : 0.95
    
    '''
    #Calculando a matriz de correlação absoluta
    matrix_corr = data.corr().abs()
    
    #Filtrando apenas a matriz superior
    matrix_corr_upper = matrix_corr.where(np.triu(np.ones(matrix_corr.shape), k=1).astype(np.bool))
  
    #Selecionando as colunas que estão apresentam uma correlação maior que o valor de corte
    cols_drop = [col for col in matrix_corr_upper.columns if any(matrix_corr_upper[col] >= threshold)]
    
    return cols_drop

class drop_equal_var_mean(BaseEstimator, TransformerMixin):
    
    def __init__(self, alpha_var=0.05, alpha_mean=0.05):
        self.alpha_var = alpha_var
        self.alpha_mean = alpha_mean
        self.cols_drop = []
    
    def fit(self, X, y):
        p_values = compute_equal_var_mean(X.select_dtypes('float64'), y)
        for column in X.select_dtypes('float64').columns:
            if p_values['var'][column] > self.alpha_var or np.isnan(p_values['var'][column]):
                if p_values['mean'][column] > self.alpha_mean or np.isnan(p_values['mean'][column]):
                    self.cols_drop.append(column)
        return self
    
    def transform(self, X, y=None):
        return X.drop(self.cols_drop, axis=1)

def compute_equal_var_mean(X, y):
    
    p_values = {'var':{},'mean':{}}
    group0 = X[y == 0]
    group1 = X[y == 1]
    
    for column in X.columns:
        
        _, p_var = levene(group0[column], group1[column])
        p_values['var'][column] = p_var
        
        test0 = DescrStatsW(group0[column])
        test1 = DescrStatsW(group1[column])
        test_mean = CompareMeans(test1, test0)
        _, p_mean = test_mean.ztest_ind()
        p_values['mean'][column] = p_mean

    return p_values
