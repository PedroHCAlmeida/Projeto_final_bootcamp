from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import chi2 as sklearn_chi2
import pandas as pd
import numpy as np

def binary_features(data):
    
    #Criando a lista vazia para armazenar as features binarias
    features = []
    
    #Percorrendo as colunas do conjunto de dados
    for feature in data.columns:
        
        #Verificando se a coluna é binária
        if len(data[feature].unique()) == 2:
            
            #Salvando o nome da coluna na lista
            features.append(feature)
    
    return features
        
class chi2_drop(BaseEstimator, TransformerMixin):
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.cols_drop = []
            
    def fit(self, X ,y):
        
        cols_binary = binary_features(X)
        p_values = compute_chi2(X[cols_binary].abs(), y)
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
    
    def __init__(self, threshold=0.95):
        self.thd = threshold
    
    def fit(self, X, y=None):
        self.cols_high_corr = compute_high_corr(data=X, threshold=self.thd)
        return self
    
    def transform(self, X, y=None):
        return X.drop(self.cols_high_corr, axis=1, errors='ignore')
        
def compute_high_corr(data, threshold=0.95):

    data_float = data.select_dtypes('float64')
    matrix_corr = data_float.corr().abs()
    matrix_corr_upper = matrix_corr.where(np.triu(np.ones(matrix_corr.shape), k=1).astype(np.bool))
  
    #Selecionando as colunas que estão apresentam uma correlação maior que o valor de corte
    cols_drop = [col for col in matrix_corr_upper.columns if any(matrix_corr_upper[col] >= threshold)]
    
    return cols_drop

class low_var_drop(BaseEstimator, TransformerMixin):
    
    def __init__(self, threshold=0):
        self.thd = threshold
    
    def fit(self, X, y=None):
        self.cols_low_var = compute_low_var(data=X, threshold=self.thd)
        return self
    
    def transform(self, X, y=None):
        return X.drop(self.cols_low_var, axis=1, errors='ignore')
    
def compute_low_var(data, threshold=0):

    var = data.var()
    cols_drop = [col for col in var.index if var[col] <= threshold]
    return cols_drop

'''
def features_chi2(data, alpha=0.05, write=True, target_variable='ICU', direc='../dados/cols_drop_txt/', file='chi2'):
    
        cols_drop = []
        
        cols_binary = binary_features(data.drop(target_variable, axis=1))
        
        #Criando uma lista vazia para armazenar as colunas onde as médias são consideradas iguais com uma significância igual ao p_valor passado no __init__
        chi2 = sklearn_chi2(data[cols_binary], data[target_variable])
        
        for i,col in zip(range(len(cols_binary)),cols_binary):
            p_value = chi2[1][i]
            if p_value > alpha or np.isnan(p_value):
                cols_drop.append(col)
       
        if write:
            with open(direc + file +  f'_alpha_{alpha}' + '.txt', 'w') as f:
                for feature in cols_drop:
                    f.write(feature+'\n')  
        
        return cols_drop

class Drop_features_files(BaseEstimator, TransformerMixin):
    Classe para ser passada para um pipeline do sklearn que removerá as colunas contidas em determinados arquivos de texto, podem ter colunas repetidas nos arquivos e se a coluna não d
    for encontrada irá ignorar
    
    Parâmetros do construtor:
    -------------------------
    files : lista com os nomes dos arquivos sem a extensão .txt, padrão : []
    direc : diretório onde se encontram os arquivos, padrão : '../dados/cols_drop_txt/'
    
    Atributos do construtor:
    ------------------------
    files : lista com os nomes dos arquivos
    features : lista vazia para armazenar as features a serem removidas
    direc : diretório onde se encontram os arquivos
    
    Funções:
    -------
    def __init__(self, files:list = [], direc='../dados/cols_drop_txt/'):
        self.files = files
        self.features = []
        self.direc = direc

    def fit(self, X=None, y=None):
        
        for file in self.files:
            with open(self.direc + file, 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        self.features.append(line.strip())
        return self
    
    def transform(self, X, y=None):
        X = X.drop(self.features,axis=1,errors='ignore')
        return X
'''