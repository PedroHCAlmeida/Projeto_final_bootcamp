from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW
from scipy.stats import levene
from sklearn.feature_selection import chi2 as sklearn_chi2
from itertools import chain
import pandas as pd
import numpy as np


class Drop_features_files(BaseEstimator, TransformerMixin):
    '''
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
    '''
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

def features_high_corr(data, write=True, threshold=0.95, target_variable='ICU', direc='../dados/cols_drop_txt/', file='high_corr'):

    data_float = data.select_dtypes('float64')
        
    matrix_corr = data_float.corr().abs()
        
    matrix_corr_upper = matrix_corr.where(np.triu(np.ones(matrix_corr.shape), k=1).astype(np.bool))
  
    #Selecionando as colunas que estão apresentam uma correlação maior que o valor de corte
    cols_drop = [col for col in matrix_corr_upper.columns if any(matrix_corr_upper[col] >= threshold)]
    if write:
        with open(direc + file + f'_thd_{threshold}' + '.txt', 'w') as f:
            for feature in cols_drop:
                f.write(feature+'\n')
    return cols_drop
    
def features_equal_var_mean(data, write=True, alpha_mean=0.05, alpha_var=0.05,target_variable='ICU', direc='../dados/cols_drop_txt/', file='equal_mean'):
        
        #Criando uma lista vazia para armazenar as colunas onde as médias são consideradas iguais com uma significância igual ao p_valor passado no __init__
        equal = []
        
        #Separando os dois grupos pela feature dependente
        group_1 = data[data[target_variable] == 1] 
        group_2 = data[data[target_variable] == 0]
        
        #Separando as colunas contínuas
        cols_float = data.select_dtypes('float64').columns
        
        #Criando um for pelas colunas contínuas
        for column in cols_float:
            
            #Desconsiderando as colunas binárias
            if len(data[column].unique()) == 2:
                pass
            else:
                p_value_var = levene(group_1[column], group_2[column])[1]
                
                if p_value_var > alpha_var or np.isnan(p_value_var):
                    
                    #Criando os teste do grupo 1 
                    group_1_test = DescrStatsW(group_1[column])
                
                    #Criando os teste do grupo 2
                    group_2_test = DescrStatsW(group_2[column])
                
                    #Criando o teste de comparação das médias
                    test = CompareMeans(group_1_test, group_2_test)
                
                    #Obtendo o p_valor da Hipótese nula igual as médias serem iguais
                    p_value_mean = test.ztest_ind()[1]
                
                    #Verificando se o p_valor é maior que a significância para não rejeitar a hipótese nula
                    if p_value_mean > alpha_mean or np.isnan(p_value_mean):
                        equal.append(column)
        if write:
            with open(direc + file + f'_alpha_mean_{alpha_mean}_var_{alpha_var}' + '.txt', 'w') as f:
                for feature in equal:
                    f.write(feature+'\n')
        return equal
    
def features_equal_mean(data, write=True, alpha=0.05, target_variable='ICU', direc='../dados/cols_drop_txt/', file='equal_mean'):
        
        #Criando uma lista vazia para armazenar as colunas onde as médias são consideradas iguais com uma significância igual ao p_valor passado no __init__
        equal = []
        
        #Separando os dois grupos pela feature dependente
        group_1 = data[data[target_variable] == 1] 
        group_2 = data[data[target_variable] == 0]
        
        #Separando as colunas contínuas
        cols_float = data.select_dtypes('float64').columns
        
        #Criando um for pelas colunas contínuas
        for column in cols_float:
            
            #Desconsiderando as colunas binárias
            if len(data[column].unique()) == 2:
                pass
            else:
                #Criando os teste do grupo 1 
                group_1_test = DescrStatsW(group_1[column])
                
                #Criando os teste do grupo 2
                group_2_test = DescrStatsW(group_2[column])
                
                #Criando o teste de comparação das médias
                test = CompareMeans(group_1_test, group_2_test)
                
                #Obtendo o p_valor da Hipótese nula igual as médias serem iguais
                p_value = test.ztest_ind()[1]
                
                #Verificando se o p_valor é maior que a significância para não rejeitar a hipótese nula
                if p_value > alpha or np.isnan(p_value):
                    equal.append(column)
        if write:
            with open(direc + file + f'_alpha_{alpha}' + '.txt', 'w') as f:
                for feature in equal:
                    f.write(feature+'\n')
        return equal
    
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