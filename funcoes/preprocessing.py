from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class target_encoder(BaseEstimator, TransformerMixin):
    '''
    Classe que transforma dados categóricos em dados numéricos através das proporções encontradas nos dados de treino e pode ser passada em um pipeline do sklearn
    OBS: Usada para problemas de classificação onde a variável dependente é 0 ou 1
    
    Parâmetros do construtor:
    -------------------------
    target_variable : nome da variável dependente a ser prevista no modelo, padrão:'AGE_PERCENTIL'
    encoder_variable : nome da variável categórica a ser transformada, padrão:'ICU'
    
    Atributos do construtor:
    ------------------------
    target_variable : nome da variável dependente a ser prevista no modelo
    encoder_variable : nome da variável categórica a ser transformada
    
    Funções:
    --------
    
        fit():
        ------
        Cria um dicionário associando cada categoria a proporção da mesma achada na variável dependente
        
            Parâmetros:
            ----------
            X : dados com as variáveis independentes
            y : dados com a variável dependente
            
            Atributos:
            ----------
            map_dict : dicionário associando cada valor categórico ao numérico
            
        transform():
        ------------
        Transforma a variável categória mapeando a pelo dicionário estabelecido na função fit()
            
            Parâmetros:
            ----------
            X : dados com as variáveis independentes
            y : dados com a variável dependente, padrão : None
            
            Retorno:
            --------
            Retorna o conjunto de dados das variáveis dependentes(X) com a variável categórica transformada       
    '''
    def __init__(self, encoder_variable='AGE_PERCENTIL', target_variable='ICU'):
        self.target_variable = target_variable
        self.encoder_variable = encoder_variable
        
    def fit(self, X, y):
        data = pd.concat([X,y], axis=1)
        self.map_dict = {}
        for value in data[self.encoder_variable].unique():
            self.map_dict[value] = data[data[self.encoder_variable] == value][self.target_variable].mean()
        return self
    
    def transform(self, X, y=None):
        X[self.encoder_variable] = X[self.encoder_variable].map(self.map_dict)
        return X

def fill_table(rows):
    '''
    Função que recebe as linhas correspondente a apenas um paciente e preenche os valores nulos das colunas contínuas com dados de outras janelas
    
    Parâmetros:
    -----------
    rows : as linhas correspondentes ao mesmo paciente
    
    Retorno:
    --------
    Retorna as linhas preenchidas
    '''
    rows[rows.select_dtypes('float64').columns] = rows[rows.select_dtypes('float64').columns].fillna(method='bfill').fillna(method='ffill')
    return rows
    
def fill_exam(data, group_id='PATIENT_VISIT_IDENTIFIER', target_variable='ICU'): 
    '''
    Função que agrupa os exames por paciente e pela variável independente(ICU), após isso chama a função fill_table para realizar o preenchimento dos valores nulos, dessa forma  
    garantindo que os dados usados para preencher serão somente antes do paciente ter ido para UTI
    
    Parâmetros:
    -----------
    data : conjunto de dados
    group_id : coluna correspondente ao id do paciente, padrão : 'PATIENT_VISIT_IDENTIFIER'
    target_variable : nome da variável dependente, padrão : 'ICU'
    
    Retono:
    -------
    Retorna o conjunto de dados preenchido
    
    OBS: NÃO ESQUECE QUE MUDOU 
    '''
    data = data.groupby([group_id]).apply(fill_table)
    return data

def select_window(rows, window='0-2', target_variable='ICU'):
    '''
    Função que seleciona a janela correspondente e verifica se em algum momento o paciente foi para UTI, com isso modifica a coluna dependente para se em qualquer janela de tempo esse 
    paciente foi para UTI
    
    Parâmetros:
    -----------
    rows : as linhas correspondentes ao mesmo paciente
    window : coluna correspondente à janela que será mantida, padrão : '0-2'
    target_variable : nome da variável dependente, padrão : 'ICU'
    
    Retono:
    -------
    Retorna o conjunto de dados preenchido
    '''
    if(np.any(rows[target_variable])):
        rows.loc[rows["WINDOW"]==window, target_variable] = 1
    return rows.loc[rows["WINDOW"] == window]

def filter_window(data, window='0-2', target_variable='ICU', group_id='PATIENT_VISIT_IDENTIFIER'):
    
    '''
    Função que agrupa os dados por paciente filtra para janela selecionada e verifica se o paciente foi para UTI em algum momento, e com isso modifica o valor da coluna dependente.Além 
    disso elimina os dados dos pacientes que já estavam na UTI na janela selecionada
    
    Parâmetros:
    -----------
    data : conjunto de dados 
    window : coluna correspondente à janela que será mantida, padrão : '0-2'
    target_variable : nome da variável dependente, padrão : 'ICU'
    group_id : coluna correspondente ao id do paciente, padrão : 'PATIENT_VISIT_IDENTIFIER'
    
    Retorno:
    -------
    Retorna o conjunto de dados na janela definida e com a coluna ICU se esse paciente foi para UTI em algum momento
    '''
    def select_window(rows, window=window, target_variable=target_variable):
        '''
        Função que seleciona a janela correspondente e verifica se em algum momento o paciente foi para UTI, com isso modifica a coluna dependente para se em qualquer janela de tempo 
        esse paciente foi para UTI
    
        Parâmetros:
        -----------
        rows : as linhas correspondentes ao mesmo paciente
        window : coluna correspondente à janela que será mantida
        target_variable : nome da variável dependente
    
        Retorno:
        -------
        Retorna o conjunto de dados preenchido
        '''
        if(np.any(rows[target_variable])):
            rows.loc[rows["WINDOW"]==window, target_variable] = 1
        return rows.loc[rows["WINDOW"] == window]
    
    rows_to_drop = data[(data['WINDOW'] == window) & (data[target_variable] == 1)].index
    data.drop(index=rows_to_drop, inplace=True)
    data = data.groupby(group_id).apply(select_window)
    
    return data

