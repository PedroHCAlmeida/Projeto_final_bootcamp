from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

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
    
    OBS: OS VALORES DE QUANDO O PACIENTE JÁ ESTÁ NA UTI NÃO SÃO UTILIZADOS PARA PREENCHER DADOS ANTERIORES
    '''
    data = data.groupby([group_id, target_variable]).apply(fill_table)
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
    rows_to_drop = data[(data['WINDOW'] == window) & (data[target_variable] == 1)].index
    data.drop(index=rows_to_drop, inplace=True)
    data = data.groupby(group_id).apply(select_window)
    
    return data