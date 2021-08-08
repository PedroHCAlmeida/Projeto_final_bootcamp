from numpy import any as numpy_any

def fill_table(rows):
    '''
    Função que recebe as linhas correspondentes a apenas um paciente e preenche os 
    valores nulos das colunas contínuas com dados de outras janelas.
    Utiliza os métodos "ffill" e "bfill"
    
    Parâmetros:
    -----------
    rows : as linhas correspondentes ao mesmo paciente
    
    Retorno:
    --------
    Retorna as linhas preenchidas
    '''
    #Aplicando os métodos de 'bfill' e 'ffill' nas colunas contínuas 
    rows[rows.select_dtypes('float64').columns] = rows[rows.select_dtypes('float64').columns].fillna(method='bfill').fillna(method='ffill')
    return rows

def select_window(rows, window='0-2', target_variable='ICU'):
    '''
    Função que seleciona a janela correspondente e verifica se em algum momento o paciente foi para UTI, 
    com isso modifica a coluna dependente para se em qualquer janela de tempo esse paciente foi para UTI
    
    Parâmetros:
    -----------
    rows : as linhas correspondentes ao mesmo paciente
    window : coluna correspondente à janela que será mantida, padrão : '0-2'
    target_variable : nome da variável dependente, padrão : 'ICU'
    
    Retono:
    -------
    Retorna o conjunto de dados preenchido
    '''
    #Verificando se o paciente foi para UTI em qualquer momento
    if(numpy_any(rows[target_variable])):
        rows.loc[rows["WINDOW"]==window, target_variable] = 1
    #Retornando apenas os dados da janela especificada
    return rows.loc[rows["WINDOW"] == window]