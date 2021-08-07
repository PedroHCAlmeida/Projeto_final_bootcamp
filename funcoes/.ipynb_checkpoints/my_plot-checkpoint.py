import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import seaborn as sns
import pandas as pd
import numpy as np


def labs(ax, title:str='', subtitle:str='', xlabel:str='', ylabel:str=''):
    '''
    Função que plota as informações adicionais dos gráficos, título, subtítulo, rótulos, labels, fontes
    
    Parâmetros:
    ----------
    ax : eixo a ser plotado o gráfico, se nenhum for passado será criado automaticamnete, tipo : matplotlib.axes
    title : título do gráfico, tipo : str, padrão : ''
    subtitle : subtítulo do gráfico, tipo : str, padrão : ''
    xlabel : rótulo do eixo x, tipo : str, padrão : ''
    ylabel : rótulo do eixo y, tipo : str, padrão : ''
    '''
    #Seta o eixo passado
    plt.sca(ax)
    #Se o subtítulo for passado adiciona uma nova linha
    if subtitle != '':
        title = title + '\n' 
    #Plota o título
    plt.title(title, fontsize=25, loc='left')
    #Plota o subtítulo
    plt.text(0,1.02, subtitle, color='gray', transform=ax.transAxes, fontsize=15)
    #Plota os rótulos dos eixos
    plt.xlabel(xlabel, color='#333333', fontsize=15)
    plt.ylabel(ylabel, color='#333333', fontsize=15)
    #Muda a fonte e cor dos eixos
    plt.yticks(fontsize=15, color='#333333')
    plt.xticks(fontsize=15, color='#333333')

def central_trend(data, ax, axis:str='x', colors=None):
    '''
    Recebe um conjunto de dados calcula a média e a mediana e plota linhas verticais(axis="x") ou horizontais(axis="y")
    
    Parâmetros:
    -----------
    data : conjunto de dados, qualquer tipo aceito pela função np.mean, np.median, exemplos : list, np.array, pd.Series ... 
    ax : eixo do matplotlib onde será plotada as linhas, tipo : matplotlib.Axes
    axis : eixo indicando a horientação das linhas, tipo : str, padrão : 'x', 'x' indica linhas verticais e 'y' horinzontais, tipo : matplotlib.axes
    colors : dicionário da forma {'Mean':cor, 'Median':cor}, se não for passado a média será na cor vermelha e a mediana na cor azul, tipo : dict, padrão : None 
    
    Retorno:
    --------
    Retorna o eixo do matplotlib passado nos parâmetros
    '''
    #Checa se as cores foram especificadas
    if colors==None:
        #Defini as cores das linhas
        colors = {'Mean':'red', 'Median':'darkblue'}
        
    #Calcula a média
    mean = np.mean(data, axis=0)
    
    #Calcula a mediana
    median = np.median(data, axis=0)
    
    #Seta o eixo passado
    plt.sca(ax)
    
    #Checa a orientação
    if axis == 'x':
        
        #Plota as linhas
        plt.axvline(mean, alpha=0.7, linestyle='--', color=colors['Mean'])
        plt.axvline(median, alpha=0.7, linestyle='--', color=colors['Median'])
        
        #Plota as legendas
        plt.text(mean, 0.90,'Média' + str(mean), fontsize=15, transform = transforms.blended_transform_factory(ax.transData, ax.transAxes), color=colors['Mean'])
        plt.text(median, 0.85,'Mediana' + str(median), fontsize=15, transform = transforms.blended_transform_factory(ax.transData, ax.transAxes), color=colors['Median'])
    
    #Checa a orientação
    elif axis == 'y':
        
        #Plota as linhas
        plt.axhline(mean, alpha=0.7, linestyle='--', color=colors['Mean'])
        plt.axhline(median, alpha=0.7, linestyle='--', color=colors['Median'])
        
        #Plota as legendas
        plt.text(0.9, mean, 'Média ' + str(mean) , fontsize=15, transform = transforms.blended_transform_factory(ax.transAxes, ax.transData), color=colors['Mean'])
        plt.text(0.85, median,'Mediana ' + str(median), fontsize=15, transform = transforms.blended_transform_factory(ax.transAxes, ax.transData), color=colors['Median'])
    
    #Retona o eixo
    return ax

def annot_bar(ax, prop=True):
    '''
    Realiza anotações em cima das barras em um gráfico de barras 
    
    Parâmetros:
    -----------
    ax : ax : eixo do matplotlib onde será plotada as linhas, tipo : matplotlib.axes
    prop : indicando se os valores serão representados como porcentagens ou não, tipo : bool, padrão : True
    
    OBS : precisa ser chamada após o gráfico de barras
    '''
    #Salva as informações das barras que estão no gráfico
    rects = ax.patches
    
    #Verifica se é do tipo porcentagem
    if prop:
        str_symbol = '%'
        mult_value = 100
    else:
        str_symbol = ''
        mult_value = 1
    
    #Anota no gráfico em cima de cada barra os valores 
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, str(round(height * mult_value,2)) + str_symbol, color='black', ha='center', va='bottom', fontsize=15)