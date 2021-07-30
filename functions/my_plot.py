import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import seaborn as sns
import pandas as pd
import numpy as np


def labs(ax, title:str='', subtitle:str='', xlabel:str='', ylabel:str='',
                 spines_invisible:list=['top', 'right'], kwargs_grid:dict={}):
    '''
    Função que plota as informações adicionais dos gráficos, título, subtítulo, rótulos, labels, fontes
    
    Parâmetros:
    ----------
    ax : eixo a ser plotado o gráfico, se nenhum for passado será criado automaticamnete, tipo : matplotlib.axes
    title : título do gráfico, tipo : str, padrão : ''
    subtitle : subtítulo do gráfico, tipo : str, padrão : ''
    xlabel : rótulo do eixo x, tipo : str, padrão : ''
    ylabel : rótulo do eixo y, tipo : str, padrão : ''
    fonte : fonte dos dados para ser plotada no gráfico(embaixo), tipo : str, padrão : 'https://brasil.io/dataset/covid19/caso_full/'
    spines_invisible : nome dos eixos a serem ocultados, tipo : list, padrão : ['top', 'right']
    kwargs_grid : argumentos a serem passados pra função matplotlib.pyplot.grid, tipo : dict, padrão : {'axis':'y', 'alpha':0.6}
    '''
    
    plt.sca(ax)
    if subtitle != '':
        title = title + '\n' 
    plt.title(title, fontsize=25, loc='left')
    plt.text(0,1.03, subtitle, color='gray', transform=ax.transAxes, fontsize=15)
    plt.xlabel(xlabel, color='#333333', fontsize=15)
    plt.ylabel(ylabel, color='#333333', fontsize=15)
    plt.yticks(fontsize=15, color='#333333')
    plt.xticks(fontsize=15, color='#333333')
    
    for spine in spines_invisible:
        ax.spines[spine].set_visible(False)
    
    plt.grid(**kwargs_grid)

def central_trend(data, ax, colors=None):
    
    if colors==None:
        colors = {'Mean':'red', 'Median':'darkblue'}
    mean = np.mean(data, axis=0)
    median = np.median(data, axis=0)
    plt.sca(ax)
    plt.axvline(mean, alpha=0.7, linestyle='--', color=colors['Mean'])
    plt.axvline(median, alpha=0.7, linestyle='--', color=colors['Median'])
    plt.text(mean, 0.90,'Média', fontsize=15, transform = transforms.blended_transform_factory(ax.transData, ax.transAxes), color=colors['Mean'])
    plt.text(median, 0.85,'Mediana', fontsize=15, transform = transforms.blended_transform_factory(ax.transData, ax.transAxes), color=colors['Median'])
    
    return ax