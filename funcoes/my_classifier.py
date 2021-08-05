import pandas as pd
import numpy as np
from scipy.stats import norm
from math import ceil
import time
from tqdm import tqdm_notebook as tqdm 

import matplotlib.pyplot as plt
import seaborn as sns
from my_plot import labs, central_trend

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics

class Classifier:
    ''''
    Classe que recebe um modelo de machine learning, os dados que serão realizados os treinamentos e previsões, com isso embaralha esses dados e separa em um conjunto de dados para
    as variáveis preditoras(x) e a variável de resposta.Realiza uma validação cruzada através da função cross_val() e imprime seus resultados pela função report(), plota a curva roc
    pela função plot_roc_curve() e plota os histogramas das métricas calculadas pela função hist_metrics().
    
    Parâmetros Construtor:
    ---------------------
    estimator : modelo de machine learning com que possua um método fit(), predict() e predict_proba()
    df : DataFrame com os dados a serem previstos, tipo : pd.DataFrame
    SEED : número que define a semente aleatória utilizada em todos os métodos, tipo : int, padrão : 64541
    target_variable : nome da variável de resposta(alvo), tipo : str, padrão : 'ICU'
    
    Atributos definidos no construtor:
    ---------
    seed : número inteiro que define a aleatoriedade
    x : conjunto de dados sem a váriavel alvo
    y : conjunto de dados com as respostas da variável alvo
    estimator : modelo de machine learning
    
    Métodos:
    --------
        cross_val() : Realiza uma validação cruzada do tipo RepeatedStratifiedKFold e salva os resultados das métricas passadas no parâmetro scoring
        
            Parâmetros:
            ----------
            scoring : lista os nomes das métricas que serão calculadas na validação cruzada, essa lista aceita as seguintes métricas {'f1','roc_auc','precision','recall','accuracy'},
                      tipo : list, padrão : ['f1', 'roc_auc', 'precision', 'recall', 'accuracy'],
            n_splits : número inteiro indicando a quantidade de divisões realizadas no dataset na hora da validação cruzada, padrão : 5
            n_repeats : número inteiro indicando a quantidade de repetições realizadas da validação cruzada, tipo : int, padrão : 10
            report : valor booleano indicando a chamada da função report que imprime os resultados, tipo : bool, padrão : True

            Atributos definidos:
            --------------------
            n_splits: número inteiro indicando a quantidade de divisões realizadas no dataset na hora da validação cruzada
            n_repeats : número inteiro indicando a quantidade de repetições realizadas da validação cruzada
            len : número inteiro indicando a quantidade total de treinamentos realizados, igual a n_splits * n_repeats
            scores : dicionário com os resultados de todos os treinamentos realizados na validação cruzada
            means : dicionário com as médias de todas as métricas calculadas na validação cruzada
            stds : dicionário com os desvios padrões amostrais de todas as métricas calculadas na validação cruzada
            time_mean : lista com os tempos médios de treinamento
            tprs_mean : lista com as médias dos valores verdadeiros positivos da curva ROC
            tprs_std : lista com os desvios padrões dos valores verdadeiros positivos da curva ROC
            fpr_mean : lista com os valores falsos positivos da curva ROC
            
            
        
        report():
            Imprime os resultados da validação cruzada realizada na função cross_val()
            
        plot_roc_curve():
            
            Parâmetros:
            -----------
            ax : eixo a ser plotado o gráfico, se nenhum for passado será criado automaticamnete, tipo : matplotlib.axes, padrão : None
            name_estimator: Nome do modelo que aparece na legenda do gráfico, se nenhum for passado será imprimido como a função foi passada, tipo : str, padrão : None
            **kwargs_lineplot : argumentos adicionais a serem passados para função lineplot do seaborn
            
            Retorno:
            --------
            ax : eixo que o gráfico foi plotado, tipo : matplotlib.axes
        
        hist_metrics():
            
            Parâmetros:
            -----------
            ax : eixo a ser plotado o gráfico, se nenhum for passado será criado automaticamnete, tipo : matplotlib.axes, padrão : None
            central : booleano indicando se é para plotar a média e a mediana das métricas no histograma, tipo : bool, padrão : True, 
            bins : quantidade de barras do histograma, tipo : int, padrão : 5
            **kwargs_histplot : argumentos adicionais a serem passados para função lineplot do seaborn
            
            Retorno:
            --------
            ax : eixo que o gráfico foi plotado, tipo : matplotlib.axes
        
            
    '''
    def __init__(self,
                 estimator,
                 df:pd.DataFrame,
                 SEED:int = 64541,
                 target_variable='ICU',
                  **estimator_args):
    
        self.seed = SEED
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        self.x = df.drop('ICU', axis=1)
        self.y = df['ICU']
        if callable(estimator):
            np.random.seed(self.seed)
            self.estimator = estimator(**estimator_args)
        else:
            self.estimator = estimator     
    
    def cross_val(self, 
                  scoring:list = ['f1', 'roc_auc', 'precision', 'recall', 'accuracy'],
                  n_splits = 5,
                  n_repeats = 10,
                  report=True):
        '''
        Parâmetros:
        ----------
        scoring : lista os nomes das métricas que serão calculadas na validação cruzada, essa lista aceita as seguintes métricas {'f1','roc_auc','precision','recall','accuracy'},
        tipo : list, padrão : ['f1', 'roc_auc', 'precision', 'recall', 'accuracy'],
        n_splits : número inteiro indicando a quantidade de divisões realizadas no dataset na hora da validação cruzada, padrão : 5
        n_repeats : número inteiro indicando a quantidade de repetições realizadas da validação cruzada, tipo : int, padrão : 10
        report : valor booleano indicando a chamada da função report que imprime os resultados, tipo : bool, padrão : True
        '''
        
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.len = n_splits * n_repeats
        self.scores = {metric:[] for metric in ['f1', 'roc_auc', 'precision', 'recall', 'accuracy']}

        self.fpr_mean = np.linspace(0, 1, 100)
        time_list = []
        tprs = []

        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=self.seed)

        for _, (train, test) in zip(tqdm(range(self.len)),cv.split(self.x, self.y)):
            start = time.time()
            self.estimator.fit(self.x.iloc[train,:], self.y.iloc[train])
            end = time.time()
            time_list.append(end-start)
            pred_proba = self.estimator.predict_proba(self.x.iloc[test,:])
            pred = self.estimator.predict(self.x.iloc[test,:])
            kwargs_func_scores = {'roc_auc': {'y_score':pred_proba[:,1]}, 'accuracy': {'y_pred':pred}, 'f1': {'y_pred':pred, 'average':'macro'},
                                     'precision': {'y_pred':pred, 'average':'macro'},'recall': {'y_pred':pred, 'average':'macro'}}

            for metric in scoring:
                score = getattr(metrics, f'{metric}_score')(self.y.iloc[test], **kwargs_func_scores[metric])  
                self.scores[metric].append(score)

            fpr, tpr, _ = metrics.roc_curve(self.y.iloc[test], pred_proba[:,1])
            interp_tpr = np.interp(self.fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        self.means = dict(map(lambda kv: (kv[0], np.mean(kv[1],axis=0)), self.scores.items()))
        self.stds = dict(map(lambda kv: (kv[0], np.std(kv[1],ddof=1)), self.scores.items()))
        self.time_mean = np.mean(time_list,axis=0)
        self.tprs_mean = np.mean(tprs, axis=0)
        self.tprs_std = np.std(tprs, axis=0, ddof=1)

        if report:
            self.report()

    def report(self):
        '''
        Imprime os resultados da validação cruzada realizada na função cross_val()
        '''
        More_less = u"\u00B1"
        print(f'{self.n_repeats} repetições de Validação Cruzada com {self.n_splits} divisões no dataset')
        print(f'----------------------------------------------------------------------------------')
        print(f'CLASSIFICADOR                           : {self.estimator}')
        print(f'----------------------------------------------------------------------------------')
        print(f'Métricas no dataset de teste:             ')
        print(f'Intervalo de 95% da média')
        print(f'----------------------------------------------------------------------------------')
        print(r'ROC AUC MÉDIA      : %0.3f %s %0.3f     ' %\
             (np.round(self.means['roc_auc'],3), More_less, np.round(norm.ppf(0.975) * self.stds['roc_auc'] / np.sqrt(self.len),3)))
        print(f'----------------------------------------------------------------------------------')
        print(f'ACCURACY  MÉDIA    : %0.3f %s %0.3f ' %\
             (np.round(self.means['accuracy'],3), More_less, np.round(norm.ppf(0.975) * self.stds['accuracy'] / np.sqrt(self.len),3)))

        print(f'PRECISÃO  MÉDIA    : %0.3f %s %0.3f '%\
                  (np.round(self.means['precision'],3), More_less, np.round(norm.ppf(0.975) * self.stds['precision'] / np.sqrt(self.len),3)))
        print(f'RECALL MÉDIO       : %0.3f %s %0.3f '%\
                  (np.round(self.means['recall'],3), More_less, np.round(norm.ppf(0.975) * self.stds['recall'] / np.sqrt(self.len),3)))
        print(f'F1-SCORE  MÉDIO    : %0.3f %s %0.3f'%\
                  (np.round(self.means['f1'],3), More_less, np.round(norm.ppf(0.975) * self.stds['f1'] / np.sqrt(self.len),3)))
        print(f'\nTEMPO MÉDIO DE TREINAMENTO:{np.round(self.time_mean,3)}')   


    def hist_metrics(self, 
                    ax=None,  
                    central:bool=True, 
                    bins:int=5,
                    **kwargs_histplot):
        '''
        Parâmetros:
        -----------
        ax : eixo a ser plotado o gráfico, se nenhum for passado será criado automaticamnete, tipo : matplotlib.axes, padrão : None
        central : booleano indicando se é para plotar a média e a mediana das métricas no histograma, tipo : bool, padrão : True, 
        bins : quantidade de barras do histograma, tipo : int, padrão : 5
        **kwargs_histplot : argumentos adicionais a serem passados para função lineplot do seaborn
        '''
        if ax == None:
            fig, ax = plt.subplots(ceil(len(self.scores)/2),2, figsize = (20, int(8*len(self.scores)/2)))
        i=0
        j=0
        for k, v in self.scores.items():
            plt.sca(ax[i,j])
            sns.histplot(v, bins=bins, **kwargs_histplot)
            labs(title=k.upper() + ' SCORE',xlabel='Valor',ylabel='Frequência', ax=ax[i,j])
            plt.ylim([0,1])
            if central:
                central_trend(v, ax[i,j])
            if j == 1:
                j = 0
                i += 1
            else:
                j+=1
        if j == 1:
            plt.delaxes(ax= ax[-1][1])
            pos1 = ax[-1][0].get_position() 
            ax[-1][0].set_position([pos1.x0 + 0.2, pos1.y0, pos1.width, pos1.height])

        ax[0,0].text(0,1.18,f'Distribuição dos desempenhos do modelo {self.estimator}', fontsize=25, transform=ax[0,0].transAxes)
        ax[0,0].text(0,1.12,'MÉTRICAS OBTIDAS PELO MÉTODO REPEATEDSTRATIFIEDKFOLD', fontsize=15, transform=ax[0,0].transAxes, color='gray')

        return ax

    def plot_roc_curve(self, 
                      ax=None, 
                      name_estimator:str=None,
                      **kwargs_lineplot):

        '''
        Parâmetros:
        -----------
        ax : eixo a ser plotado o gráfico, se nenhum for passado será criado automaticamnete, tipo : matplotlib.axes, padrão : None
        name_estimator: Nome do modelo que aparece na legenda do gráfico, se nenhum for passado será imprimido como a função foi passada, tipo : str, padrão : None
        **kwargs_lineplot : argumentos adicionais a serem passados para função lineplot do seaborn
        '''

        if ax == None:
            fig,ax = plt.subplots(figsize=(20,10))
        if name_estimator == None:
            name_estimator = str(self.estimator)

        sns.lineplot(self.fpr_mean, self.tprs_mean, ax=ax, label= r'ROC CURVE (AUC MEAN = %0.3f $\pm$ %0.3f)' % \
                         (np.round(self.means['roc_auc'],3), np.round(norm.ppf(0.975) * self.stds['roc_auc'] / np.sqrt(self.len),3)) + f'{name_estimator}', estimator=None, **kwargs_lineplot)

        plt.sca(ax)
        tprs_upper = np.minimum(self.tprs_mean + norm.ppf(0.975) * self.tprs_std / np.sqrt(self.len), 1)
        tprs_lower = np.maximum(self.tprs_mean - norm.ppf(0.975) * self.tprs_std / np.sqrt(self.len), 0)
        plt.fill_between(self.fpr_mean, tprs_lower, tprs_upper, color=ax.lines[-1].get_color(), alpha=0.05)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        labs(title = 'ROC CURVE', xlabel='Taxa de Falsos Positivos', ylabel='Taxa de Verdadeiros Positivos', ax=ax, \
             subtitle='ROC CURVE COMPUTADA PELAS VALORES DE VERDADEIROS POSITIVOS E FALSOS POSITIVOS OBTIDOS PELA VALIDAÇÃO CRUZADA')
        plt.legend(loc='lower right', fontsize=12)

        return ax