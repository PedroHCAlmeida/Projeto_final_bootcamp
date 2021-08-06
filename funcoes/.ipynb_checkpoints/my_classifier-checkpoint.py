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
    as variáveis preditoras(x) e a variável de resposta(y).Realiza uma validação cruzada através da função cross_val() e imprime seus resultados pela função report(), plota a curva roc
    pela função plot_roc_curve() e plota os histogramas das métricas calculadas pela função hist_metrics().
    
    Parâmetros Construtor:
    ---------------------
    estimator : modelo de machine learning com que possua um método fit(), predict() e predict_proba()
    df : DataFrame com os dados a serem previstos, tipo : pd.DataFrame
    SEED : número que define a semente aleatória utilizada em todos os métodos, tipo : int, padrão : 64541
    target_variable : nome da variável de resposta(alvo), tipo : str, padrão : 'ICU'
    **estimator_args : parâmetros a serem passados para o modelo se o mesmo ainda não ter sido instanciado
    
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
            confusion_matrix_mean : média das matrizes de confusão geradas em cada FOLD da validação cruzada
            stds : dicionário com os desvios padrões amostrais de todas as métricas calculadas na validação cruzada
            time_mean : lista com os tempos médios de treinamento
            tprs_mean : lista com as médias dos valores verdadeiros positivos da curva ROC
            tprs_std : lista com os desvios padrões amostrais dos valores verdadeiros positivos da curva ROC
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
        
        #Definindo a semente aleatória como atributo
        self.seed = SEED
        
        #Embaralhando o dataset para tentar remover o máximo de aleatoriedade nos resultados 
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        #definindo o x e y como atributos atráves da variável reposta
        self.x = df.drop(target_variable, axis=1)
        self.y = df[target_variable]
        
        #Verificando se a função do modelo já foi instanciado, e se não, instancia com os parâmetros passados
        if callable(estimator):
            
            #Definindo a semente aleatória
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
        
        #Definindo os atributos
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.len = n_splits * n_repeats
        
        #Definindo o dicionário onde serão armazenados os resultados
        self.scores = {metric:[] for metric in ['f1', 'roc_auc', 'precision', 'recall', 'accuracy']}
        
        #Gerando uma matriz de confusão de zeros para realizar a soma
        confusion_matrixs_sum = np.zeros([2,2])
        
        #Gerando as listas que serão usadas para plotar a curva roc 
        self.fpr_mean = np.linspace(0, 1, 100)
        tprs = []
        
        #Iniciando uma lista para registrar os tempos de treinamento
        time_list = []

        #Definindo a validação cruzada
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=self.seed)

        #Gerando os indíces do x e do y para cada FOLD gerado pelo objeto cv pelo método split
        #a função tqdm gera uma barra de progresso no notebook
        for _, (train, test) in zip(tqdm(range(self.len)),cv.split(self.x, self.y)):
            #iniciando o tempo
            start = time.time()
            #Treinando o modelo com os indexes gerados para treino
            self.estimator.fit(self.x.iloc[train,:], self.y.iloc[train])
            #terminando o tempo
            end = time.time()
            #salvando o tempo gasto
            time_list.append(end-start)
            #prevendo a probabilidade de cada classe
            pred_proba = self.estimator.predict_proba(self.x.iloc[test,:])
            #prevendo cada classe
            pred = self.estimator.predict(self.x.iloc[test,:])
            #gerando a matriz de confusão e realizando a soma com as geradas anteriormente
            confusion_matrixs_sum += metrics.confusion_matrix(self.y.iloc[test], pred)
            #definindo os parâmetros para cada função das métricas
            kwargs_func_scores = {'roc_auc': {'y_score':pred_proba[:,1]}, 'accuracy': {'y_pred':pred}, 'f1': {'y_pred':pred, 'average':'macro', 'pos_label':0},
                                     'precision': {'y_pred':pred, 'average':'macro', 'pos_label':0},'recall': {'y_pred':pred, 'average':'macro', 'pos_label':0}}
            #Percorrendo a lista das métricas
            for metric in scoring:
                #Calculando cada métrica pelo módulo metrics do sklearn
                score = getattr(metrics, f'{metric}_score')(self.y.iloc[test], **kwargs_func_scores[metric])  
                #Salvando o resultados no dicionário das métricas
                self.scores[metric].append(score)
                
            #Calculando os valores que serão utilizadas para plotar a curva ROC
            fpr, tpr, _ = metrics.roc_curve(self.y.iloc[test], pred_proba[:,1])
            interp_tpr = np.interp(self.fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
        
        #Calculando a média das matrizes de confusão
        self.confusion_matrix_mean = confusion_matrixs_sum / self.len
        
        #Calculando a média dos resultados para cada métrica
        self.means = dict(map(lambda kv: (kv[0], np.mean(kv[1],axis=0)), self.scores.items()))
        
        #Calculando o desvio padrão amostral dos resultados para cada métrica
        self.stds = dict(map(lambda kv: (kv[0], np.std(kv[1],ddof=1)), self.scores.items()))
        
        #Calculando o tempo médio
        self.time_mean = np.mean(time_list,axis=0)
        
        #Calculando a média e desvio padrão dos tprs que serão utilizados para plotar a curva ROC
        self.tprs_mean = np.mean(tprs, axis=0)
        self.tprs_std = np.std(tprs, axis=0, ddof=1)
        
        #Chamando a função report se o parâmetro report for True
        if report:
            self.report()

    def report(self):
        '''
        Imprime os resultados da validação cruzada realizada na função cross_val()
        '''
        #Gerando os valores de verdadeiros e falsos negativos e positivos através da média das matrizes de confusão
        tp = self.confusion_matrix_mean[1][1]
        fp = self.confusion_matrix_mean[0][1]
        fn = self.confusion_matrix_mean[1][0]
        tn = self.confusion_matrix_mean[0][0]
        
        #Definindo a string que indica o sinal de mais ou menos
        More_less = u"\u00B1"
        
        #Imprimindo as métricas
        print(f'{self.n_repeats} repetições de Validação Cruzada com {self.n_splits} divisões no dataset')
        print(f'----------------------------------------------------------------------------------')
        print(f'CLASSIFICADOR                           : {self.estimator}')
        print(f'----------------------------------------------------------------------------------')
        print(f'Métricas no dataset de teste:        |    ')
        print(f'Intervalo de 95% da média            |     Média por classe')
        print(f'-------------------------------------|--------------------------------------------')
        print(r'ROC AUC MÉDIA      : %0.3f %s %0.3f   |  ' %\
             (np.round(self.means['roc_auc'],3), More_less, np.round(norm.ppf(0.975) * self.stds['roc_auc'] / np.sqrt(self.len),3)))
        print(f'ACCURACY  MÉDIA    : %0.3f %s %0.3f   |' %\
             (np.round(self.means['accuracy'],3), More_less, np.round(norm.ppf(0.975) * self.stds['accuracy'] / np.sqrt(self.len),3)))
        print(f'-------------------------------------|--------------------------------------------')
        print(f'                      MÉDIA MACRO    |CLASSE 0  |CLASSE 1')
        print(f'----------------------------------------------------------------------------------')
        print(f'PRECISÃO  MÉDIA    : %0.3f %s %0.3f   |%0.3f     |%0.3f    '%\
                  (np.round(self.means['precision'],3), More_less, np.round(norm.ppf(0.975) * self.stds['precision'] / np.sqrt(self.len),3),
                   np.round(tn/(tn+fn),3),np.round(tp/(tp+fp),3)))
        print(f'RECALL MÉDIO       : %0.3f %s %0.3f   |%0.3f     |%0.3f     '%\
                  (np.round(self.means['recall'],3), More_less, np.round(norm.ppf(0.975) * self.stds['recall'] / np.sqrt(self.len),3),
                   np.round(tn/(tn+fp),3), np.round(tp/(tp+fn),3)))
                
        print(f'F1-SCORE  MÉDIO    : %0.3f %s %0.3f   |%0.3f     |%0.3f    '%\
                  (np.round(self.means['f1'],3), More_less, np.round(norm.ppf(0.975) * self.stds['f1'] / np.sqrt(self.len),3),
                   np.round((2 * (tn/(tn+fn) * (tn/(tn+fp)))) / ((tn/(tn+fn)+(tn/(tn+fp)))), 3),
                   np.round((2 * (tp/(tp+fp) * (tp/(tp+fn)))) / ((tp/(tp+fp)+(tp/(tp+fn)))), 3)))
                   
        print(f'\nTEMPO MÉDIO DE TREINAMENTO:{np.round(self.time_mean,3)}')   
    
    def plot_confusion(self,
                      ax=None,
                      name_estimator='',
                      **kwargs_heatmap):
        '''
        Plota um mapa de calor com os valores compuatados de verdadeiros positivos, falsos positivos, verdadeiros neagtivos e falsos negativos. Esses valores são obtidos através da média de diversas 
        matrizes de confusões geradas na validação cruzada. OBS: os valores foram aproximados para o número inteiro mais próximo da média.
        
        Parâmetros:
        -----------
        ax : eixo a ser plotado o gráfico, se nenhum for passado será criado automaticamnete, tipo : matplotlib.axes, padrão : None
        name_estimator: Nome do modelo que aparece na legenda do gráfico, se nenhum for passado será imprimido como a função foi passada, tipo : str, padrão : None
        **kwargs_heatmap : argumentos adicionais a serem passados para função heatmap do seaborn
        
        Retorno:
        --------
        ax : Retorna o eixo do matplotlib onde foi gerado o gráfico
        '''
        
        #Verificando se foi passado um eixo
        if ax == None:
            fig, ax = plt.subplots(figsize = (16,12))
        #Verificando se algum nome para o modelo foi passado
        if name_estimator == '':
            name_estimator = self.estimator
            
        #Setando o eixo atual
        plt.sca(ax)
        
        #Gerando o gráfico com os valores da matriz de confusão média
        sns.heatmap(self.confusion_matrix_mean.round(), annot=True, cmap='Blues', annot_kws={"fontsize":30}, **kwargs_heatmap)
        
        #Gerando as labels e títulos
        labs(title=f'Matriz de confusão do modelo {name_estimator}', ax=ax, xlabel='VALORES PREVISTOS', ylabel='VALORES REAIS',
             subtitle=f'VALORES CALCULADOS PELA APROXIMAÇÃO DA MÉDIA DA VALIDAÇÃO CRUZADA COM {self.n_repeats} REPETIÇÕES E COM {self.n_splits} DIVISÕES NO DATASET')
        
        #Mudando o tamnho da fonte da barra de cor
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        
        #Retorna o eixo
        return ax
        

    def hist_metrics(self, 
                    central:bool=True, 
                    bins:int=5,
                    **kwargs_histplot):
        '''
        Parâmetros:
        -----------
        central : booleano indicando se é para plotar a média e a mediana das métricas no histograma, tipo : bool, padrão : True, 
        bins : quantidade de barras do histograma, tipo : int, padrão : 5
        **kwargs_histplot : argumentos adicionais a serem passados para função lineplot do seaborn
        
        Retorno:
        --------
        ax : Retorna o eixo do matplotlib onde foi gerado o gráfico
        
        '''
        #Cria uma figura sempre com 2 colunas
        fig, ax = plt.subplots(ceil(len(self.scores)/2),2, figsize = (20, int(8*len(self.scores)/2)))
        
        #Defini a linha e coluna como zeros
        i=0
        j=0
        #Percorre o dicionário de resultados
        for k, v in self.scores.items():
            #Verifica se aquelqa métrica foi calculada ou não
            if len(v) == 0:
                #Se não foi calculada passa para próxima
                pass
            
            else:
                #Define o eixo com a linha e coluna atual
                plt.sca(ax[i,j])
                
                #Gerando o histograma com os valores da métrica
                sns.histplot(v, bins=bins, **kwargs_histplot)
                
                #Gerando as labels e títulos
                labs(title=k.upper() + ' SCORE',xlabel='Valor da métrica',ylabel='Frequência', ax=ax[i,j])
                
                #Definindo o eixo x
                plt.xlim([0,1])
                
                #Verificando se é para plotar as medidas de tendência central(média e mediana)
                if central:
                    #Chama a função central trend
                    central_trend(v, ax[i,j])
                #Define a próxima linha e coluna
                if j == 1:
                    j = 0
                    i += 1
                else:
                    j+=1
                    
        #Verifica se o último gráfico ficou sozinho na última linha 
        if j == 1:
            #Alinha esse último gráfico
            plt.delaxes(ax= ax[-1][1])
            pos1 = ax[-1][0].get_position() 
            ax[-1][0].set_position([pos1.x0 + 0.2, pos1.y0, pos1.width, pos1.height])
        
        #Define o título da figura inteira
        ax[0,0].text(0,1.18,f'Distribuição dos desempenhos do modelo {self.estimator}', fontsize=25, transform=ax[0,0].transAxes)
        #Define o subtítulo da figura inteira
        ax[0,0].text(0,1.12,'MÉTRICAS OBTIDAS PELO MÉTODO REPEATEDSTRATIFIEDKFOLD', fontsize=15, transform=ax[0,0].transAxes, color='gray')
        
        #Retorna o eixo
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
        
        Retorno:
        --------
        ax : Retorna o eixo do matplotlib onde foi gerado o gráfico
        '''
        #Verificando se foi passado um eixo
        if ax == None:
            fig,ax = plt.subplots(figsize=(20,10))
        
        #Verificando se algum nome para o modelo foi passado
        if name_estimator == None:
            name_estimator = str(self.estimator)
        
        #Gerando o gráfico com os valores das taxas de verdadeiros positivos e falsos positivos(curva ROC)
        sns.lineplot(self.fpr_mean, self.tprs_mean, ax=ax, label= r'ROC CURVE (AUC MEAN = %0.3f $\pm$ %0.3f)' % \
                         (np.round(self.means['roc_auc'],3), np.round(norm.ppf(0.975) * self.stds['roc_auc'] / np.sqrt(self.len),3)) + f'{name_estimator}', estimator=None, **kwargs_lineplot)
        
        #Setando o eixo atual
        plt.sca(ax)
        
        #Gerando o intervalo de confiança da média de 95% com o desvido padrão amostral
        tprs_upper = np.minimum(self.tprs_mean + norm.ppf(0.975) * self.tprs_std / np.sqrt(self.len), 1)
        tprs_lower = np.maximum(self.tprs_mean - norm.ppf(0.975) * self.tprs_std / np.sqrt(self.len), 0)
        
        #Preenchendo esse intervalo com a mesma cor
        plt.fill_between(self.fpr_mean, tprs_lower, tprs_upper, color=ax.lines[-1].get_color(), alpha=0.05)
        
        #Plota uma linha de referência no meio do gráfico do minímo aceitável
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        #Gerando as labels e títulos
        labs(title = 'ROC CURVE', xlabel='Taxa de Falsos Positivos', ylabel='Taxa de Verdadeiros Positivos', ax=ax, \
             subtitle='ROC CURVE COMPUTADA PELAS VALORES DE VERDADEIROS POSITIVOS E FALSOS POSITIVOS OBTIDOS PELA VALIDAÇÃO CRUZADA')
        plt.legend(loc='lower right', fontsize=12)
        
        #Retorna o eixo
        return ax