import pandas as pd
import numpy as np
from scipy.stats import norm
from math import ceil
import time
from tqdm import tqdm_notebook as tqdm 

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import seaborn as sns
from plot import labs, central_trend

from sklearn.model_selection import RepeatedStratifiedKFold, ParameterGrid, ParameterSampler
from sklearn import metrics
from sklearn.pipeline import Pipeline

from IPython.display import clear_output

class Model:
    
    def __init__(self, model, dados:pd.DataFrame, pipeline_args:list=None):
        
        np.random.seed(64541)
        dados = dados.sample(frac=1).reset_index(drop=True)
        self.x = dados.drop(columns='ICU')
        self.y = dados['ICU']
        self.model = model
        self.pipeline_args = pipeline_args
        
        if pipeline_args is None:
            pipeline_args = [('feature_selection', None)]
        pipeline_args.append(tuple(['clf', self.model]))
        self.pipe = Pipeline(pipeline_args)
        
    def search(self, params, score='roc_auc', n_iter=100, n_splits=5, n_repeats=10, report_final:bool=True):
        
        ps = ParameterSampler(params, n_iter=n_iter, random_state=64541)
        
        score_grid = []
        for _, grid in zip(tqdm(range(ps.__len__())),ps.__iter__()):
            new_model = self.model.__class__
            new_pipe = self.pipe[:-1].steps
            new_pipe.append(tuple(['clf', new_model(**grid)]))
            self.pipe = Pipeline(new_pipe)
            try:
                self.cross_val(scoring=[score], fprs_tprs=False, train_results=False, report=False, n_splits=n_splits, n_repeats=n_repeats, print_tqdm=False)
                score_grid.append(tuple([grid, self.means[score]]))
            except ValueError:
                continue   
        self.best_params = max(score_grid, key = lambda t: t[1])[0]
        if report_final:
            new_model = self.model.__class__
            new_pipe = self.pipe[:-1].steps
            new_pipe.append(tuple(['clf', new_model(**self.best_params)]))
            self.pipe = Pipeline(new_pipe)
            self.cross_val(n_splits=n_splits, n_repeats=n_repeats)
        return self.best_params
        
    def cross_val(self, train_results:bool=True , report:bool=True, n_splits:int=5, n_repeats:int=10, scoring=['f1', 'roc_auc', 'precision', 'recall', 'accuracy'], 
                  fprs_tprs=True, print_tqdm=True):

        self.len = n_splits * n_repeats
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        np.random.seed(64541)
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
        time_list = []
        self.scores = {metric:[] for metric in scoring}
        self.scores_train = {metric:[] for metric in ['f1', 'roc_auc', 'precision', 'recall', 'accuracy']}
        tprs = []
        self.fpr_mean = np.linspace(0, 1, 100)
        
        if print_tqdm == True:
            bar = tqdm(range(self.len))
        else:
            bar = range(self.len)
        
        for _, (train, test) in zip(bar,cv.split(self.x, self.y)):
            start = time.time()
            self.pipe.fit(self.x.iloc[train,:], self.y.iloc[train])
            end = time.time()
            time_list.append(end-start)
            pred_proba = self.pipe.predict_proba(self.x.iloc[test,:])
            pred = self.pipe.predict(self.x.iloc[test,:])
            
            for metric in scoring:
                try:
                    _= getattr(metrics, f'{metric}_score')(self.y.iloc[test], pred_proba[:,1])  
                    self.scores[metric].append(_)  
                except ValueError:
                    try:
                        _ = getattr(metrics, f'{metric}_score')(self.y.iloc[test], pred, average='macro') 
                        self.scores[metric].append(_)
                    except TypeError:
                        _ = getattr(metrics, f'{metric}_score')(self.y.iloc[test], pred) 
                        self.scores[metric].append(_)
            
            if fprs_tprs:
                fpr, tpr, _ = metrics.roc_curve(self.y.iloc[test], pred_proba[:,1])
                interp_tpr = np.interp(self.fpr_mean, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                
            if train_results:
                pred_proba = self.pipe.predict_proba(self.x.iloc[train,:])
                pred = self.pipe.predict(self.x.iloc[train,:])
                for metric in scoring:
                    try:
                        _ = getattr(metrics, f'{metric}_score')(self.y.iloc[train], pred_proba[:,1])  
                        self.scores_train[metric].append(_)  
                    except ValueError:
                        try:
                            _ = getattr(metrics, f'{metric}_score')(self.y.iloc[train], pred, average='macro') 
                            self.scores_train[metric].append(_)
                        except TypeError:
                            _ = getattr(metrics, f'{metric}_score')(self.y.iloc[train], pred) 
                            self.scores_train[metric].append(_)
        
        self.means = dict(map(lambda kv: (kv[0], np.mean(kv[1],axis=0)), self.scores.items()))
        self.stds = dict(map(lambda kv: (kv[0], np.std(kv[1],ddof=1)), self.scores.items()))
        self.means_train = dict(map(lambda kv: (kv[0], np.mean(kv[1],axis=0)), self.scores_train.items()))
        self.stds_train = dict(map(lambda kv: (kv[0], np.std(kv[1],ddof=1)), self.scores_train.items())) 
        self.time_mean = np.mean(time_list,axis=0)
        self.tprs_mean = np.mean(tprs, axis=0)
        tprs_std = np.std(tprs, axis=0, ddof=1)
        self.time_mean = np.mean(time_list,axis=0)
        
        if report:
            self.report()
        
    def report(self):
        More_less = u"\u00B1"
        print(f'{self.n_repeats} repetições de Validação Cruzada com {self.n_splits} divisões no dataset')
        print(f'----------------------------------------------------------------------------------')
        print(f'CLASSIFICADOR                   : {self.pipe["clf"]}')
        print(f'MÉTODOS DE SELEÇÃO DE VARIÁVEIS : {self.pipe["feature_selection"]}\n')
        print(f'------------------------------------------|---------------------------------------')
        print(f'Métricas no dataset de teste:             | Métricas no dataset de treino: ')
        print(f'------------------------------------------|---------------------------------------')
        print(f'ROC AUC MÉDIA         : %0.3f             |ROC AUC MÉDIA         : %0.3f' % (np.round(self.means['roc_auc'],3), np.round(self.means_train['roc_auc'],3)))
        print(f'ROC AUC DESVIO PADRÃO : %0.3f             |ROC AUC DESVIO PADRÃO : %0.3f'% (np.round(self.stds['roc_auc'],3), np.round(self.stds_train['roc_auc'],3)))
        print(r'ROC AUC ITERVALO      : %0.3f %s %0.3f     |ROC AUC ITERVALO      : %0.3f %s %0.3f' %\
              (np.round(self.means['roc_auc'],3), More_less, np.round(norm.ppf(0.975) * self.stds['roc_auc'] / np.sqrt(self.len),3), 
              np.round(self.means_train['roc_auc'],3), More_less, np.round(norm.ppf(0.975) * self.stds_train['roc_auc'] / np.sqrt(self.len),3)))
        print(f'DE CONFIÂNCIA DE 95%                      |DE CONFIÂNCIA DE 95%\n  ')
        print(f'DEMAIS MÉTRICAS                           |DEMAIS MÉTRICAS')
        print(f'------------------------------------------|---------------------------------------')
        print(f'ACCURACY  MÉDIA       : %0.3f             |ACCURACY  MÉDIA       : %0.3f' % (np.round(self.means['accuracy'],3), np.round(self.means_train['accuracy'],3)))
        print(f'PRECISÃO  MÉDIA       : %0.3f             |ACCURACY  MÉDIA       : %0.3f' % (np.round(self.means['precision'],3), np.round(self.means_train['precision'],3)))
        print(f'RECALL MÉDIO          : %0.3f             |RECALL MÉDIO          : %0.3f'% (np.round(self.means['recall'],3), np.round(self.means_train['recall'],3)))
        print(f'F1-SCORE  MÉDIO       : %0.3f             |F1-SCORE  MÉDIO       : %0.3f'% (np.round(self.means['f1'],3), np.round(self.means_train['f1'],3)))
        print(f'\nTEMPO MÉDIO DE TREINAMENTO:{np.round(self.time_mean,3)}')   
        
    def hist_metrics(self, ax=None, show=True, central=True, **kwargs_histplot):
    
        if ax == None:
            fig, ax = plt.subplots(ceil(len(self.scores)/2),2, figsize = (20, int(8*len(self.scores)/2)))
        i=0
        j=0
        for k, v in self.scores.items():
            plt.sca(ax[i,j])
            sns.histplot(v, bins=10, **kwargs_histplot)
            labs(title=k.upper() + ' SCORE',xlabel='Valor',ylabel='Frequência', ax=ax[i,j])
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

        ax[0,0].text(0,1.18,f'Distribuição dos desempenhos do modelo {self.model}', fontsize=25, transform=ax[0,0].transAxes)
        ax[0,0].text(0,1.12,'MÉTRICAS OBTIDAS PELO MÉTODO REPEATEDSTRATIFIEDKFOLD', fontsize=15, transform=ax[0,0].transAxes, color='gray')
        
        if show:
            plt.show()
        else:
            return ax
    
    def plot_roc_curve(self, ax=None, **kwargs_lineplot):
        
        if ax==None:
            fig,ax=plt.subplots(figsize=(20,10))
            
        name = self.model
        sns.lineplot(self.fpr_mean, self.tprs_mean, ax=ax, label= r'ROC CURVE (AUC MEAN = %0.3f $\pm$ %0.2f)' % \
                     (np.round(self.means['roc_auc'],3), np.round(norm.ppf(0.975) * self.stds['roc_auc'] / np.sqrt(self.len),3)) + f'{name}', estimator=None, **kwargs_lineplot)
        
        plt.sca(ax)
        tprs_upper = np.minimum(self.tprs_mean + norm.ppf(0.975) * self.stds_train['roc_auc'] / np.sqrt(self.len), 1)
        tprs_lower = np.maximum(self.tprs_mean - norm.ppf(0.975) * self.stds_train['roc_auc'] / np.sqrt(self.len), 0)
        plt.fill_between(self.fpr_mean, tprs_lower, tprs_upper, color=ax.lines[-1].get_color(), alpha=.05)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        labs(title = 'ROC CURVE', xlabel='False Positive Rate', ylabel='True Positive Rate', ax=ax, \
             subtitle='ROC CURVE COMPUTADA PELAS VALORES DE VERDADEIROS POSITIVOS E FALSOS POSITIVOS OBTIDOS PELA VALIDAÇÃO CRUZADA')
        plt.legend(loc='lower right', fontsize=12)
        
        return ax