# Prevendo a necessidade de internação para pacientes com COVID-19
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=flat-square&logo=Jupyter)](https://jupyter.org/try) [![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg?style=flat-square)](https://github.com/PedroHCAlmeida/analise_temporal_COVID_Brasil/edit/main/LICENSE)

![Alt](https://github.com/PedroHCAlmeida/Projeto_final_bootcamp/blob/main/img/previsao_banner.png?raw=true)

# **Sumário:**
<!--ts-->
   * [Apresentação](#apre)
   * [Resumo](#res)
   * [Contexto](#cont)
   * [Estrutura do projeto](#estru)
   * [Modelo Final](#model)
   * [Tecnologias utilizadas](#tec)
   * [Agradecimentos](#agrad)
   * [Referências](#ref)
   * [Contato](contato)
   <!--te-->
   
<a name="apre"></a>
# Apresentação

Olá, meu nome é Pedro Henrique, e esse é meu repositório referente ao projeto final do [Bootcamp de Data Science Aplicada](https://www.alura.com.br/bootcamp/data-science-aplicada/matriculas-abertas) promovido pela [Alura](https://www.alura.com.br/).

<a name="res"></a>
# Resumo 📜
Esse projeto teve como _**objetivo**_ criar um modelo capaz de prever se um paciente com suspeita de COVID-19 precisará ou não ser internado na UTI levando em consideração apenas os dados obtidos até as _**duas primeiras horas**_ que o mesmo chega ao local. Para isso foi utilizada uma base de dados disponibilizada pelo Hospital Sírio-Libanês, essa base de dados pode ser encontrada no [kaggle](https://www.kaggle.com/S%C3%ADrio-Libanes/covid19). 

Os _**impactos**_ esperados com a criação desse modelo são de um lado conseguir melhorar a logística de recursos do hospital, e o mais importante prever quais são aqueles pacientes que mais necessitam dessa internação, dando a prioridade para quem mais precisará, uma vez que _**cada leito vago pode significar uma vida salva**_.

![curva_covid](https://github.com/PedroHCAlmeida/Projeto_final_bootcamp/blob/main/img/curva_covid.gif?raw=true)

<a name="cont"></a>
# Contexto 🦠

O ano de 2020 começou de uma maneira completamente inesperada, o mundo foi atingido por uma das maiores crises sanitárias da história contemporânea, e uma palavra tomou conta das notícias nos jornais e mídias sociais brasileiras, _**"Leito"**_, a superlotação dos hopitais e a falta de leitos se tornou normal em todo o Brasil, nos anos de 2020 e 2021. Diante de uma doença desconhecida, encontrar como identificar seus perigos e as alterações corporais se tornou um desafio para medicina moderna.

Partindo desse problema o projeto teve como ponto de partida analisar dados acerca de resultados de exames de sangue, sinais vitais, gases sanguíneos, grupos de doenças apresentadas pelo paciente, além de informações demográficas do mesmo. A partir disso foi procurado quais as relações de cada variável com a necessidade de um leito, para no final sermos capazes de prever com a maior certeza quem são aqueles que mais necessitam dessa internação.

<a name="estru"></a>
# Estrutura do projeto 🧱

## [dados](https://github.com/PedroHCAlmeida/Projeto_final_bootcamp/tree/main/data):

Neste diretório se encontram os dados utilizados no projeto, esses dados estão dividos em duas pastas:

* [dados_brutos](https://github.com/PedroHCAlmeida/Projeto_final_bootcamp/tree/main/dados/dados_brutos) : aqui está o arquivo .xlsx original que foi disponibilizado pelo Hospital Sírio-Libanês
* [dados_preprocessados](https://github.com/PedroHCAlmeida/Projeto_final_bootcamp/tree/main/dados/dados_preprocessados) : aqui está o aquivo .csv com os dados pré-processados na [Análise Exploratória]()

## [funcoes](https://github.com/PedroHCAlmeida/Projeto_final_bootcamp/tree/main/functions):

Este diretório foi destinado as funções utilizadas no projeto, essas foram divididas em 4 arquivos .py dependendo do objetivo das mesmas, são eles:

* [preprocessing](https://github.com/PedroHCAlmeida/Projeto_final_bootcamp/blob/main/funcoes/preprocessing.py) : funções relacionadas ao pré-processamento dos dados
* [feature](https://github.com/PedroHCAlmeida/Projeto_final_bootcamp/blob/main/funcoes/feature.py) : funções relacionadas a seleção de variáveis(feature selection)
* [my_plot](https://github.com/PedroHCAlmeida/Projeto_final_bootcamp/blob/main/funcoes/my_plot.py) : funções relacionadas a geração de gráficos
* [my_classifier](https://github.com/PedroHCAlmeida/Projeto_final_bootcamp/blob/main/funcoes/my_classifier.py) : classe utilizada para criação dos modelos de machine learning

## [notebooks](https://github.com/PedroHCAlmeida/Projeto_final_bootcamp/tree/main/notebooks):

Aqui foram criados os notebooks onde foram realizadas as análises e criação dos modelos de Machine Learning, foram divididos em dois arquivos .ipynb, são eles:

* [Analise_exploratoria](https://github.com/PedroHCAlmeida/Projeto_final_bootcamp/blob/main/notebooks/Analise_exploratoria.ipynb): notebook desenvolvido no jupyter lab onde foi realizado o pré-processamento dos dados brutos e toda a parte de Análise exploratória.
* [Previsoes](https://github.com/PedroHCAlmeida/Projeto_final_bootcamp/blob/main/notebooks/Previsoes.ipynb) : notebook destinado aos modelos de Machine Learning a partir dos dados pré-processados

## [arquivos_modelo](https://github.com/PedroHCAlmeida/Projeto_final_bootcamp/tree/main/arquivos_modelo):

Neste repositório estão os objetos criados dentro do notebook e salvos em arquivos para serem carregados em outros locais,estes foram organizados em duas pastas:

* [SearchCV_salvo](https://github.com/PedroHCAlmeida/Projeto_final_bootcamp/tree/main/arquivos_modelo/SearchCV_salvo) : possui o arquivo onde salvo do objeto que foi utilizado na otimização dos hyperparâmetros. Optei por salvá-lo, uma vez que é um processo computacionalmente custoso e demorado, dessa forma é possível carregá-lo em outros ambientes.
* [Modelo_Salvo](https://github.com/PedroHCAlmeida/Projeto_final_bootcamp/tree/main/arquivos_modelo/Modelo_salvo) : aqui está salvo o modelo final que obteve as melhores métricas de acordo com o previsto no início do projeto. Esse modelo, no final, foi treinado com todos os dados e salvo em um aquivo .joblib nesta pasta .

## [img](https://github.com/PedroHCAlmeida/Projeto_final_bootcamp/tree/main/img)
Diretório com todas as imagens utilizadas no projeto.


## [extras](https://github.com/PedroHCAlmeida/Projeto_final_bootcamp/tree/main/extras):
Neste diretório experimentei testar diferentes funcionalidades que é possível entre o Rstudio e python. Para isso criei um projeto de um website dentro do Rstudio, e transformei os arquivos .ipynb em .Rmd, após isso configurei o Rstudio para utilizar o mesmo ambiente do anaconda que foi desenvolvido os notebooks originais. <br>

Por fim fiz pequenas mudanças em relação as leituras de dados e importações de funções locais e utilizei o arquivo [SearchCV_salvo/random_searchcv](https://github.com/PedroHCAlmeida/Projeto_final_bootcamp/blob/main/arquivos_modelo/SearchCV_salvo/random_searchcv) para não realizar o processo de otimização de hyperparâmetros novamente.

* Comando utilizado para utilizar o python no Rstudio no mesmo ambiente que foi desenvolvido os notebooks pelo jupyter lab
![](https://raw.githubusercontent.com/PedroHCAlmeida/Projeto_final_bootcamp/main/img/Screenshot%202021-08-07%20214512.png?token=ASNXTHUWDLAGNV2LIYRY3BDBB7664)

* Comando utilizado para gerar uma cópia dos notebooks no formato .Rmd
![](https://raw.githubusercontent.com/PedroHCAlmeida/Projeto_final_bootcamp/main/img/Screenshot%202021-08-07%20214603.png?token=ASNXTHWGF235V7HEU2OBBQLBB77AA)
 
* Obs: esses comandos só foram realizados uma vez na hora de iniciar a sessão, depois foram retirados

* O resultado final do site pode ser conferido [aqui](https://prevendo-uti-bootcamp-alura.netlify.app/)

<a name="model"></a>
# Modelo Final

<a name="tec"></a>
# Tecnologias utilizadas 💻

Esse projeto foi realizado utilizando a linguagem Python versão 3.9.6, e os notebooks foram desenvolvidos através da ferramenta jupyter lab dentro de um ambiente criado pela plataforma anaconda, as principais bibliotecas usadas foram:
* Pandas versão 1.3.1 : biblioteca rápida e poderosa usada para manipulação de dados
* Matplotlib versão 3.1.3 : biblioteca usada para visualização de dados
* Seaborn versão 0.11.1 : biblioteca baseada no Matplotlib para visualização de gráficos estatísticos mais complexos
* Numpy versão 1.20.2 : biblioteca utilizada para computação matemática
* Scikit-learn versão 0.24.2 : biblioteca utilizada na criação de modelos de Machine Learning
* Todas as bibliotecas e versões podem ser encontradas no arquivo [requirements.txt](https://github.com/PedroHCAlmeida/)

<a name="agrad"></a>
# Agradecimentos 😀

Gostaria de deixar o meu agradecimento a Alura por essa oportunidade incrível de participar do bootcamp de Data Science Aplicada, aos instrutores Thiago Santos, Guilherme Silveira, Allan Spadini e Karoline Penteado, que nos acompanharam durante todo o bootcamp. Ao Paulo Vasconcellos, que sempre participou das lives nesse período e trouxe dicas valiosas para melhorar os projetos. Além disso, queria agradecer a todo pessoal do discord e do Scuba team que sempre ajudou quando foi preciso.

<a name="apre"></a>
# Referências 📚

https://medium.com/data-hackers/como-selecionar-as-melhores-features-para-seu-modelo-de-machine-learning-faf74e357913<br>
https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html<br>
https://machinelearningmastery.com/k-fold-cross-validation/<br>
https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/<br>
https://towardsdatascience.com/5-reasons-why-you-should-use-cross-validation-in-your-data-science-project-8163311a1e79<br>
https://g1.globo.com/sp/sao-paulo/noticia/2020/11/18/pacientes-e-medicos-relatam-falta-de-leitos-de-uti-em-ao-menos-tres-grandes-hospitais-particulares-da-cidade-de-sp.ghtml<br>
https://caiquecoelho.medium.com/um-guia-completo-para-o-pr%C3%A9-processamento-de-dados-em-machine-learning-f860fbadabe1<br>
https://searchsoftwarequality.techtarget.com/definition/garbage-in-garbage-out#:~:text=George%20Fuechsel%2C%20an%20early%20IBM,processes%20what%20it%20is%20given<br>
https://machinelearningmastery.com/k-fold-cross-validation/#:~:text=Cross%2Dvalidation%20is%20a%20resampling,k%2Dfold%20cross%2Dvalidation<br>
https://numpy.org/doc/stable/index.html<br>
https://pandas.pydata.org/docs/index.html<br>
https://scikit-learn.org/stable/index.html<br>
https://matplotlib.org/<br>
https://seaborn.pydata.org/<br>

<a name="contato"></a>
# Contato ☎️

[<img src="https://img.shields.io/badge/pedrocorrea-0A66C2?style=flat-square&logo=linkedin&logoColor=white" />](https://www.linkedin.com/in/pedro-henrique-corr%C3%AAa-de-almeida-15398b105/)<br>
[<img src="https://img.shields.io/badge/GitHub-PedroHCAlmeida-DCDCDC?style=flat-square" />](https://github.com/PedroHCAlmeida)<br>
