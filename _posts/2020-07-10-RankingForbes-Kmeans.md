---
layout: tutorial
comments: true
title: Clusterização dos Esportistas mais Bem Pagos do Mundo
subtitle: "Algoritmo K-Means utilizando o Ranking da Forbes de 2020"
lang: pt
date: 2020-07-10
true-dt: 2020-07-10
tags: [Tutorial,python,Jupyter]
author: "Ricardo Avila"
comments: true
header-img: "img/ranking.png"
---
## Conteúdo

0. [Características do Algoritmo K-Means](#modelo)
1. [Conjunto de Dados](#dados)
2. [Criando o Modelo de Treinamento](#modelo)
3. [Considerações Finais](#fim)

## Características do Algoritmo K-Means <a name="modelo"></a>

O algoritmo de clusterização K-Means é capaz de executar a classificação de informações utilizando a análise e a comparação entre valores numéricos dos dados. Essa classificação ocorre de forma automática e sem a necessidade de supervisão humana. Por ser um algoritmo de fácil implementação, mas de custo computacional alto como veremos mais a frente, o K-Means é um dos modelos de aprendizagem de máquina não supervisionado mais utilizados para a mineração de dados.

Para identificar as classes e classificar cada uma das ocorrências, o algoritmo K-Means executa a comparação entre cada valor por meio da distância. Normalmente é utilizada a distância euclidiana, porém outros tipos de medidas podem ser aplicados como a Manhattan, Minkowski, dentre outras. O calculo da distância depende da quantidade de atributos de cada registro fornecido. Quanto mais registros e atributos, maior será o tempo de processamento, aumentando o custo computacional. Isso se deve ao fato que deverá ser calculado as distâncias dos centroides para cada uma das classes. A cada iteração do algortimo K-Means, o valor de cada um dos centroide é refinado pela média dos valores de cada atributo de cada ocorrência que pertence a este centroide. Desse modo, são gerados K centroides para em seguida colocar as ocorrências da conjunto de dados de acordo com sua distância dos centroides.

O algoritmo K-Means é composto de cinco etapas:

* 1ª Etapa: Fornecer valores para os centroides
* 2ª Etapa: Gerar uma matriz de distância entre cada ponto e os centroides
* 3ª Etapa: Colocar cada ponto nas classes de acordo com a sua distância do centroide da classe
* 4ª Etapa: Calcular os novos centroides para cada classe
* 5ª Etapa: Repetir o processo a partir da 2ª etapa para o refinamento do cálculo das coordenadas dos centroides

Para esse exemplo, primeiramente iremos importar as bibliotecas básicas que serão utilizadas nesse exemplo.

{% highlight python %}
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

%matplotlib inline
{% endhighlight %}

Além das bibliotecas Pandas e NumPy, também iremos utilizar o Scikit-learn para utilizar o algoritmo K-Means e a biblioteca MatPlotLib para gerar os gráficos.

## Conjunto de Dados <a name="dados"></a>

Diferentemente dos outros exemplos realizado até o momento, dessa vez optamos por uma abordagem diferente. A lista com os 50 atletas mais bem pagos do mundo foi coletada diretamente do Ranking da Forbes fornecido em seu site. Esse processo de coleta foi realizado utilizando a biblioteca Beautiful Soup. Para que o tutorial não ficassde muito extenso, optei por não colocar as etapas executadas nessa postagem. No futuro eu posso fazer um tutorial de como utilizar a biblioteca Beautiful Soup para coletar dados de sites. Por esse motivo o conjunto de dados em formato CSV está disponível para download no meu <a href="https://github.com/theavila">GitHub</a>. 

Se quiser conhecer o Ranking da Forbes ou coletar mais dados, o site é esse: <a href="https://www.forbes.com/athletes/list/#tab:overall">https://www.forbes.com/athletes/list/#tab:overall</a>. 

<strong>Obs.</strong>: Vale ressaltar que o valor apresentado se refere ao que cada atleta irá receber no ano de 2020. Não se trata da fortuna acumulada por cada um dos atletas ao longo da carreira, uma vez que para isso seria necessário coletar os dados de outras fontes e/ou de anos anteriores da própria Forbes. Mesmo assim, os valores apresentados servem para deixar qualquer mortal admirado com o numerário recebido pelos atletas.

Os atributos que compõe o Ranking coletado são:

* Nome do Atleta
* Salário Total
* Salário recebido por Marketing
* Salário de Atleta
* Esporte do Atleta
* Idade do Atleta

Vamos carregar o conjunto de dados para dar início ao nosso exemplo.

{% highlight python %}
df = pd.read_csv('salaries50.csv', delimiter=';')
df.head(10)
{% endhighlight %}

<img class="img-responsive center-block thumbnail" src="/img/headForbes.png" alt="Forbes-head" style="width:80%"/>

Conforme pode ser visto, a lista está em ordem alfabetica pelo nome do atleta. Vamos ordenar os registros pelo valor total recebido por cada atleta.

<img class="img-responsive center-block thumbnail" src="/img/headForbesSalary.png" alt="Salary-Forbes-head" style="width:80%"/>

Uau! Não sei definir com o que fiquei mais impressionado. Nesse top 10 muitas coisas chamaram a atenção. O salário recebido em Marketing pelo excelente tenista Roger Federer é mais de 15 vezes o valor do seu salário com premiações e títulos. Pelo que conheço do tenista, ele é um exemplo de atleta e pessoa dentro e fora das quadras. Por passa essa imagem, Federer é patrocinado pelas marcas: Barilla, Credit Suisse Group (ADS), Mercedes-Benz, Rolex, Uniqlo e Wilson Sporting Goods.

Logo em seguida, na segunda posição do Ranking, vem o robozão Cristiano Ronaldo. A sua renda anual ficou ligeiramenmte melhor distruída, mesmo possuindo o terceiro maior salário dentre todos os atletas aqui listado. O maior salário pertence a Lionel Messi e o segundo a Neymar. Aliás, o atleta do Paris Saint Germain é o segundo atleta mais jovem do ranking com 28 anos. Nada mal para o menino Neymar. 

Como o nosso objetivo não +e ficar descrevendo os dados coletados, vamos em frente. Primeiramente, iremos verificar os tipos de dados.

{% highlight python %}
df.dtypes
{% endhighlight %}

E a saída produzida foi:

```
Name          object
Total        float64
Salary       float64
Marketing    float64
Sport         object
Age            int64
dtype: object
```

Sensacional! Os principais tipos de dados já estão no formato ideal para serem processados, pois iremos trabalhar a clusterização dos valores em float64 e inteiro: Total, Salary,Marketing e Age. 

Com uma simples descrição do dataframe podemos obter mais informações importantes:

{% highlight python %}
df.describe()
{% endhighlight %}

<img class="img-responsive center-block thumbnail" src="/img/describeForbes.png" alt="Describe-Forbes" style="width:80%"/>

Antes de seguir em frente, apenas para não passarmos em branco com essa Análise Exploratória de Dados (sempre importante para entender melhor o conjunto de dados independente da quantidade de registros), podemos observar que as colunas de salário e idade podem ser consideradas bem distribuídas com os valores de média, minímo e desvio padrão relativamente coerentes. o mesmo não pode ser dito da coluna de Marketing, que apresenta os valores de média e desvio padrão muito próximos, evidenciando que o máximo e mínimo estão muito separados. De fato, o valor máximo de 100 milhões e mínimo de 300 mil indica que esse atleta em particular precisa melhorar a sua imagem ou trocar de empresário. ;)

Vamos plotar alguns gráficos para ter uma melhor compreensão da relação da idade dos atletas com os outros atributos.

{% highlight python %}
f, axs = plt.subplots(1, 3, sharey=True)
f.suptitle('Idade em relação aos outros atributos')
df.plot(kind='scatter', x='Total', y='Age', ax=axs[0], figsize=(15, 3))
df.plot(kind='scatter', x='Salary', y='Age', ax=axs[1])
df.plot(kind='scatter', x='Marketing', y='Age', ax=axs[2])
{% endhighlight %}

<img class="img-responsive center-block thumbnail" src="/img/scatterIdadeOutros.png" alt="Scatter-Forbes" style="width:80%"/>

De acordo com os gráficos de dispersão (scatter plots), o grande diferencial são os valores pagos com Marketing. Podemos até ver 3 pontos separados de um grande aglomerado alinahdo a esquerda. Os outros atributos de Salário Total e Salário possuem distribuições razoáveis. Um fato importante que podemos destacar em relação a idade dos atletas é o fato do mais jovem possuir 21 anos e o mais velho 50 anos, sendo a média de idade de 31 anos e 8 meses.

Para entender um pouco mais o conjunto de dados (e finalmente acabar com essa Análise Exploratória de Dados, que aliás eu poderiam ser feitas outras), vamos plotar a relação entre Marketing e Idade:

{% highlight python %}
means = df.groupby('Age').mean()['Marketing']
errors = df.groupby('Age').std()['Marketing'] / np.sqrt(df.groupby('Age').count()['Marketing'])
ax = means.plot.bar(yerr=errors,figsize=(15,5))
{% endhighlight %}

<img class="img-responsive center-block thumbnail" src="/img/scatterMarketingIdade.png" alt="Marketing-idade-plot" style="width:70%"/>

De acordo com a imagem gerada, podemos verificar que as idades de 27, 32 e 34 anos se sobressaem em relação as outras. Pode até parecer que são informações irrelevantes, porém de acordo com os ranking de anos anteriores da própria Forbes, a média de idade dos atletas também ficou bem próxima dos 31 anos. Essa é exatamente a média dessa amostra com os valores 27, 32 e 34 e do conjunto de dados com os 50 atletas.

Coincidências a parte, vamos seguir em frente utilizando o algoritmo K-Means.

## Criando o Modelo de Treinamento <a name="modelo"></a>

Para utlizar o algoritmo de Lógica Nebulosa, fuzzificar  as entradas. Para isso iremos criar uma função para passar uma amostra da base de dados e, para cada atributo da amostra, iremos transformá-lo em outros 3 atributos (de acordo com o grau de pertinência para cada uma das 3 classes).

{% highlight python %}
def fuzzying(vetor):
    retorno = []
    for i in range(len(vetor)):
        retorno.append(fuzz.interp_membership(universo, fuzz.gaussmf(universo, media_setosa, desvio_setosa), vetor[i]))
        retorno.append(fuzz.interp_membership(universo, fuzz.gaussmf(universo, media_versicolor, desvio_versicolor), vetor[i]))
        retorno.append(fuzz.interp_membership(universo, fuzz.gaussmf(universo, media_virginica, desvio_virginica), vetor[i]))
    return np.asarray(retorno).T
{% endhighlight %}

Agora que a função está criada, vamos testá-la. Para isso iremos passar 3 valores para a função <strong>fuzzying</strong> e verificar a sua saída.

{% highlight python %}
fuzzying([0.2, 0.56, 0.3, 0.8])
{% endhighlight %}

```
array([0.99051683, 0.18656282, 0.08544104, 0.38156028, 0.77825999,
       0.90014951, 0.95995498, 0.53004173, 0.22417374, 0.05538159,
       0.0558928 , 0.77485862])
```

Ótimo! A função está funcionando conforme o esperado.

Agora iremos criar uma outra função para utilizar o algoritmo Perceptron Multicamadas da biblioteca do Scikit-Learn:

{% highlight python %}
def gerar_mlp(neuronios, taxa_aprendizado, max_iteracoes):
    mlp = MLPRegressor(
        solver='adam',
        hidden_layer_sizes=(neuronios,),
        random_state=1,
        learning_rate='constant',
        learning_rate_init=taxa_aprendizado,
        max_iter=max_iteracoes,
        activation='logistic',
        momentum=0.1
    )
    return mlp
{% endhighlight %}

Utilizando a função de fuzzificação, iremos processar todo o conjunto de dados da Iris:

{% highlight python %}
entradas_fuzzy = []
for i in range(150):
    entradas_fuzzy.append(fuzzying(entrada[i,:]))

entradas_fuzzy = np.asarray(entradas_fuzzy)
entradas_fuzzy
{% endhighlight %}

E transforma os vetores de saída para o tipo de dado <strong>float</strong>, mudando de [-1,0,1] em [0,0.5,1]:

{% highlight python %}
saida = np.arange(150, dtype=float)
for i in range(150):
    if i <50:
        saida[i] = 0
    if 50 <= i < 100:
        saida[i] = 0.5
    if 100 <= i < 150:
        saida[i] = 1
saida = saida.reshape(150,1)
{% endhighlight %}

Em seguida, como o objetivo é que a saída da Perceptron Multicamadas seja um valor processado pela Lógica Nebulosa, uma vez que teremos um único neurônio na camada de saida (dando uma resposta numérica, assim como uma regressão), precisamos criar uma função para indicar as faixas de resposta para cada uma das classes:

{% highlight python %}
def obter_classe(respostas, valor_menor, valor_maior):
    label_previsto = []
    for i in respostas:
        if i < valor_menor:
            label_previsto.append('Setosa')
        elif i >= valor_menor and i <= valor_maior:
            label_previsto.append('Versicolor')
        else:
            label_previsto.append('Virginica')
    return label_previsto
{% endhighlight %}

Outro passo importante é criar uma última função para mudar as saídas de [0,0.5,1] para ['Setosa', 'Versicolor', 'Virginica']:

{% highlight python %}
def traduz_saida(saidas):
    resposta = []
    for i in saidas:
        if i == 0:
            resposta.append('Setosa')
        elif i == 0.5:
            resposta.append('Versicolor')
        else:
            resposta.append('Virginica')
    return resposta
{% endhighlight %}

Finalmente, depois de tantas funções e transformações, podemos aplicaar o algoritmo Perceptron Multicamadas utilizando uma validação cruzada de tamanho três e gerando um relatório de desempenho para cada uma das interações:

{% highlight python %}
skf = StratifiedShuffleSplit(n_splits=3, test_size=0.25)    
for train_idx, test_idx in skf.split(entradas_fuzzy,saida.ravel()):
    x_treinamento = entradas_fuzzy[train_idx]
    y_treinamento = saida[train_idx]
    x_teste = entradas_fuzzy[test_idx]
    y_teste = saida[test_idx]
    mlp = gerar_mlp(20, 0.01, 100)
    mlp.fit(x_treinamento, y_treinamento.ravel())
    previsao = obter_classe(mlp.predict(x_teste), 0.25, 0.8)
    print('######################################################')
    print(classification_report(traduz_saida(y_teste), previsao))
{% endhighlight %}

A saída gerada será algo semelhante com o relatório a seguir:

```
######################################################
              precision    recall  f1-score   support

      Setosa       1.00      1.00      1.00        13
  Versicolor       0.83      0.83      0.83        12
   Virginica       0.85      0.85      0.85        13

    accuracy                           0.89        38
   macro avg       0.89      0.89      0.89        38
weighted avg       0.89      0.89      0.89        38

######################################################
              precision    recall  f1-score   support

      Setosa       0.92      0.92      0.92        12
  Versicolor       0.85      0.85      0.85        13
   Virginica       0.92      0.92      0.92        13

    accuracy                           0.89        38
   macro avg       0.90      0.90      0.90        38
weighted avg       0.89      0.89      0.89        38

######################################################
              precision    recall  f1-score   support

      Setosa       0.92      0.92      0.92        12
  Versicolor       0.86      0.92      0.89        13
   Virginica       1.00      0.92      0.96        13

    accuracy                           0.92        38
   macro avg       0.92      0.92      0.92        38
weighted avg       0.92      0.92      0.92        38
```

Utilizando a Lógica Nebulosa com o algoritmo Perceptron Multicamadas foi obtido uma precisão entre 89% e 92%. Um resultado que considero muito bom para esse conjunto de dados.

## Considerações Finais <a name="fim"></a>

Mais uma vez utilizamos o conjunto de dados de flores Iris para apresentar outros dois importantes modelos de aprendizagem de máquina bastante utilizado pelos Cientistas de Dados. Obtivemos um resultado entre 89% e 92% de precisão. Como sempre, recomenda-se conhecer o funcionamento e características desses modelos para tirar proveitos das suas vantagens e conhecer as suas limitações. Mesmo com tantas etapas de transformação dos dados e fuzzificação, a Lógica Nebulosa é muito aplicada e apresenta uma série de vantagens, conforme apresentado na sua descrição.

Como sempre sugiro, você também pode aplicar esse modelo em outras bases disponibilizadas na Internet, bastando fazer alguns ajustes quando necessário. Todo o código e mais um pouco está disponível no meu <a href="https://github.com/theavila">GitHub</a>

Os passos de execução deste tutorial foram testados com `Python 3.6` e tudo ocorreu sem problemas. No entanto, é possível que alguém encontre alguma dificuldade ou erro no meio do caminho. Se for o caso, por favor comente a sua dificuldade ou erro neste post.