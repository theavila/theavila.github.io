---
layout: tutorial
comments: true
title:  Classificação de Laranjas e Maças
subtitle: "Utilizando Máquina de Suporte Vetorial (SVM)"
lang: pt
date: 2020-07-20
true-dt: 2020-07-20
tags: [Tutorial,python,Jupyter]
author: "Ricardo Avila"
comments: true
header-img: "img/laranjamaca.png"
---

## Conteúdo

0. [Características da Máquina de Suporte Vetorial](#modelo)
1. [Conjunto de Dados](#dados)
2. [Criando o Modelo de Aprendizagem](#modelo)
3. [Considerações Finais](#fim)

## Características da Máquina de Suporte Vetorial <a name="modelo"></a>

O algoritmo Support Vector Machine (Máquina de Suporte Vetorial ou SVM) é um modelo de aprendizagem de máquina supervisionado utilizado para a classificação, regressão e também para encontrar outliers. O SVM é capaz de realizar a separação de um conjunto de objetos com diferentes classes utiliza a estratégia de planos de decisão. Para entender como isso é realizado, observe a figura a seguir.

<img class="img-responsive center-block thumbnail" src="/img/svm.png" alt="svm-image" style="width:50%"/>

Na figura é possível observar duas classes de objetos: verde ou vermelho. A linha que os separa define o limite em que se encontram os pontos verdes e os pontos vermelhos. Quando um novos objetos forem analisados, estes serão classificados como verdes se estiverem à esquerda e como vermelhos caso situem-se à direita. Nesse exemplo, os objetos foram separados por meio de uma linha (hiperplano) em seu respectivo grupo, caracterizando esse modelo como um classificador linear.

O principal objetivo do SVM é segregar os dados fornecidos da melhor maneira possível. Quando a segregação é feita, a distância entre os pontos mais próximos é conhecida como <strong>margem</strong>. A abordagem é selecionar um <strong>hiperplano</strong> com a margem máxima possível entre os <strong>vetores de suporte</strong> nos conjuntos de dados fornecidos.

<img class="img-responsive center-block thumbnail" src="/img/svm1.png" alt="svm-image1" style="width:50%"/>

Para selecionar o hiperplano máximo nos conjuntos fornecidos, o SVM segue os seguintes conjuntos:

* Gerando hiperplanos capaz de segregar as classes da melhor maneira possível
* Selecionando o hiperplano correto, que possua a segregação máxima dos pontos de dados mais próximos

Para esse exemplo, primeiramente iremos importar as bibliotecas básicas que serão utilizadas nesse exemplo.

{% highlight python %}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

from sklearn import svm
from sklearn import datasets
from sklearn.metrics import confusion_matrix
%matplotlib inline
{% endhighlight %}

Além das bibliotecas Pandas e Numpy, também iremos utilizar o Scikit-learn para utilizar o algoritmo SVM e as bibliotecas MatPlotLib e Seaborn para gerar os gráficos.

## Conjunto de Dados <a name="dados"></a>

Para esse exemplo, utilizaremos um conjunto de dados em formato CSV está disponível para download no meu <a href="https://github.com/theavila/tutoriaisML">GitHub</a>. 

Vamos carregar o conjunto de dados para dar início ao nosso exemplo.

{% highlight python %}
fruits = pd.read_csv('applesOranges.csv')
fruits.head()
{% endhighlight %}

<img class="img-responsive center-block thumbnail" src="/img/svmHead.png" alt="SVM-head" style="width:20%"/>

Os dados possuem apenas três variáveis, Peso (Weight), Tamanho (Size) e Classe (laranja ou maça). Vamos plotar uma figura para visualizar os objetos do conjunto de dados.

{% highlight python %}
sns.lmplot('Weight', 'Size', data=fruits, hue='Class',
           palette='Set1', fit_reg=False, scatter_kws={"s": 70});
{% endhighlight %}

<img class="img-responsive center-block thumbnail" src="/img/plotSVM.png" alt="SVM-plot" style="width:45%"/>

Onde seria o melhor local para inserir uma reta que separaria os dois grupos da melhor forma possível? O próximo passo é a construção do nosso modelo utilizando o algoritmo Support Vector Machine e a criação de um hiperplano capaz de separar as classes do cojunto de dados.

## Criando o Modelo de Aprendizagem <a name="modelo"></a>

Como o SVM é um modelo supervisionado, precisamos indicar os valores que serão utilizados para treinar o modelo. Nesse caso, utilizamos o Peso e Tamanho:

{% highlight python %}
fruta = fruits[['Weight', 'Size']].values
tipo = np.where(fruits['Class']=='orange', 65, 4)
{% endhighlight %}

Para em seguida treinar o modelo:

{% highlight python %}
model = svm.SVC(kernel='linear', decision_function_shape=None)
model.fit(fruta, tipo)
{% endhighlight %}

Feito o treinamento, iremos criar dois pontos para definir uma linha (para visualizar) como os objetos estão separados:

{% highlight python %}
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(60, 80)
yy = a * xx - (model.intercept_[0]) / w[1]
{% endhighlight %}

E plotamos tudo em um gráfico:

{% highlight python %}
sns.lmplot('Weight', 'Size', data=fruits, hue='Class',
           palette='Set2', fit_reg=False, scatter_kws={"s": 70});
plt.plot(xx, yy, linewidth=2, color='black');
{% endhighlight %}

<img class="img-responsive center-block thumbnail" src="/img/graficoSVM1.png" alt="SVM-grafico1" style="width:45%"/>

A reta do hiperplano foi encontrada, tendo como base as margens. Vamos inserir as duas linhas das margens para entender o processo.

Primeiro, criamos o modelo:

{% highlight python %}
b = model.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = model.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])
{% endhighlight %}

E geramos um gráfico utilizando o código a seguir:

{% highlight python %}
sns.lmplot('Weight', 'Size', data=fruits, hue='Class', palette='Set2', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=80, facecolors='none');
{% endhighlight %}

<img class="img-responsive center-block thumbnail" src="/img/graficoSVM2.png" alt="svm-grafico2" style="width:60%"/>

Seguindo as margens, o hiperplano desse Vetor de Suporte foi criado.

De posse desse modelo de previsão podemos criar um pequeno método, de acordo com o código a seguir:

{% highlight python %}
def apple_orange(tam, pes):
    if (model.predict([[tam, pes]])) == 0:
        print('Isto é uma laranja')
    else:
        print('Isto é uma maça')
{% endhighlight %}

Utilizando o método, podemos fazer algumas previsões:

{% highlight python %}
apple_orange(67, 4)
{% endhighlight %}

E a saída obtida foi:

```
Isto é uma maça
```

Parece que está funcionando corretamente. Com isso, criamos um modelo capaz de prever (classificar) laranjas e maças de acordo com o seu Peso e Tamanho.


## Considerações Finais <a name="fim"></a>

A Máquina de Suporte Vetorial é amplamente utilizado em diversas áreas devido às suas vantagens de aplicação. Dentre suas vantagens destaca-se o bom desempenho de generalização, a tratabilidade matemática, a interpretação geométrica e a utilização para a exploração de dados não rotulados.

Utilize esse exemplo para criar outros modelos de aprendizagem de máquina utilizando outras bases disponibilizadas na Internet. Para isso, basta efetuasr alguns ajustes quando for necessário. Todo o código e mais um pouco está disponível no meu <a href="https://github.com/theavila/tutoriaisML">GitHub</a>.

Os passos de execução deste tutorial foram testados com `Python 3.6` e tudo ocorreu sem problemas. No entanto, é possível que alguém encontre alguma dificuldade ou erro no meio do caminho. Se for o caso, por favor comente a sua dificuldade ou erro neste post.