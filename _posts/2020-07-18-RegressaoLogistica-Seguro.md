---
layout: tutorial
comments: true
title:  Tomada de Decisão para a Adesão de Seguro de Vida
subtitle: "Utilizando Algoritmo de Regressão Logística"
lang: pt
date: 2020-07-18
true-dt: 2020-07-18
tags: [Tutorial,python,Jupyter]
author: "Ricardo Avila"
comments: true
header-img: "img/seguro.png"
---

## Conteúdo

0. [Características do Algoritmo de Regressão Logística](#modelo)
1. [Conjunto de Dados](#dados)
2. [Criando o Modelo de Regressão Linear](#modelo)
3. [Considerações Finais](#fim)

## Características do Algoritmo de Regressão Logística <a name="modelo"></a>

O algoritmo de Regressão Logística é análogo ao de Regressão Linear, também sendo utilizado para problemas de classificação quando o objetivo é categorizar alguma variável por classes.

Os problemas de classificação podem ser de dois tipos:
* Binários - O cliente quer ou não quer aderir ao seguro
* Multiclasse - Qual candidato eu devo votar?

Vamos começar com o tipo Binário e futuramente veremos o outro.

Para esse exemplo, primeiramente iremos importar as bibliotecas básicas que serão utilizadas nesse exemplo.

{% highlight python %}
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from scipy.special import expit

%matplotlib inline
{% endhighlight %}

Além da biblioteca Pandas, também iremos utilizar o Scikit-learn para utilizar o algoritmo de Regressão Logística e a biblioteca MatPlotLib para gerar os gráficos. Finalmente, o método <strong>expit</strong> da biblioteca SciPy será utilizado para traçar a <strong>sigmóide</strong>. 

## Conjunto de Dados <a name="dados"></a>

Para esse exemplo, utilizaremos um conjunto de dados em formato CSV está disponível para download no meu <a href="https://github.com/theavila/tutoriaisML">GitHub</a>. 

Vamos carregar o conjunto de dados para dar início ao nosso exemplo.

{% highlight python %}
df = pd.read_csv('seguro.csv')
df.head()
{% endhighlight %}

<img class="img-responsive center-block thumbnail" src="/img/headSeguro.png" alt="Seguro-head" style="width:20%"/>

Os dados possuem apenas duas variáveis, sendo uma a idade e a outra se a pessoa possui (valor 1) ou não possui (valor 0) um seguro de vida.

O próximo passo é a construção do nosso modelo de Regressão Logística.

## Criando o Modelo de Regressão Linear <a name="modelo"></a>

Utilizando as variáveis selecionadas, iremos treinar o modelo:

{% highlight python %}
reg = LogisticRegression(solver='lbfgs')
reg.fit(df[['idade']], df.seguro)
{% endhighlight %}

Agora que foi determinado que a classe <strong>idade</strong> é o ponto focal para determinar se a pessoa deve ou não aderir ao plano de seguro de vida, vamos realizar algumas previsões. Para isso, basta passar como parâmetro o valor <strong>idade</strong> para que a função retorne o resultado.

{% highlight python %}
reg.predict([[13], [26], [44], [64], [80]])
{% endhighlight %}

E a saída obtida foi:

```
array([0, 0, 1, 1, 1], dtype=int64)
```

De acordo com o resultado, as pessoas com 13 e 26 anos não aderem ao seguro de vida, enquanto que as com 44, 64 e 80 aderem. 

E para gerar o gráfico, utilizamos o código a seguir:

{% highlight python %}
from scipy.optimize import curve_fit
import numpy as np

def f_sigmoide(x, x0, k):
    y = 1.0 / (1 + np.exp(-np.dot(k, x-x0)))
    return y

popt, pconv = curve_fit(f_sigmoide, df['idade'], df.seguro)
sigm1 = f_sigmoide(df['idade'], *popt)

plt.plot(df['idade'], sigm1)
plt.scatter(df['idade'], df.seguro, c=df.seguro, cmap='Paired', edgecolors='b')
plt.show()
{% endhighlight %}

<img class="img-responsive center-block thumbnail" src="/img/sigmoideSeguro.png" alt="sigmoide-Seguro-Vida" style="width:60%"/>

A função chamada sigmóide, também conhecida como <strong>função de achatamento</strong>, é uma função matemática que tem uma característica de <strong>S<strong> em forma de curva ou sigmóide curva.

Porém, a função <strong>expit<strong> já realiza toda essa implementação, e podemos plotar o gráfico da seguinte maneira:

{% highlight python %}
sigm2 = expit(df['idade'] * reg.coef_[0][0] + reg.intercept_[0])

plt.plot(df['idade'], sigm2)
plt.scatter(df['idade'], df.seguro, c=df.seguro, cmap='Paired', edgecolors='b')
plt.show()
{% endhighlight %}

<img class="img-responsive center-block thumbnail" src="/img/expitSeguro.png" alt="expit-Seguro-Vida" style="width:60%"/>

## Considerações Finais <a name="fim"></a>

A Regressão Logística é um dos métodos de classificação amplamente conhecido e utilizado pelos Cientistas de Dados. Mesmo sendo um método que não apresenta bons resultados de acurácia, é extremamente fácil de ser implementado e pode ser aplicado em problemas onde existem poucas classes. Outro ponto importante é o fato da Regressão Logística fazer parte do bloco fundamental onde são construídas as redes neurais artificias. Desse modo, a sua compreensão é muito importante para a construção de modelos de classificação.

Utilize esse exemplo para criar outros modelos de aprendizagem de máquina utilizando outras bases disponibilizadas na Internet. Para isso, basta efetuasr alguns ajustes quando for necessário. Todo o código e mais um pouco está disponível no meu <a href="https://github.com/theavila/tutoriaisML">GitHub</a>.

Os passos de execução deste tutorial foram testados com `Python 3.6` e tudo ocorreu sem problemas. No entanto, é possível que alguém encontre alguma dificuldade ou erro no meio do caminho. Se for o caso, por favor comente a sua dificuldade ou erro neste post.