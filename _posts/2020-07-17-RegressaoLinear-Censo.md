---
layout: tutorial
comments: true
title: Previsão de Crescimento Populacional e da Expectativa de Vida
subtitle: "Utilizando Dados do Censo e Algoritmo de Regressão Linear"
lang: pt
date: 2020-07-17
true-dt: 2020-07-17
tags: [Tutorial,python,Jupyter]
author: "Ricardo Avila"
comments: true
header-img: "img/censo.png"
---

## Conteúdo

0. [Características do Algoritmo de Regressão Linear](#modelo)
1. [Conjunto de Dados](#dados)
2. [Criando o Modelo de Regressão Linear](#modelo)
3. [Considerações Finais](#fim)

## Características do Algoritmo de Regressão Linear <a name="modelo"></a>

O algoritmo de Regressão Linear é utilizado para estimar o valor de algo tendo como base uma serie de outros dados históricos. Esse modelo permite estudar as relações entre duas variáveis numéricas contínuas, que são valores que crescem ou decrescem constantemente. Essas variáveis podem ser definidas como:

* Uma variável de entrada (X) também chamada de variável preditora, explicativa ou independente.
* Uma variável de saída (Y) também chamada de variável dependente resposta ou resultado.

Assumimos que com a Regressão Linear uma variável dependente (Y) é influenciada por uma variável independente (X). As informações sobre a relação entre as variáveis é usada para prever e/ou descrever as mudanças futuras. Desse modo, é possível prever o que acontecerá com Y tendo como base o valor de X.

Para esse exemplo, primeiramente iremos importar as bibliotecas básicas que serão utilizadas nesse exemplo.

{% highlight python %}
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

%matplotlib inline
{% endhighlight %}

Além das bibliotecas Pandas e NumPy, também iremos utilizar o Scikit-learn para utilizar o algoritmo de Regressão Linear e a biblioteca MatPlotLib para gerar os gráficos.

## Conjunto de Dados <a name="dados"></a>

Para esse exemplo, os dados foram coletados do Portal do Instituto Brasileiro de Geografia e Estatistica (IBGE), mais especificamente na seção de Projeções e estimativas da população do Brasil e das Unidades da Federação. Por esse motivo o conjunto de dados em formato CSV está disponível para download no meu <a href="https://github.com/theavila/tutoriaisML">GitHub</a>. 

Se quiser conhecer mais sobre o Censo Brasileiro ou coletar mais dados, o site é esse: <a href="https://www.ibge.gov.br/apps/populacao/projecao/">https://www.ibge.gov.br/apps/populacao/projecao/</a>. 

Vamos carregar o conjunto de dados para dar início ao nosso exemplo.

{% highlight python %}
df = pd.read_csv('censo.csv')
df.head()
{% endhighlight %}

<img class="img-responsive center-block thumbnail" src="/img/headCenso.png" alt="Censo-head" style="width:30%"/>

Os dados possuem apenas dois valores, com o Ano e População, no intervalo de 1960 até 2020. Vamos plotar um gráfico com esse conjunto de dados para visualizar melhor.

{% highlight python %}
plt.xlabel('Ano')
plt.ylabel('População')
plt.scatter(df.Ano, df.Populacao, color='green', marker='+')
{% endhighlight %}

<img class="img-responsive center-block thumbnail" src="/img/plotCenso.png" alt="Censo-plot" style="width:70%"/>

Esse simples modelo mostra o seu potencial de execução, no qual podemos observar que a medida que forem alterados o valor da variável "Ano" o valor da variável "População" também será afetado. Isso mostra que existe um relacionamento linear entre elas.

De posse desta premissa básica, onde esse relacionamento deve existir, partiremos para a construção do nosso modelo de Regressão Linear.

## Criando o Modelo de Regressão Linear <a name="modelo"></a>

Utilizando as variáveis selecionadas, iremos treinar o modelo:

{% highlight python %}
reg = LinearRegression()
reg.fit(df[['Ano']], df.Populacao)
prev = reg.predict([[2021]])
print("Previsão de 2021 é: %d" % prev)
{% endhighlight %}

E a saída obtida foi:

```
Previsão de 2021 é: 222679197
```

E para gerar o gráfico, utilizamos o código a seguir:

{% highlight python %}
plt.xlabel('Ano')
plt.ylabel('População')
plt.scatter(df.Ano, df.Populacao, color='red', marker='+')
plt.plot(df.Ano, reg.predict(df[['Ano']]), color='blue')
{% endhighlight %}

<img class="img-responsive center-block thumbnail" src="/img/plotPopulacao.png" alt="Plot-populacao" style="width:70%"/>

E para efetuar a previsão da expectativa de vida da população, primeiro carregamos o conjunto de dados.

{% highlight python %}
df = pd.read_csv('expectativa.csv')
df.head()
{% endhighlight %}

E calculamos a previsão.

{% highlight python %}
reg = linear_model.LinearRegression()
reg.fit(df[['Ano']], df.Expectativa)
prev = reg.predict([[2021]])
print("Previsão 2021 é: %d" % prev)
{% endhighlight %}

E a saída obtida foi:

```
Previsão 2021 é: 77
```

E para gerar o gráfico, utilizamos o código a seguir:

{% highlight python %}
plt.xlabel('Ano')
plt.ylabel('Populacao')
plt.scatter(df.Ano, df.Expectativa, color='green', marker='+')
plt.plot(df.Ano, reg.predict(df[['Ano']]), color='blue')
{% endhighlight %}

<img class="img-responsive center-block thumbnail" src="/img/plotVida.png" alt="Plot-Vida" style="width:70%"/>

## Considerações Finais <a name="fim"></a>

É importante ressaltar que <strong>correlação não é causalidade</strong>. Ou seja, duas variáveis correlacionadas não implicam que uma variável é a causa da outra. Mesmo com uma base de dados confiável, não é possível prever o valor exato da variável de resposta relevante pois podem existir fatores omitidos que podem influenciar a variável de resposta. A regressão linear sempre tem um risco de erro, uma vez que na vida real uma variável independente nunca é uma perfeita preditora da variável dependente.

Como sempre sugiro, você também pode aplicar esse modelo em outras bases disponibilizadas na Internet, bastando fazer alguns ajustes quando necessário. Todo o código e mais um pouco está disponível no meu <a href="https://github.com/theavila/tutoriaisML">GitHub</a>.

Os passos de execução deste tutorial foram testados com `Python 3.6` e tudo ocorreu sem problemas. No entanto, é possível que alguém encontre alguma dificuldade ou erro no meio do caminho. Se for o caso, por favor comente a sua dificuldade ou erro neste post.