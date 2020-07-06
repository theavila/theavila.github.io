---
layout: tutorial
comments: true
color: "#ffc107"
title: Compra de Veículo utilizando Árvore de Decisão
subtitle: "Árvore de Decisão"
lang: pt
date: 2020-07-6
true-dt: 2020-07-6
tags: [Tutorial,python,Jupyter]
author: "Ricardo Avila"
comments: true
header-img: "img/carros.png"
---
## Conteúdo

0. [Árvore de Decisão](#arvore)
1. [Conjunto de Dados de Veículos](#dados)
2. [Pré-processamento dos registros](#pre)
3. [Criando a Árvore de Decisão](#modelo)
4. [Considerações Finais](#fim)

## Árvore de Decisão <a name="arvore"></a>

A Árvore de Decisão é um modelo que cria uma estrutura de árvore com a representação das possíveis decisões que podem ser tomadas, possibilitando separar as classes de dados de acordo com suas características. A ideia principal é gerar uma árvore de condições lógicas, que, por meio das características e valores dos registros,  seja capaz de separar ou filtrar os exemplos que pertencem a cada ramificação e assim chegar ao Nó folha e identificar a qual classe pertence os exemplos.

A Árvore de Decisão possui uma estrutura que facilita a sua interpretação, uma vez que é possível a partir de uma amostra percorrer seus nós de condições até chegar em uma folha que identifica a classe dessa amostra. Resumidamente, uma decisão é tomada através do caminhamento a partir do nó raiz até o nó folha.

Primeiramente, iremos importar as bibliotecas básicas que serão utilizadas nesse exemplo.

{% highlight python %}
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
{% endhighlight %}

Além das bibliotecas Pandas e NumPy, também iremos utilizar a LabelEncoder para manipular os dados no DataFrame e do objeto Tree que irá manipular a classe no modelo.

## Conjunto de Dados de Veículos <a name="dados"></a>

Nesse exemplo utilizamos um conjunto de dados de características de veículos, um simples modelo hierárquico de decisão útil para testar métodos de indução construtiva e descoberta de estrutura.

Esse conjunto de dados possui 1.728 registros com 6 características diferentes:

buying - preço de compra (v-high, high, med, low)
maint - preço da manutenção (v-high, high, med, low)
doors - número de portas (2, 3, 4, 5-more)
persons - número de pessoas para transportar (2, 4, more)
lug_boot - o tamanho da mala de bagagem (small, med, big)
safety - segurança estimada do carro (low, med, high)
class - classificação do carro (unacc, acc, good, v-good)

Dataset disponível em: <a href="http://archive.ics.uci.edu/ml/datasets/Car+Evaluation" target="_blank">http://archive.ics.uci.edu/ml/datasets/Car+Evaluation</a>

{% highlight python %}
df = pd.read_csv('car.csv')
df.head()
{% endhighlight %}

<img class="img-responsive center-block thumbnail" src="/img/cars.png" alt="iris-head" style="width:60%"/>

## Pré-processamento dos registros <a name="pre"></a>

Conforme pode ser visto no carregamento do conjunto de dados, algumas caracterísitcas dos veículos são representadas como texto. Para criarmos a Árvore de Decisão precisamos que os campos que participarão da composição sejam variáveis numéricas categóricas. Iremos transformar os valores em numéricos para permitir o uso do classificador.

{% highlight python %}
le_buying = LabelEncoder()
le_maint = LabelEncoder()
le_lug_boot = LabelEncoder()
le_safety = LabelEncoder()
le_class = LabelEncoder()
le_doors = LabelEncoder()
le_persons = LabelEncoder()

df['buying_n'] = le_buying.fit_transform(df['buying'])
df['maint_n'] = le_maint.fit_transform(df['maint'])
df['lug_boot_n'] = le_lug_boot.fit_transform(df['lug_boot'])
df['safety_n'] = le_safety.fit_transform(df['safety'])
df['class_n'] = le_class.fit_transform(df['class'])
df['doors_n'] = le_doors.fit_transform(df['doors'])
df['persons_n'] = le_persons.fit_transform(df['persons'])
{% endhighlight %}

A biblioteca LabelEncoder foi utilizada para transformar as variáveis textuais, atribuir uma valor para cada uma das possibilidades de cada categoria (característica do veículo). De um modo mais simples, criamos um novo DataFrame (entradas) sem as variáveis descritivas (v-high, high, med, low, etc.).

## Criando a Árvore de Decisão <a name="arvore"></a>

O próximo passo é determinar qual será o objetivo desse modelo de classificação. Por exemplo, determinar que o tipo de veículo seja de baixo custo. Colocamos a resposta em uma variável <strong>objetivo</strong>:

{% highlight python %}
objetivo = pd.Series(np.where(df['buying_n']<=1, 1, 0))
print(objetivo)
{% endhighlight %}

Em seguida, geramos um novo DataFrame com os dados que serão utilizados.

{% highlight python %}
entradas = df.drop(['buying','maint','lug_boot','safety','doors','persons','class'], axis='columns')
entradas.head()
{% endhighlight %}

E finalmente treinar o modelo utilizando o parâmetro objetivo criado anteriomente como saída desejada:

{% highlight python %}
model = tree.DecisionTreeClassifier()
model.fit(entradas, objetivo)
{% endhighlight %}

Para verificar se o modelo responde ao nosso objetivo, podemos verificar se existe algum veículo de baixo custo, baixa manutencão, seguro, 2 portas e pelo menos 4 lugares:

{% highlight python %}
# buying, maint, lug_boot, safety, class, doors, persons
model.predict([[1, 1, 1, 1, 1, 1, 1]])
{% endhighlight %}

E teremos como resposta 1 indicando que existe pelo menos um veículo com essas características. Vale a pena ressaltar que a Árvore de Decisão é um principais modelos utilizados em Ciência de Dados, sendo recomendável entender o seu funcionamento e aplicá-lo em problemas de classificação para comparar o resultado com outras estratégias.

Outro ponto importante é não pensar que para encontrar o seu objetivo bastava ter aplicado alguns filtros em alguma planilha eletrônica. Isso seria relativamente fácil com um conjunto de dados pequeno, porém inviável para uma base com muitos dados.

## Considerações Finais <a name="fim"></a>

Mesmo sendo um conjunto de dados simples, esse exemplo pode ser aplicado a outras bases disponibilizadas na Internet. Os conceitos serão os mesmos, bastando adaptar o código aqui aplicado. Todo o código e mais um pouco está disponível no meu <a href="https://github.com/theavila">GitHub</a>

Os passos de execução deste tutorial foram testados com `Python 3.6` e tudo ocorreu sem problemas. No entanto, é possível que alguém encontre alguma dificuldade ou erro no meio do caminho. Se for o caso, por favor comente a sua dificuldade ou erro neste post.