---
layout: tutorial
comments: true
title: Análise de Hábitos de Compras Utilizando Regras de Associação
subtitle: "Algoritmo Apriori"
lang: pt
date: 2022-06-23
true-dt: 2020-06-23
tags: [Tutorial,python,Jupyter]
author: "Ricardo Avila"
comments: true
header-img: "img/basketMarket.png"
---
## Conteúdo

0. [Apriori (Regras de Associação)](#apriori)
1. [Conjunto de Dados](#dados)
2. [Pré-processamento dos registros](#pre)
3. [Apriori utiliza 3 variáveis](#regras)
4. [Considerações Finais](#fim)

## Apriori (Regras de Associação) <a name="apriori"></a>

Vamos ver o algoritmo Apriori em ação. Usaremos o algoritmo Apriori para encontrar regras que descrevem associações entre diferentes produtos comprados.

Este processo analisa os hábitos de compra de clientes por meio da descoberta de associações entre diferentes itens que aparecem no carrinho de compras. A descoberta destas associações ajuda os varejistas no desenvolvimento de estratégias de marketing, uma vez revelam quais itens são frequentemente comprados juntos pelos clientes.

{% highlight python %}
import pandas as pd
from apyori import apriori
{% endhighlight %}

## Conjunto de dados <a name="dados"></a>

Utilizando um cojunto de dados com 7500 registros de compras que ocorreram em um supermercado francês no período de uma semana.

Dataset disponível em: <a href="https://drive.google.com/file/d/1y5DYn0dGoSbC22xowBq2d4po6h1JxcTQ/view?usp=sharing" target="_blank">https://drive.google.com/file/d/1y5DYn0dGoSbC22xowBq2d4po6h1JxcTQ/view?usp=sharing</a>

{% highlight python %}
cesto = pd.read_csv('cesto.csv', header=None)
cesto.head()
{% endhighlight %}

{% highlight python %}
cesto.count(axis='columns')
{% endhighlight %}

## Pré-processamento dos registros <a name="pre"></a>

A biblioteca Apriori utiliza conjunto de dados como lista de listas.

Portanto, iremos formatar os registros em uma grande lista onde cada transação no conjunto de dados terá uma lista interna da grande lista externa.

{% highlight python %}
registros = []
for i in range(0, 7501):
    registros.append([str(cesto.values[i,j]) for j in range(0, 20)])
{% endhighlight %}

{% highlight python %} 
regras = list(apriori(registros, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2))
{% endhighlight %}

{% highlight python %} 
print(len(regras))
print(regras[0])
{% endhighlight %}

## Apriori utiliza 3 variáveis <a name="regras"></a>

Suporte (support) e confiança (confidence) são duas medidas de “interessabilidade” (interestingness), que refletem respectivamente a utilidade e confiabilidade da regra descoberta.

Um suporte de 2% para uma regra de associação significa que 4% de todas as transações sob análise mostram que frango e creme de leite são comprados juntos. O suporte do item I é definido como a razão entre o número de transações que contêm o item I pelo número total de transações.

A confiança de 29% significa que 29% das compras onde os clientes compraram frango também apresentam o item creme de leite como item vendido. Isso é medido pela proporção de transações com o item I1, nas quais o item I2 também aparece. A confiança entre dois itens I1 e I2, em uma transação, é definida como o número total de transações contendo os itens I1 e I2 dividido pelo número total de transações contendo I1.

Lift: Aumento é a razão entre a confiança e o suporte.

Tipicamente, regras de associação são consideradas de interesse se elas satisfazem tanto um suporte mínimo quanto uma confiança mínima.

{% highlight python %} 
mostrar = 0
for item in regras:
    items = [x for x in item[0]]
    print("Relação   - " + items[0] + " -> " + items[1])
    print("Suporte   - " + str(item[1]))
    print("Confiança - " + str(item[2][0][2]))
    print("Lift      - " + str(item[2][0][3]))
    print("#################################")
    mostrar += 1
    if (mostrar == 5):
        break
{% endhighlight %}

## Considerações Finais <a name="fim"></a>

Os passos de execução deste tutorial foram testados com `Python 3.6` e tudo ocorreu sem problemas. No entanto, é possível que alguém encontre alguma dificuldade ou erro no meio do caminho. Se for o caso, por favor comente a sua dificuldade ou erro neste post.