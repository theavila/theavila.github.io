---
layout: tutorial
comments: true
title: Classificador Probabilístico com o Algoritmo Naïve Bayes
subtitle: "Classificação de E-mails como SPAM ou HAM"
lang: pt
date: 2020-07-15
true-dt: 2020-07-15
tags: [Tutorial,python,Jupyter]
author: "Ricardo Avila"
comments: true
header-img: "img/spam.png"
---
## Conteúdo

0. [Características do Algoritmo Naïve Bayes](#modelo)
1. [Conjunto de Dados](#dados)
2. [Criando o Modelo de Treinamento](#treino)
3. [Considerações Finais](#fim)

## Características do Algoritmo Naïve Bayes <a name="modelo"></a>

O modelo probabilístico conhecido como Teorema de Bayes ou simplesmente chamado de Algoritmo de Naïves Bayes e um dos mais utilizados na área de Ciência de Dados. O Teorema de Bayes foi desenvolvido por Thomas Bayes, um pastor presbiteriano e matemático inglês que foi membro da Royal Society. O Pastor Bayes desenvolveu o algoritmo com o objetivo de tentar provar a existência de Deus. Embora não tenha provado a existência de Deus, o Teorema de Bayes se mostrou útil de outras maneiras. (Obs.: A minha fé crê na existência de Deus muito mais que no acaso evolutivo de uma sopa primordial. A criação não anula a evolução e vice-versa.)
 
O Teorema de Bayes auxilia os cientistas e pesquisadores a avaliar a probabilidade de que algo seja verdade com base em novos dados. Por exemplo, os médicos podem usar o resultado de um exame de mamografia, que às vezes está errado, para avaliar se devem revisar a avaliação de que uma paciente tem câncer de mama ou não. A fórmula mostra que o grau de alteração depende da precisão do teste.

Por ser de fácil implementação e baixo custo computacional, possui um desempenho considerado melhor do que outros classificadores. Outra característica importante é precisar de um número menor de dados de teste para executar as classificações e alcançar uma boa precisão.

O fato de ser chamado de “naïve” (ingênuo) se leva ao fato do algoritmo desconsiderar a correlação entre as variáveis (features). Por exemplo, se determinada fruta é considerada uma <strong>Laranja</strong> se ela for da cor <em>laranja</em>, <em>redonda</em> e possui <em>100gr de peso</em>, o algoritmo não considerará a correlação entre esses fatores, tratando cada um deles de forma independente.

Nesse exemplo iremos utilizar a biblioteca Scikit Learn (ou sklearn). No Sklearn o algoritmo Naïve Bayes é implementado de 3 formas:

* Gaussian
* Multinomial
* Bernoulli

Cada implementação é utilizada para objetivos diferentes.

Para esse exemplo, primeiramente iremos importar as bibliotecas básicas que serão utilizadas nesse exemplo.

{% highlight python %}
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
{% endhighlight %}

Além da biblioteca NumPy, também iremos utilizar o Scikit-learn para utilizar o algoritmo Naïve Bayes em suas 3 formas.

## Conjunto de Dados <a name="dados"></a>

O conjunto de dados que será utilizado nesse exemplo contém 48 atributos reais contínuos (com valores de 0, 100) do tipo word_freq_WORD = porcentagem de palavras no e-mail que correspondem a WORD, ou seja, 100<strong>*</strong>(número de vezes que a WORD aparece no e-mail)<strong>/</strong>número total de palavras no e-mail. Uma WORD ou "palavra" neste caso é qualquer sequência de caracteres alfanuméricos delimitada por caracteres não alfanuméricos ou fim de sequência.

Características dos atributos:
* 6 atributos reais contínuos (0,100) do tipo char_freq_CHAR = porcentagem de caracteres no e-mail que correspondem a CHAR, ou seja, 100<strong>*</strong>(número de ocorrências de CHAR)<strong>/</strong>total de caracteres no email
* 1 atributo real contínuo do tipo capital_run_length_average = comprimento médio de sequências ininterruptas de letras maiúsculas
* 1 atributo inteiro contínuo do tipo capital_run_length_longest = comprimento da maior sequência ininterrupta de letras maiúsculas
* 1 atributo inteiro contínuo do tipo capital_run_length_total = soma do comprimento das sequências ininterruptas de letras maiúsculas = número total de letras maiúsculas no e-mail
* 1 atributo de classe nominal {0,1} do tipo spam = indica se o e-mail foi considerado spam (1) ou não (0), ou seja, e-mail comercial não solicitado.

Dataset disponível em: <a href="https://archive.ics.uci.edu/ml/datasets/spambase" target="_blank">https://archive.ics.uci.edu/ml/datasets/spambase</a>

{% highlight python %}
dataset = np.loadtxt('spam.csv', delimiter=',')
print(dataset[0])
{% endhighlight %}

<img class="img-responsive center-block thumbnail" src="/img/spamHEAD.png" alt="spam-head" style="width:85%"/>

## Criando o Modelo de Treinamento <a name="treino"></a>

O primeiro passo após carregar o conjunto de dados é criar os conjuntos de treino e teste. Para isso, executamos o comando a seguir:

{% highlight python %}
X = dataset[:, 0:48]
y = dataset[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .40, random_state = 17)
{% endhighlight %}

Como o conjunto de dados é pequeno, optamos por selecionar 40% para teste, o que pode ser ajustado caso os resultados de precisão não fiquem bons. Nesse caso, esse valor foi determinado após algumas iterações.

O próximo passo é executar os testes e treinos utilizando os algoritmos Naïve Bayes implementados no Sklearn em suas 3 formas. 

Primeiramente o MultinomialNB, que é adequado para classificação com recursos discretos (por exemplo, a contagem de palavras para classificação de texto).

{% highlight python %}
MultiNB = MultinomialNB()
MultiNB.fit(X_train, y_train)
print(MultiNB)

y_expect = y_test
y_pred = MultiNB.predict(X_test)
print(accuracy_score(y_expect, y_pred))
{% endhighlight %}

E a saída obtida foi:

```
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
0.8750678978815861
```

Nada mal! Esse modelo obteve 87,50% de precisão.

Vamos para a segunda implementação com BernoulliNB, que também é um classificador indicado para dados discretos. A sua diferença é ter sido projetado para recursos binários/booleanos, enquanto que o MultinomialNB trabalha com contagens de ocorrências.

{% highlight python %}
BernNB = BernoulliNB(binarize = 0.0)
BernNB.fit(X_train, y_train)
print(BernNB)

y_expect = y_test
y_pred = BernNB.predict(X_test)
print(accuracy_score(y_expect, y_pred))
{% endhighlight %}

Com essa abordagem, a saída obtida foi:

```
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
0.8837588267246062
```

Esse resultado foi um pouco melhor. Claramente por causa da base possuir muitos dados discretos com valores binários/booleanos. ;)

E finalmente, vamos utilizar a implementação com GaussianNB. Esse método executa chamadas recursivas várias vezes em diferentes partes de um conjunto de dados. Isso é especialmente útil quando o conjunto de dados é muito grande para caber na memória de uma só vez. Em nosso caso, por ser um conjunto de dados pequeno, iremos executar apenas para avaliar o resultado.

{% highlight python %}
GausNB = GaussianNB()
GausNB.fit(X_train, y_train)
print(GausNB)

y_expect = y_test
y_pred = GausNB.predict(X_test)
print(accuracy_score(y_expect, y_pred))
{% endhighlight %}

E o resultado obtido foi:

```
GaussianNB(priors=None, var_smoothing=1e-09)
0.8126018468223791
```

Conforme esperado, a implementação com GaussianNB obteve o resultado mais baixo dentre os modelos disponíveis de algoritmos Naïve Bayes implementados no Sklearn.

Vamos executar apenas mais um ajuste no modelo que obteve o melhor resultado, no caso o BernoulliNB, para verificar se conseguimos obter uma melhor classificação. Iremos ajustar o 
parâmetro binarize que é relativo ao limite para binarização (mapeamento para booleanos) de recursos de amostra. Anteriormente havíamos definido como 0, no qual presume-se que a entrada já consista em vetores binários. Vamos ajustar para 0.2 e verificar o que acontece.

{% highlight python %}
BernNB = BernoulliNB(binarize = 0.2)
BernNB.fit(X_train, y_train)
print(BernNB)

y_expect = y_test
y_pred = BernNB.predict(X_test)
print(accuracy_score(y_expect, y_pred))
{% endhighlight %}

E finalmente, o melhor resultado obtido foi:

```
BernoulliNB(alpha=1.0, binarize=0.2, class_prior=None, fit_prior=True)
0.8935361216730038
```

## Considerações Finais <a name="fim"></a>

Não é atôa que o algoritmo Naïves Bayes é um dos modelos de aprendizagem de máquina mais utilizados pelos Cientistas de Dados. Caso o problema que você esteja trabalhando seja a classificação de textos ou algo parecido, esse modelo pode ser uma das melhores alternativas. Sendo de fávil compreensão e implementação, vale a pena comparar os resultados do Naïves Bayes. Obviamente, caso a relação entre as features sejam importantes, o algorirmo Naïves Bayes não é o mais indicado, pois pode falhar na predição de uma nova informação.

Fica a sugestão de aplicar esse modelo em outras bases disponibilizadas na Internet, bastando fazer alguns ajustes caso seja necessário. Todo o código e mais um pouco está disponível no meu <a href="https://github.com/theavila/tutoriaisML">GitHub</a>.

Os passos de execução deste tutorial foram testados com `Python 3.6` e tudo ocorreu sem problemas. No entanto, é possível que alguém encontre alguma dificuldade ou erro no meio do caminho. Se for o caso, por favor comente a sua dificuldade ou erro neste post.