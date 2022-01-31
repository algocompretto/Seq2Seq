# Modelos Seq2Seq
<p align="center">
  <img src="https://c.tenor.com/T6U1_ahsOEcAAAAC/star-wars-c3po.gif" alt="C3PO saying - 'We are doomed.'" />
</p>

Modelos sequence to sequence são arquiteturas de Redes Neurais Recorrentes (RNNs) que se mostram extremamente úteis para resolver problemas de tradução, resposta a perguntas e _chatbots_.

> Redes Neurais Recorrentes são apenas redes neurais configuradas de tal modo que, resultados do passado influenciam o que a rede fará em situações futuras - funcionando como uma espécie de memória.

Como o nome sugere, o modelo recebe uma sequência de palavras/sentenças e gera uma saída contendo outra sequência de palavras. É um método baseado em codificar-decodificar que mapeia uma entrada para uma saída com uma _tag_ e um _valor de atenção_. A ideia é basicamente usar 2 RNNs que trabalharão em conjunto com um token e tentarão prever a próxima sequência de estado a partir da primeira.
<p align="center">
  <img src="https://miro.medium.com/max/1372/1*3lj8AGqfwEE5KCTJ-dXTvg.png" alt="Encoder-decoder representation" />
</p>
De modo geral, um modelo encoder-decoder pode ser imaginado como dois blocos:

- O encoder processa cada token da entrada. Ele busca extrair toda informação sobre a sequência de entrada em um vetor de tamanho fixo - chamado de vetor de contexto. O **vetor de contexto** é construído de tal maneira que é esperado encapsular todo o significado da frase de entrada, no intuito de gerar uma predição acurada.

- O decoder lê o vetor de contexto e tenta prever a frase token por token.

### Problemas de modelagem de sequências
Problemas de modelagem de sequências se referem a problemas no qual tanto a entrada quanto a saída. Considere um simples problema de crítica de filme - nesse caso, a entrada é uma sequência de palavras e a saída é algum número entre 0 e 1. Usando redes neurais profundas comuns, teríamos que codificar o nosso texto de entrada como um vetor usando _Word2Vec_, _BOW_, etc. Contudo, usando esses métodos, a sequência de palavras não é preservada, logo o valor semântico é perdido.

Assim, para resolver esse problema, as RNNs se mostram mais eficiente. Em essência, para qualquer entrada X = (x₀, x₁, x₂, ... xₜ) com um número variável de recursos, a cada passo de tempo, uma célula RNN pega um item / token xₜ como entrada e produz uma saída hₜ enquanto passa alguns informações para a próxima etapa de tempo. Essas saídas podem ser usadas de acordo com o problema em questão.


### Referências:
[Sequence to Sequence Learning with Neural Networks by Ilya Sutskever, et al](https://arxiv.org/abs/1409.3215)
