# Modelos Seq2Seq
<p align="center">
  <img src="https://c.tenor.com/T6U1_ahsOEcAAAAC/star-wars-c3po.gif" alt="C3PO saying - 'We are doomed.'" />
</p>

Modelos sequence to sequence são arquiteturas de Redes Neurais Recorrentes (RNNs) que se mostram extremamente úteis para resolver problemas de tradução, resposta a perguntas e até mesmo _chatbots_.

> Redes Neurais Recorrentes são apenas redes neurais configuradas de tal modo que, resultados do passado influenciam o que a rede fará em situações futuras - como se fosse uma memória.

Como o nome sugere, o modelo recebe uma sequência de palavras/sentenças e gera uma saída contendo outra sequência de palavras. É um método baseado em codificar-decodificar que mapeia uma entrada para uma saída com uma _tag_ e um _valor de atenção_. A ideia é basicamente usar 2 RNNs que trabalharão em conjunto com um token e tentarão prever a próxima sequência de estado a partir da primeira.
<p align="center">
  <img src="https://miro.medium.com/max/1372/1*3lj8AGqfwEE5KCTJ-dXTvg.png" alt="Encoder-decoder representation" />
</p>
De modo geral, um modelo encoder-decoder pode ser imaginado como dois blocos:

- O encoder processa cada token da entrada. Ele busca extrair toda informação sobre a sequência de entrada em um vetor de tamanho fixo - chamado de vetor de contexto. O **vetor de contexto** é construído de tal maneira que é esperado encapsular todo o significado da frase de entrada, no intuito de gerar uma predição acurada.

- O decoder lê o vetor de contexto e tenta prever a frase token por token.

Referências:
https://medium.com/analytics-vidhya/encoder-decoder-seq2seq-models-clearly-explained-c34186fbf49b
https://www.guru99.com/seq2seq-model.html
https://towardsdatascience.com/day-1-2-attention-seq2seq-models-65df3f49e263#:~:text=A%20Seq2Seq%20model%20is%20a,outputs%20another%20sequence%20of%20items.&text=In%20the%20case%20of%20Neural,the%20translated%20series%20of%20words.
https://towardsdatascience.com/sequence-to-sequence-model-introduction-and-concepts-44d9b41cd42d

