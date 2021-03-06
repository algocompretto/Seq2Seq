from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import csv
import itertools
import os
import random
import re
import unicodedata
from io import open

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')
CORPUS_FILE = ""
CORPUS = os.path.join("data", CORPUS_FILE)


def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)
# printLines(os.path.join(CORPUS, "movie_lines.txt"))


## Criando um arquivo formatado
# Por conveniência, vamos criar um arquivo bem formatado onde cada
# linha contém um par: sentence : response sentence

def loadLines(fileName, fields):
    """
    Divide cada linha do arquivo em um dicionário de campos
    :param fileName:
    :param fields:
    :return:
    """
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extrai campos
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines


def loadConversations(fileName, lines, fields):
    """
    Agrupa campos de linhas de `loadLines` em conversações baseado em `movie_conversations.txt`
    :param fileName:
    :param lines:
    :param fields:
    :return:
    """
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extrai campos
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]

            # Converte string para list
            utterance_id_pattern = re.compile('L[0-9]+')
            lineIds = utterance_id_pattern.findall(convObj['utteranceIDs'])

            # Reajusta linhas
            convObj['lines'] = []
            for lineId in lineIds:
                convObj['lines'].append(lines[lineId])
            conversations.append(convObj)
    return conversations


def extractsSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Itera sobre todas as linhas da conversação
        for i in range(len(conversation["lines"]) - 1):
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i + 1]["text"].strip()
            # Filtra amostras erradas
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


# Definindo caminho para o novo arquivo
datafile = os.path.join(CORPUS, 'formatted_movie_lines.txt')
delimiter = '\t'
delimiter = str(codecs.decode(delimiter, 'unicode_escape'))

# Inicializa dicionário de linhas, lista de conversas e os ids
lines = {}
conversations = []
MOVIE_LINES_FIELDS = ['lineID', 'characterID', 'movieID', 'character', 'text']
MOVIE_CONVERSATIONS_FIELDS = ['character1ID', 'character2ID', 'movieID', 'utteranceIDs']

# Carrega linhas e processa conversas
print('\nProcessando corpus...')
lines = loadLines(os.path.join(CORPUS, 'movie_lines.txt'), MOVIE_LINES_FIELDS)
print("\nCarregando conversas...")
conversations = loadConversations(os.path.join(CORPUS, "movie_conversations.txt"),
                                  lines, MOVIE_CONVERSATIONS_FIELDS)

# Escreve um arquivo .csv
print('\nEscrevendo o novo arquivo formatado...')
with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter,
                        lineterminator='\n')

    for pair in extractsSentencePairs(conversations):
        writer.writerow(pair)

# Imprime algumas linhas
print('\nAmostragem do arquivo:')

printLines(datafile)

# Carrega e recorta dados

# Tokens base
PAD_token = 0  # Usado para sentenças curtas
SOS_token = 1  # Token início de frase
EOS_token = 2  # Token fim de frase

class Voc:
    """
    A próxima etapa é carregar o par query/response em memória
    Como estamos lidando com sequência de palavras, vamos criar uma class que mapeia
    as palavras com índices, um mapeamento de índice->palavra, contagem de cada palavra e contagem total.
    """

    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print(f"keep_words {len(keep_words)}/{len(self.word2index)} = {len(keep_words) / len(self.word2index):.4f}")

        # Reinicializa dicionários
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: 'PAD', SOS_token: 'SOS', EOS_token: 'EOS'}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)


MAX_LENGTH = 10  # Tamanho máximo de frase

# Converte a string em Unicode para ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Transforma para caixa baixa, remove espaços excessivos, e remove caracteres que não são letras
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ",s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


# Lê o par query/response e retorna o objeto VOC
def readVocs(datafile, corpus_name):
    print('Lendo linhas...')
    # Lê o arquivo e separa em linhas
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')

    # Divide cada linha em pares e normaliza
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs


# Retorna True se as sentenças estão dentro do limiar MAX_LENGTH
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# Filtra os pares de acordo com a condição
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# Usando as funções definidas para popular o objeto VOC e a lista de pares
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print('Começando o preparo dos dados de treino...')
    voc, pairs = readVocs(datafile, corpus_name)
    print(f'Lendo {len(pairs)} pares.')
    pairs = filterPairs(pairs)
    print(f'Recortado para {len(pairs)} pares.')
    print('Contando palavras...')
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print('Palavras contadas: ', voc.num_words)
    return voc, pairs


# Carrega e monta voc e pares
save_dir = os.path.join('data', 'save')
voc, pairs = loadPrepareData(CORPUS, CORPUS_FILE, datafile, save_dir)

# Imprime alguns pares para validar
print('\n pares:')
for pair in pairs[:10]:
    print(pair)


# Vamos remover palavras raras do vocabulário
MIN_COUNT = 3  # threshold

def trimRareWords(voc, pairs, MIN_COUNT):
    # Recorta as palavras
    voc.trim(MIN_COUNT)
    # Filtra os pares com as palavras removidas
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True

        # Confere a frase de entrada
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        if keep_input and keep_output:
            keep_pairs.append(pair)

        print(f'Do total de {len(pairs)} pares, foram recortados {len(keep_pairs)}. Cerca de {len(keep_pairs)/len(pairs):.4f}% do total')

        return keep_pairs


pairs = trimRareWords(voc, pairs, MIN_COUNT)


# Preparing data for models
# Convertemos as frases em tensores
def indexesFromSentences(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Retorna o tensor e os comprimentos da sequência de entrada preenchida
def inputVar(l, voc):
    indexes_batch = [indexesFromSentences(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Retorna o tensor de sequência de destino preenchido, máscara de preenchimento e comprimento máximo do destino
def outputVar(l, voc):
    indexes_batch = [indexesFromSentences(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x:len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


# Exemplo para validação
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable , lengths, target_variable, mask, max_target_len = batches

print("Variável de entrada:", input_variable)
print("Tamanho:", lengths)
print("Variável alvo:", target_variable)
print("Máscara:", mask)
print("Tamanho máximo:", max_target_len)


# Modelo Seq2Seq
# https://jeddy92.github.io/JEddy92.github.io/ts_seq2seq_intro/
# O objetivo de um modelo seq2seq é pegar uma sequência de comprimento variável como
# entrada e retornar uma sequência de comprimento variável como saída usando um modelo de tamanho fixo.
# Uma RNN atua como encoder e a outra RNN atua como decoder.

# Encoder
class EncoderRNN(nn.Module):
    """
    O codificador RNN itera através da frase de entrada um token (por exemplo, palavra) de cada vez,
    em cada etapa de tempo emitindo um vetor de "saída" e um vetor de "estado oculto".
    O vetor de estado oculto é então passado para a próxima etapa de tempo, enquanto o vetor de saída é registrado.
    O codificador transforma o contexto visto em cada ponto da sequência em um conjunto de pontos em um espaço de
    alta dimensão, que o decodificador usará para gerar uma saída significativa para a tarefa dada.

    https://colah.github.io/posts/2015-09-NN-Types-FP/

    :param: input_seq = Lote de frases com formato (max_length, batch_size)
    :param: input_lengths = Lista de comprimentos de frase correspondentes a cada frase do lote, formato = (batch_size)
    :param: hidden = hidden state, formato = (n_layers * num_directions, batch_size, hidden_size)
    :return: outputs = recursos de saída da última camada oculta do GRU (soma das saídas bidirecionais);
    forma = (max_length, batch_size, hidden_size)
    :return: hidden = estado oculto atualizado de GRU; forma = (n_layers * num_directions, batch_size, hidden_size)
    """

    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Converta índices de palavras em embeddings
        embedded = self.embedding(input_seq)
        # Pacote de lote preenchido de sequências para o módulo RNN
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Passe para a frente através de GRU
        outputs, hidden = self.gru(packed, hidden)
        # Descompacte o preenchimento
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Soma de saídas GRU bidirecionais
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # Saída de retorno e estado oculto final
        return outputs, hidden

# Camada de atenção de Luong
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, 'não é um método de atenção legítimo!')
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size*2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1),
                                      encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calcula os pesos da atenção (energia) baseado no método
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        attn_energies = attn_energies.t()

        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size,
                 n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.droput = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers==1 else dropout))
        self.concat = nn.Linear(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)


    def forward(self, input_step, last_hidden, encoder_outputs):
        # Uma palavra por vez
        # Embedding da palavra atual

        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calcula os pesos da atenção
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiplica a energia aos outputs do encoder
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatena as energias e o output do GRU
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # Prevê próxima palavra usando equação de Luong
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)

        # Retorna output e o estado final
        return output, hidden


# Definição dos procedimentos para treino
def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1,1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(DEVICE)
    return loss, nTotal.item()

# Training iteration
def train(input_variable, lengths, target_variable, mask,
          max_target_len, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer,
          batch_size, clip, max_length=MAX_LENGTH):
    """
    1. Passe para frente todo o lote de entrada através do codificador.
    2. Inicialize as entradas do decodificador como SOS_token e o estado oculto como o estado oculto final do codificador.
    3. Encaminhe a sequência de lote de entrada por meio do decodificador, uma etapa de cada vez.
    4. Se o teacher forçar: defina a próxima entrada do decodificador como o alvo atual; else: define a próxima entrada do decodificador como saída do decodificador atual.
    5. Calcule e acumule perdas.
    6. Execute a retropropagação.
    7. Clip gradientes.
    8. Atualize os parâmetros do codificador e do modelo do decodificador.
    """

    # Zera gradientes
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Envia para device
    input_variable = input_variable.to(DEVICE)
    target_variable = target_variable.to(DEVICE)
    mask = mask.to(DEVICE)
    # Lengths for rnn packing should always be on the cpu
    lengths = lengths.to("cpu")

    # Inicializa variáveis
    loss = 0
    print_losses = []
    n_totals = 0

    # Manda pro encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Crie a entrada do decodificador inicial (comece com tokens SOS para cada frase)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(DEVICE)

    # Defina o estado oculto do decodificador inicial para o estado oculto final do codificador
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine se estamos usando o teacher forcing esta iteração
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Encaminhar lote de sequências por meio do decodificador, uma etapa de cada vez
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: próximo input é a target variable
            decoder_input = target_variable[t].view(1, -1)
            # Calcula e acumula loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Sem teacher forcing: Próximo input é o output do próprio decoder
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(DEVICE)
            # Calcula e acumula loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Faz backpropagation
    loss.backward()

    # Gradientes são modificados inplace
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Ajusta os pesos
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer,
               decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers,
               save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):
    """
    Finalmente, é hora de amarrar todo o procedimento de treinamento aos dados.
    A função trainIters é responsável por executar n_iterações de treinamento dados os modelos,
    otimizadores, dados, etc. aprovados. Esta função é bastante autoexplicativa,
    pois fizemos o levantamento de peso com a função trem.
    """


    # Carrega lotes para cada iteração
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    # Inicializações
    print('Inicializando ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Loop de treino
    print("Treinando a rede neural...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Percorre a iteração do treino
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # Imprime o progresso
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteração: {}; Porcentagem completa: {:.1f}%; Função de perda: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

# Greedy Decoding, para cada time step, escolhemos o output com maior
# valor softmax

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward da entrada
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepara a última hidden layer do encoder para ser a primeira hidden entrada no decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        decoder_input = torch.ones(1, 1, device=DEVICE, dtype=torch.long) * SOS_token
        # Inicializa tensores para registrarem as palavras decodadas
        all_tokens = torch.zeros([0], device=DEVICE, dtype=torch.long)
        all_scores = torch.zeros([0], device=DEVICE)

        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                          decoder_hidden,
                                                          encoder_outputs)

            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)

            # Adiciona mais uma dimensão para próximo token
            decoder_input = torch.unsqueeze(decoder_input, 0)

        return all_tokens, all_scores


# Avaliação do texto
def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    # Formata a entrada
    # words -> indexes
    indexes_batch = [indexesFromSentences(voc, sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpõe as dimensões
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(DEVICE)
    lengths = lengths.to('cpu')

    tokens, scores = searcher(input_batch, lengths, max_length)

    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]

    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while True:
        try:
            input_sentence = input('> ')
            if input_sentence == 'q' or input_sentence == 'SAIR': break

            input_sentence = normalizeString(input_sentence)

            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)

            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Skynet:', " ".join(output_words))

        except KeyError:
            print('Erro: Encontrou palavra inesperada.')


# Configura modelo
model_name = 'cb_model'
attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Configura checkpoint
loadFilename = None
checkpoint_iter = 4000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Carrega modelo se já existe
if loadFilename:
    # Se o modelo está na máquina local
    checkpoint = torch.load(loadFilename)
    # GPU ou CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Construindo encoder e decoder ...')
# Inicializa word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Inicializa modelos de encoder e decoder
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size,
                              voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

encoder = encoder.to(DEVICE)
decoder = decoder.to(DEVICE)
print('Modelos construídos e prontos para o papo!')


# Configura parâmetros
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 8000
print_every = 1
save_every = 500

# Configura as camadas para o modo treino
encoder.train()
decoder.train()

# Inicializa otimizadores
print('Construindo otimizadores...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)


for state in encoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

for state in decoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

# Começa a iteração
print("Começando o treinamento do modelo!")
trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, CORPUS_FILE, loadFilename)


# Modo de avaliação
encoder.eval()
decoder.eval()

# Inicializa o searcher
searcher = GreedySearchDecoder(encoder, decoder)

# PAPO ROLANDO SOLTO
evaluateInput(encoder, decoder, searcher, voc)