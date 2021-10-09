from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')
CORPUS_FILE = "cornell movie-dialogs corpus"
CORPUS = os.path.join('data', CORPUS_FILE)


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
    with open(fileName, 'r', encoding='iso-8559-1') as f:
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
            convObj['lines'] = {}
            for lineId in lineIds:
                convObj['lines'].append(lines[lineId])
            conversations.append(convObj)
    return conversations


def extractsSentencePairs(conversations):
    """
    Extrai pares de frases a partir das `conversations`
    :param conversations:
    :return:
    """
    qa_pairs = []
    for conversation in conversations:
        # Itera sobre todas as linhas da conversação
        for i in range(len(conversation['lines']) - 1):
            inputLine = conversation['lines'][i]['text'].strip()
            targetLine = conversation['lines'][i+1]['text'].strip()
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
lines = loadLines(os.path.join(CORPUS, 'movie_conversations.txt'),
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
    lines = open(datafile, enconding='utf-8').\
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