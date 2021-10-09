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