from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##### DATA PREPARATION #####
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10 # longest sentence length


'''
Help methods for to build a language object
This method contains two dictionaries: 
word -> index & index -> word, and the count for each word
'''
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.wordIndex = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.wordIndex
            self.word2count[word] = 1
            self.index2word[self.wordIndex] = word
            self.wordIndex += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Make all letters lower case, delete non-alphabet chars
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip()) # lower & remove whitespace, both sides
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

'''
This method is to read data files.
First segment the file into lines, then segment the lines into pairs.
Since all data files are English -> other language, we add 'reverse' to create other lang -> English
'''
def readLangs(language1, language2, reverse=False):
    print("Reading lines...")

    # seg the lines
    lines = open('./data/%s-%s.txt' % (language1, language2), encoding='utf-8'). \
        read().strip().split('\n')

    # seg the pairs
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # reverse 
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(language2)
        output_lang = Lang(language1)
    else:
        input_lang = Lang(language1)
        output_lang = Lang(language2)

    return input_lang, output_lang, pairs



'''
Filter the sentences
'''
MAX_LENGTH = 10

# TO BE MODIFIED!! right now only consider those sentences which begin with following prefixes
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
           p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

'''
Return the sentence in the other language for the input sentence, i.e. reference
'''
def getPairSentence(input_sentence):
    return 

'''
The entire process to prepare the data:
1. read files and segment them into lines, segment the lines into pairs
2. normalize the files, and filter the sentences according to length and content
3. create a word list from those filtered and paired sentences
'''
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.wordIndex)
    print(output_lang.name, output_lang.wordIndex)
    return input_lang, output_lang, pairs





####  SEQ2SEQ MODEL ####

'''
Encoder:
seq2seq encoder is an RNN, it outputs some value for ech word in input sentence
For every input word, encoder outputs a vector and a hidden state, and use the hidden
state for next input word
'''
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)  # dimension adjusted to 1*1*n
        output = embedded
        output, hidden = self.gru(output, hidden)  # derive output and hidden states for each GRU for subsequent attention calculation
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

e = EncoderRNN(10, 256)
print(e)


'''
Decoder:
Decoder is another RNN, used to tranlate the input sequence
'''
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


'''
Attention decoder
'''
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)  # fully connected layer
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # embedding input
        embedded = self.embedding(input).view(1, 1, -1)
        # use dropout to prevent overfitting
        embedded = self.dropout(embedded)
        # calculate the attention weights
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # multiply attention weights and encoder outputs
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        # concatenate the embedding layer with the attention layer in dim = 1
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        # add a fully connected layer, and squeeze the dim to 0
        output = self.attn_combine(output).unsqueeze(0)
        # activation function
        output = F.relu(output)
        # GRU
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

d = AttnDecoderRNN(256, 10)
print(d)







#### EVALUATION ####

# Return the index sequence from the sentence input.
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


# Create a tensor from the index sequence, and add a EOS token at the end of the sequence.
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


# Input tensor is the index of word in the input sentence,
# output tensor is the index of word in the output sentence.
def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


# Evaluate the model
def evaluate(input_lang, output_lang, encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


# Randomly choose n sentences from the dataset to test
def evaluateRandomly(input_lang, output_lang, pairs, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('input:', pair[0])
        print('target:', pair[1])
        output_words, attentions = evaluate(input_lang, output_lang, encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('predict', output_sentence)
        BLEU(pair[1], output_words) 
        print('')


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()

# Bilingual Evaluation Understudy
def BLEU(reference, candidate):
    score = sentence_bleu(reference, candidate)
    print("BLEU score: ", score)