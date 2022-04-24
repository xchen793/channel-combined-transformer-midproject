# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# do not pip install torchtext

# https://blog.csdn.net/u010366748/article/details/111269231


# GRU + seq2seq , attention
# https://blog.csdn.net/qq_42714262/article/details/119298940
# Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
# https://imzhanghao.com/2021/08/26/encoder-decoder/

# https://www.programcreek.com/python/example/107681/torch.nn.GRU

# https://github.com/shaojiawei07/BottleNetPlusPlus
# https://blog.csdn.net/weixin_38544305/article/details/115603014

# https://github.com/shaojiawei07/BottleNetPlusPlus/blob/master/BottleNet%2B%2B_ResNet50.py

import torch
import torchtext
from torchtext.legacy import data, datasets  
# for torchtext v0.12, in order to use data.Filed, data has to be imported here from torchtext.legacy even they claimed that they deleted legacy

from sklearn.model_selection import train_test_split
import random
import re
from tqdm import tqdm  
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import unicodedata
import datetime
import time
import copy
import os

print(os.getcwd())
print(torchtext.__version__)

ngpu = 1

use_cuda = torch.cuda.is_available()  
device = torch.device("cuda:0" if (use_cuda and ngpu > 0) else "cpu")
print('device=', device)


# data_dir_bg = "./data/bg.txt"
# data_dir_en = "./data/en.txt"

# data_df_bg = pd.read_csv(data_dir_bg,  # Bulgarian from https://www.statmt.org/europarl/
#                       encoding='UTF-8', sep='\n', header=None,
#                       names=['bg'], index_col=False,nrows = 400912)

# print(data_df_bg.shape)
# print(data_df_bg.values.shape)
# print(data_df_bg.values[0])
# print(data_df_bg.values[0].shape)
# data_df_bg.head()

# data_df_en = pd.read_csv(data_dir_en,  # English from https://www.statmt.org/europarl/
#                       encoding='UTF-8', sep='\n', header=None,
#                       names=['en'], index_col=False)
# print(data_df_en.shape)
# print(data_df_en.values.shape)
# print(data_df_en.values[0])
# print(data_df_en.values[0].shape)
# data_df_en.head()


# data_df_merge = pd.concat([data_df_bg, data_df_en], axis=1)  # Merge bg & en dataset in row contact
# print(data_df_merge.shape)
# print(data_df_merge.values.shape)
# print(data_df_merge.values[0])
# print(data_df_merge.values[0].shape)
# data_df_merge.head()

data_df = pd.read_csv('./data/eng-fra.txt',  
                      encoding='UTF-8', sep='\t', header=None,
                      names=['eng', 'fra'], index_col=False)

print(data_df.shape)
print(data_df.values.shape)
print(data_df.values[0])
print(data_df.values[0].shape)
data_df.head()




# data preprocessing

# unicode str 2 acsii：
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


# string regularization 
def normalizeString(s):
    s = s.lower().strip()
    s = unicodeToAscii(s)
    s = re.sub(r"([.!?])", r" \1", s)  
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  # replace any non alphabet char and non .?! to space
    s = re.sub(r'[\s]+', " ", s)  # replce all spaces to only one space：w='abc  1   23  1' 处理后：w='abc 1 23 1'
    return s


print(normalizeString("(The sitting was opened at 9.35 a.m.)"))
print(normalizeString("(Die Sitzung wird um 9.35 Uhr eröffnet.)"))

MAX_LENGTH = 30

eng_prefixes = (  
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

pairs = [[normalizeString(s) for s in line] for line in data_df.values]

print('pairs num=', len(pairs))
print(pairs[0])
print(pairs[1])



#for fast training, only sentence less than 10 words will be used
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and \
           p[0].startswith(eng_prefixes)  # startswith first arg must be str or a tuple of str


def filterPairs(pairs):
    return [[pair[1], pair[0]] for pair in pairs if filterPair(pair)]

pairs = filterPairs(pairs)
print('after trimming, pairs num=', len(pairs))
print(pairs[0])
print(pairs[1])
print(random.choice(pairs))

train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=1234)

print(len(train_pairs))
print(len(val_pairs))

tokenizer = lambda x: x.split() 

SRC_TEXT = data.Field(sequential=True,
                                tokenize=tokenizer,
                                # lower=True,
                                fix_length=MAX_LENGTH + 2,
                                preprocessing=lambda x: ['<start>'] + x + ['<end>'],
                                # after tokenizing but before numericalizing
                                # postprocessing # after numericalizing but before the numbers are turned into a Tensor
                                )
TARG_TEXT = data.Field(sequential=True,
                                 tokenize=tokenizer,
                                 # lower=True,
                                 fix_length=MAX_LENGTH + 2,
                                 preprocessing=lambda x: ['<start>'] + x + ['<end>'],
                                 )


def get_dataset(pairs, src, targ):
    fields = [('src', src), ('targ', targ)]  # filed信息 fields dict[str, Field])
    examples = []  # list(Example)
    for fra, eng in tqdm(pairs): # visualize
        # to create Example,field.preprocess will be used
        examples.append(data.Example.fromlist([fra, eng], fields))
    return examples, fields


examples, fields = get_dataset(pairs, SRC_TEXT, TARG_TEXT)

ds_train = data.Dataset(*get_dataset(train_pairs, SRC_TEXT, TARG_TEXT))
ds_val = data.Dataset(*get_dataset(val_pairs, SRC_TEXT, TARG_TEXT))

print(len(ds_train[0].src), ds_train[0].src)
print(len(ds_train[0].targ), ds_train[0].targ)

print(len(ds_val[100].src), ds_val[100].src)
print(len(ds_val[100].targ), ds_val[100].targ)
# construct dictionary
SRC_TEXT.build_vocab(ds_train)  # construct SRC_TEXT & mapping between token and ID
print(len(SRC_TEXT.vocab))
print(SRC_TEXT.vocab.itos[0])
print(SRC_TEXT.vocab.itos[1])
print(SRC_TEXT.vocab.itos[2])
print(SRC_TEXT.vocab.itos[3])
print(SRC_TEXT.vocab.stoi['<start>'])
print(SRC_TEXT.vocab.stoi['<end>'])

# try decode
res = []
for id in [  3,   5,   6,  71,  48,   5,   8,  32, 743,   4,   2,   1]:
    res.append(SRC_TEXT.vocab.itos[id])
print(' '.join(res)+'\n')

TARG_TEXT.build_vocab(ds_train)
print(len(TARG_TEXT.vocab))
print(TARG_TEXT.vocab.itos[0])
print(TARG_TEXT.vocab.itos[1])
print(TARG_TEXT.vocab.itos[2])
print(TARG_TEXT.vocab.itos[3])
print(TARG_TEXT.vocab.stoi['<start>'])
print(TARG_TEXT.vocab.stoi['<end>'])

BATCH_SIZE = 64

# construct iterator
train_iter, val_iter = data.Iterator.splits(
    (ds_train, ds_val),
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    batch_sizes=(BATCH_SIZE, BATCH_SIZE)
)

# check iteration, postprocessing might be triggered
for batch in train_iter:
    # seq_len is the first pair, pair 0
    print(batch.src[:,0])
    print(batch.src.shape, batch.targ.shape)  # [12,64], [12,64]
    break

class DataLoader:
    def __init__(self, data_iter):
        self.data_iter = data_iter
        self.length = len(data_iter)  # total pair

    def __len__(self):
        return self.length

    def __iter__(self):
        for batch in self.data_iter:
            yield (torch.transpose(batch.src, 0, 1), torch.transpose(batch.targ, 0, 1))


train_dataloader = DataLoader(train_iter)
val_dataloader = DataLoader(val_iter)

print('len(train_dataloader):', len(train_dataloader))  # 34 step/batch
for batch_src, batch_targ in train_dataloader:
    print(batch_src.shape, batch_targ.shape)  # [256,12], [256,12]
    print(batch_src[0], batch_src.dtype)
    print(batch_targ[0], batch_targ.dtype)
    break

# angle calculation: pos * 1/(10000^(2i/d))
def get_angles(pos, i, d_model):
    # 2*(i//2)ensures2i，this part calculate 1/10000^(2i/d)
    angle_rates = 1 / np.power(10000, 2 * (i // 2) / np.float32(d_model))  # => [1, 512]
    return pos * angle_rates  # [50,1]*[1,512]=>[50, 512]


# np.arange() returns, for example[1,2,3,4,5]，start is 1，end is 6，step is 1
# start=1,end=6，[1,2,3,4,5], 6 is not included

def positional_encoding(position, d_model): 
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],  # [50, 1]
                            np.arange(d_model)[np.newaxis, :],  # [1, d_model=512]
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # 2i
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # 2i+2

    pos_encoding = angle_rads[np.newaxis, ...]  # [50,512]=>[1,50,512]
    return torch.tensor(pos_encoding, dtype=torch.float32)

pos_encoding = positional_encoding(50, 512)
print(pos_encoding.shape) # [1,50,512]

from matplotlib import pyplot as plt

def draw_pos_encoding(pos_encoding):
    plt.figure()
    plt.pcolormesh(pos_encoding[0], cmap='RdBu') # plot classification 
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar() # plot bar figure with color
    plt.savefig('pos_encoding.png')
    plt.show()

draw_pos_encoding(pos_encoding)


pad = 1 # important！
def create_padding_mask(seq):  # seq [b, seq_len]
    # seq = torch.eq(seq, torch.tensor(0)).float() # check pad=0
    seq = torch.eq(seq, torch.tensor(pad)).float()  # pad!=0
    return seq[:, np.newaxis, np.newaxis, :]  # =>[b, 1, 1, seq_len]

x = torch.tensor([[7, 6, 0, 0, 1],
                  [1, 2, 3, 0, 0],
                  [0, 0, 0, 4, 5]])
print(x.shape) # [3,5]
print(x)
mask = create_padding_mask(x)
print(mask.shape, mask.dtype) # [3,1,1,5]
print(mask)

# torch.triu(tensor, diagonal=0) calculate uper diag，diagonal 0 default is that middle one
# diagonal>0，No. n diag above that middle one
# diagonal<0，No. n diag under that middle one
def create_look_ahead_mask(size):  # seq_len
    mask = torch.triu(torch.ones((size, size)), diagonal=1)
    # mask = mask.device() #
    return mask  # [seq_len, seq_len]

x = torch.rand(1,3)
print(x.shape)
print(x)
mask = create_look_ahead_mask(x.shape[1])
print(mask.shape, mask.dtype)
print(mask)


def scaled_dot_product_attention(q, k, v, mask=None):

    matmul_qk = torch.matmul(q, k.transpose(-1, -2))  # 矩阵乘 =>[..., seq_len_q, seq_len_k]


    dk = torch.tensor(k.shape[-1], dtype=torch.float32)  # depth_k
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)  # [..., seq_len_q, seq_len_k]

    if mask is not None:  # mask: [b, 1, 1, seq_len]

        scaled_attention_logits += (mask * -1e9)


    attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)  # [..., seq_len_q, seq_len_k]

    output = torch.matmul(attention_weights, v)  # =>[..., seq_len_q, depth_v]
    return output, attention_weights  # [..., seq_len_q, depth_v], [..., seq_len_q, seq_len_k]

def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)

np.set_printoptions(suppress=True) 



class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0  

        self.depth = d_model // self.num_heads  # 512/8=64

        self.wq = torch.nn.Linear(d_model, d_model)
        self.wk = torch.nn.Linear(d_model, d_model)
        self.wv = torch.nn.Linear(d_model, d_model)

        self.final_linear = torch.nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):  # x [b, seq_len, d_model]
        x = x.view(batch_size, -1, self.num_heads,
                   self.depth)  # [b, seq_len, d_model=512]=>[b, seq_len, num_head=8, depth=64]
        return x.transpose(1, 2)  # [b, seq_len, num_head=8, depth=64]=>[b, num_head=8, seq_len, depth=64]

    def forward(self, q, k, v, mask):  # q=k=v=x [b, seq_len, embedding_dim] embedding_dim其实也=d_model
        batch_size = q.shape[0]

        q = self.wq(q)  # =>[b, seq_len, d_model]
        k = self.wk(k)  # =>[b, seq_len, d_model]
        v = self.wv(v)  # =>[b, seq_len, d_model]

        q = self.split_heads(q, batch_size)  # =>[b, num_head=8, seq_len, depth=64]
        k = self.split_heads(k, batch_size)  # =>[b, num_head=8, seq_len, depth=64]
        v = self.split_heads(v, batch_size)  # =>[b, num_head=8, seq_len, depth=64]

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # => [b, num_head=8, seq_len_q, depth=64], [b, num_head=8, seq_len_q, seq_len_k]

        scaled_attention = scaled_attention.transpose(1, 2)  # =>[b, seq_len_q, num_head=8, depth=64]

        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)  # =>[b, seq_len_q, d_model=512]

        output = self.final_linear(concat_attention)  # =>[b, seq_len_q, d_model=512]
        return output, attention_weights  # [b, seq_len_q, d_model=512], [b, num_head=8, seq_len_q, seq_len_k]


def point_wise_feed_forward_network(d_model, dff):
    feed_forward_net = torch.nn.Sequential(
        torch.nn.Linear(d_model, dff),  # [b, seq_len, d_model]=>[b, seq_len, dff=2048]
        torch.nn.ReLU(),
        torch.nn.Linear(dff, d_model),  # [b, seq_len, dff=2048]=>[b, seq_len, d_model=512]
    )
    return feed_forward_net


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)  # （padding mask）(self-attention)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)

    # x [b, inp_seq_len, embedding_dim] embedding_dim=d_model
    # mask [b,1,1,inp_seq_len]
    def forward(self, x, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # =>[b, seq_len, d_model]
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  #  =>[b, seq_len, d_model]

        ffn_output = self.ffn(out1)  # =>[b, seq_len, d_model]
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  #  =>[b, seq_len, d_model]

        return out2  

class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model,
                                       num_heads)  # （look ahead mask 和 padding mask）(self-attention)
        self.mha2 = MultiHeadAttention(d_model, num_heads)  # （padding mask）(encoder-decoder attention)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm3 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)
        self.dropout3 = torch.nn.Dropout(rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x,
                                               look_ahead_mask)  # =>[b, targ_seq_len, d_model], [b, num_heads, targ_seq_len, targ_seq_len]
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)  # residual normalization [b, targ_seq_len, d_model]

        # Q: receives the output from decoder's first attention block，即 masked multi-head attention sublayer
        # K V: V (value) and K (key) receive the encoder output as inputs
        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output,
                                               padding_mask)  # =>[b, targ_seq_len, d_model], [b, num_heads, targ_seq_len, inp_seq_len]
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)  # 残差&层归一化 [b, targ_seq_len, d_model]

        ffn_output = self.ffn(out2)  # =>[b, targ_seq_len, d_model]
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)  # 残差&层归一化 =>[b, targ_seq_len, d_model]

        return out3, attn_weights_block1, attn_weights_block2
        # [b, targ_seq_len, d_model], [b, num_heads, targ_seq_len, targ_seq_len], [b, num_heads, targ_seq_len, inp_seq_len]


class Encoder(torch.nn.Module):
    def __init__(self,
                 num_layers,  
                 d_model,
                 num_heads,
                 dff,  
                 input_vocab_size,  
                 maximun_position_encoding,
                 rate=0.1):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = torch.nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=d_model)
        # self.gru= torch.nn.GRU(emb, hid ,batch_first=True)
        self.pos_encoding = positional_encoding(maximun_position_encoding,
                                                d_model) 
        self.enc_layers = torch.nn.ModuleList([EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])

        self.dropout = torch.nn.Dropout(rate)

    def forward(self, x, mask):
        inp_seq_len = x.shape[-1]

        # adding embedding and position encoding
        x = self.embedding(x)  # [b, inp_seq_len]=>[b, inp_seq_len, d_model]
        # h_n = self.gru()
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        pos_encoding = self.pos_encoding[:, :inp_seq_len, :]
        pos_encoding = pos_encoding.cuda()  # ###############
        x += pos_encoding  # [b, inp_seq_len, d_model]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)  # [b, inp_seq_len, d_model]=>[b, inp_seq_len, d_model]
        return x  # [b, inp_seq_len, d_model]

# class BEC_AWGN(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, x, p=0.2):
#         x_tmp = torch.round(x * 256)
#         #x_tmp = x_tmp.int()
#         x_tmp = x_tmp.byte()

#         p_complement = 1-p

#         std = x

#         binomial_noise = np.random.binomial(1,p_complement,(std.size())).astype(np.uint8) * 1 + \
#         np.random.binomial(1,p_complement,(std.size())).astype(np.uint8) * 2 + \
#         np.random.binomial(1,p_complement,(std.size())).astype(np.uint8) * 4 + \
#         np.random.binomial(1,p_complement,(std.size())).astype(np.uint8) * 8 + \
#         np.random.binomial(1,p_complement,(std.size())).astype(np.uint8) * 16 + \
#         np.random.binomial(1,p_complement,(std.size())).astype(np.uint8) * 32 + \
#         np.random.binomial(1,p_complement,(std.size())).astype(np.uint8) * 64 + \
#         np.random.binomial(1,p_complement,(std.size())).astype(np.uint8) * 128

#         binomial_noise = torch.ByteTensor(binomial_noise).to(device)

#         x_tmp_filter = x_tmp & binomial_noise
#         x_tmp_filter = x_tmp_filter.float()
        
#         x_tmp_filter = x_tmp_filter + (255.0 - binomial_noise.float()) / 2.0
#         x_tmp_filter /= 255.0

#         return x_tmp_filter

#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input = grad_output.clone()
#         return grad_input, None

# class BSC(torch.autograd.Function):
#     pass

# class Channel():
#     def __init__(self,  input_channel=256, hidden_channel=128, noise=10, channel = 1,spatial = 0):
#         super(compression_module, self).__init__()

        

#         self.conv1 = nn.Conv2d(input_channel+1,hidden_channel,kernel_size = 3,stride=1,padding=1)
#         self.conv2 = nn.Conv2d(hidden_channel,input_channel,kernel_size = 3,stride=1,padding=1)

#         self.batchnorm1 = nn.BatchNorm2d(hidden_channel)
#         self.batchnorm2 = nn.BatchNorm2d(input_channel)
        
#         self.conv3 = nn.Conv2d(input_channel+1,hidden_channel,kernel_size=2,stride=2)
#         self.conv4 = nn.ConvTranspose2d(hidden_channel,input_channel,kernel_size=2,stride=2)
        
#         self.noise = noise
#         self.channel = channel
#         self.spatial = spatial

#     def forward(self,x):

#         noise_factor = torch.rand(1) * self.noise

#         if self.channel == 'a':
#             x = awgn_noise(x,noise_factor)
#         elif self.channel == 'e':
#             bec =  BEC.apply
#             x = bec(x,p)
#         elif self.channel == 'w':
#             x = x
#         else:
#             print('error') 

#     def awgn_noise(x, noise_factor):
#         return x + torch.randn_like(x) * noise_factor


class Decoder(torch.nn.Module):
    def __init__(self,
                 num_layers,  # N个encoder layer
                 d_model,
                 num_heads,
                 dff,  
                 target_vocab_size,
                 maximun_position_encoding,
                 rate=0.1):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = torch.nn.Embedding(num_embeddings=target_vocab_size, embedding_dim=d_model)
        # self.gru= torch.nn.GRU(emb, hid ,batch_first=True)
        self.pos_encoding = positional_encoding(maximun_position_encoding,
                                                d_model)  # =>[1, max_pos_encoding, d_model=512]

        # self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate).cuda() for _ in range(num_layers)] 
        self.dec_layers = torch.nn.ModuleList([DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])

        self.dropout = torch.nn.Dropout(rate)

    # x [b, targ_seq_len]
    # look_ahead_mask [b, 1, targ_seq_len, targ_seq_len] 这里传入的look_ahead_mask应该是已经结合了look_ahead_mask和padding mask的mask
    # enc_output [b, inp_seq_len, d_model]
    # padding_mask [b, 1, 1, inp_seq_len]
    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        targ_seq_len = x.shape[-1]

        attention_weights = {}

        # adding embedding and position encoding
        x = self.embedding(x)  # [b, targ_seq_len]=>[b, targ_seq_len, d_model]
        # h_n = self.gru(x)

        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        pos_encoding = self.pos_encoding[:, :targ_seq_len, :]  # [b, targ_seq_len, d_model]
        pos_encoding = pos_encoding.cuda() # ###############
        x += pos_encoding  # [b, inp_seq_len, d_model]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, attn_block1, attn_block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)
            # => [b, targ_seq_len, d_model], [b, num_heads, targ_seq_len, targ_seq_len], [b, num_heads, targ_seq_len, inp_seq_len]

            attention_weights[f'decoder_layer{i + 1}_block1'] = attn_block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = attn_block2

        return x, attention_weights
        # => [b, targ_seq_len, d_model],
        # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
        #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}

sample_decoder = Decoder(num_layers=2,
                         d_model=512,
                         num_heads=8,
                         dff=2048,
                         target_vocab_size=8000,
                         maximun_position_encoding=5000)
sample_decoder = sample_decoder.to(device)



class Transformer(torch.nn.Module):
    def __init__(self,
                 num_layers,  # N个encoder layer
                 d_model,
                 num_heads,
                 dff,  # 
                 input_vocab_size,  #
                 target_vocab_size,  # 
                 pe_input,  # input max_pos_encoding
                 pe_target,  # input max_pos_encoding
                 rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers,
                               d_model,
                               num_heads,
                               dff,
                               input_vocab_size,
                               pe_input,
                               rate)

        # self.channel = Channel(channelOp)

        self.decoder = Decoder(num_layers,
                               d_model,
                               num_heads,
                               dff,
                               target_vocab_size,
                               pe_target,
                               rate)
        self.final_layer = torch.nn.Linear(d_model, target_vocab_size)

    def forward(self, inp, targ, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, enc_padding_mask)  # =>[b, inp_seq_len, d_model]

        dec_output, attention_weights = self.decoder(targ, enc_output, look_ahead_mask, dec_padding_mask)
        # => [b, targ_seq_len, d_model],
        # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
        #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}
        final_output = self.final_layer(dec_output)  # =>[b, targ_seq_len, target_vocab_size]

        return final_output, attention_weights
        # [b, targ_seq_len, target_vocab_size]
        # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
        #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}


sample_transformer = Transformer(num_layers=2,
                                 d_model=512,
                                 num_heads=8,
                                 dff=2048,
                                 input_vocab_size=8500,
                                 target_vocab_size=8000,
                                 pe_input=10000,
                                 pe_target=6000)
sample_transformer = sample_transformer.to(device)

temp_inp = torch.tensor(np.random.randint(low=0, high=200, size=(64, 42))) # [b, inp_seq_len]
temp_targ = torch.tensor(np.random.randint(low=0, high=200, size=(64, 36))) # [b, targ_seq_len]

fn_out, attn = sample_transformer(temp_inp.cuda(), temp_targ.cuda(), None, None, None)
print(fn_out.shape) # [64, 36, 8000]
print(attn['decoder_layer2_block1'].shape) # [64, 8, 36, 36]
print(attn['decoder_layer2_block2'].shape) # [64, 8, 36, 42]


num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = len(SRC_TEXT.vocab) # 3901
target_vocab_size = len(TARG_TEXT.vocab) # 2591
dropout_rate = 0.1

print(input_vocab_size, target_vocab_size)

class CustomSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warm_steps=4):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warm_steps

        super(CustomSchedule, self).__init__(optimizer)

    def get_lr(self):
        """

        arg1 = torch.rsqrt(torch.tensor(self._step_count, dtype=torch.float32))
        arg2 = torch.tensor(self._step_count * (self.warmup_steps ** -1.5), dtype=torch.float32)
        dynamic_lr = torch.rsqrt(self.d_model) * torch.minimum(arg1, arg2)
        """
        # print('*'*27, self._step_count)
        arg1 = self._step_count ** (-0.5)
        arg2 = self._step_count * (self.warmup_steps ** -1.5)
        dynamic_lr = (self.d_model ** (-0.5)) * min(arg1, arg2)
        # print('dynamic_lr:', dynamic_lr)
        return [dynamic_lr for group in self.optimizer.param_groups]

model = sample_transformer
optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
learning_rate = CustomSchedule(optimizer, d_model, warm_steps=4000)

lr_list = []
for i in range(1, 20000):
    learning_rate.step()
    lr_list.append(learning_rate.get_lr()[0])
plt.figure()
plt.plot(np.arange(1, 20000), lr_list)
plt.legend(['warmup=4000 steps'])
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
plt.show()



_model = sample_transformer
_optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
_learning_rate = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

lr_list = []
for i in range(1, 50):
    _learning_rate.step()
    lr_list.append(_learning_rate.get_lr()[0])
plt.figure()
plt.plot(np.arange(1, 50), lr_list)
plt.legend(['StepLR:gamma=0.5'])
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
plt.show()

# 'none' b loss
loss_object = torch.nn.CrossEntropyLoss(reduction='none')
# tf2【b,seq_len,vocab_size】
# pytorch pred【b,vocab_size,seq_len】
"""
- Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.

- Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional loss.
"""

# real [b, targ_seq_len]
# pred [b, targ_seq_len, target_vocab_size]
def mask_loss_func(real, pred):
    # print(real.shape, pred.shape)
    # _loss = loss_object(pred, real) # [b, targ_seq_len]
    _loss = loss_object(pred.transpose(-1, -2), real)  # [b, targ_seq_len]

    # logical_not  

    # mask = torch.logical_not(real.eq(0)).type(_loss.dtype) # [b, targ_seq_len] pad=0
    mask = torch.logical_not(real.eq(pad)).type(_loss.dtype)  # [b, targ_seq_len] pad!=0


    _loss *= mask

    return _loss.sum() / mask.sum().item()

def mask_loss_func2(real, pred):
    # _loss = loss_object(pred, real) # [b, targ_seq_len]
    _loss = loss_object(pred.transpose(-1, -2), real)  # [b, targ_seq_len]
    # mask = torch.logical_not(real.eq(0)) # [b, targ_seq_len]
    mask = torch.logical_not(real.eq(pad)) # [b, targ_seq_len] 
    _loss = _loss.masked_select(mask) # 
    return _loss.mean()


y_pred = torch.randn(3,3) # [3,3]
y_true = torch.tensor([1,2,0]) # [3]
# print(y_true.shape, y_pred.shape)
print(loss_object(y_pred, y_true))
print('loss with mask:', loss_object(y_pred, y_true).mean())
print('loss without mask:', mask_loss_func(y_true, y_pred))
print('loss without mask:', mask_loss_func2(y_true, y_pred))





def mask_accuracy_func(real, pred):
    _pred = pred.argmax(dim=-1)  # [b, targ_seq_len, target_vocab_size]=>[b, targ_seq_len]
    corrects = _pred.eq(real)  # [b, targ_seq_len] bool值

    # logical_not  

    # mask = torch.logical_not(real.eq(0)) # [b, targ_seq_len] 
    mask = torch.logical_not(real.eq(pad))  # [b, targ_seq_len] 


    corrects *= mask

    return corrects.sum().float() / mask.sum().item()


def mask_accuracy_func2(real, pred):
    _pred = pred.argmax(dim=-1) # [b, targ_seq_len, target_vocab_size]=>[b, targ_seq_len]
    corrects = _pred.eq(real).type(torch.float32) # [b, targ_seq_len]
    # mask = torch.logical_not(real.eq(0)) # [b, targ_seq_len] bool
    mask = torch.logical_not(real.eq(pad)) # [b, targ_seq_len] bool
    corrects = corrects.masked_select(mask) # 

    return corrects.mean()

def mask_accuracy_func3(real, pred):
    _pred = pred.argmax(dim=-1) # [b, targ_seq_len, target_vocab_size]=>[b, targ_seq_len]
    corrects = _pred.eq(real) # [b, targ_seq_len] bool值
    # mask = torch.logical_not(real.eq(0)) # [b, targ_seq_len] bool
    mask = torch.logical_not(real.eq(pad)) # [b, targ_seq_len] bool
    corrects = torch.logical_and(corrects, mask)
    # print(corrects.dtype) # bool
    # print(corrects.sum().dtype) #int64
    return corrects.sum().float()/mask.sum().item()

y_pred = torch.randn(3,3) # [3,3]
y_true = torch.tensor([0,2,1]) # [3] 
print(y_true)
print(y_pred)
print('acc without mask:', mask_accuracy_func(y_true, y_pred))
print('acc without mask:', mask_accuracy_func2(y_true, y_pred))
print('acc without mask:', mask_accuracy_func3(y_true, y_pred))


def create_mask(inp, targ):
    # encoder padding mask
    enc_padding_mask = create_padding_mask(inp)  # =>[b,1,1,inp_seq_len] mask=1 pad


    look_ahead_mask = create_look_ahead_mask(targ.shape[-1])  # =>[targ_seq_len,targ_seq_len] ##################
    dec_targ_padding_mask = create_padding_mask(targ)  # =>[b,1,1,targ_seq_len]
    combined_mask = torch.max(look_ahead_mask, dec_targ_padding_mask)  # 2 mask =>[b,1,targ_seq_len,targ_seq_len]


    dec_padding_mask = create_padding_mask(inp)  # =>[b,1,1,inp_seq_len] mask=1 pad

    return enc_padding_mask, combined_mask, dec_padding_mask
    # [b,1,1,inp_seq_len], [b,1,targ_seq_len,targ_seq_len], [b,1,1,inp_seq_len]


save_dir = './save/'

transformer = Transformer(num_layers,
                          d_model,
                          num_heads,
                          dff,
                          input_vocab_size,
                          target_vocab_size,
                          pe_input=input_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

print(transformer) 

transformer = transformer.to(device)
if ngpu > 1: 
    transformer = torch.nn.DataParallel(transformer,  device_ids=list(range(ngpu))) # 设置并行执行  device_ids=[0,1]

optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
lr_scheduler = CustomSchedule(optimizer, d_model, warm_steps=4000)



# inp [b,inp_seq_len]
# targ [b,targ_seq_len]

def train_step(model, inp, targ):
    targ_inp = targ[:, :-1]
    targ_real = targ[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_mask(inp, targ_inp)

    inp = inp.to(device)
    targ_inp = targ_inp.to(device)
    targ_real = targ_real.to(device)
    enc_padding_mask = enc_padding_mask.to(device)
    combined_mask = combined_mask.to(device)
    dec_padding_mask = dec_padding_mask.to(device)
    # print('device:', inp.device, targ_inp)

    model.train()

    optimizer.zero_grad()  

    # forward
    prediction, _ = transformer(inp, targ_inp, enc_padding_mask, combined_mask, dec_padding_mask)
    # [b, targ_seq_len, target_vocab_size]
    # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
    #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}

    loss = mask_loss_func(targ_real, prediction)
    metric = mask_accuracy_func(targ_real, prediction)

    # backward
    loss.backward()  
    optimizer.step() 

    return loss.item(), metric.item()



batch_src, batch_targ = next(iter(train_dataloader)) # [64,10], [64,10]
print(train_step(transformer, batch_src, batch_targ))
"""
x += pos_encoding  # [b, inp_seq_len, d_model]
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!
"""


def validate_step(model, inp, targ):
    targ_inp = targ[:, :-1]
    targ_real = targ[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_mask(inp, targ_inp)

    inp = inp.to(device)
    targ_inp = targ_inp.to(device)
    targ_real = targ_real.to(device)
    enc_padding_mask = enc_padding_mask.to(device)
    combined_mask = combined_mask.to(device)
    dec_padding_mask = dec_padding_mask.to(device)

    model.eval()  

    with torch.no_grad():
        # forward
        prediction, _ = model(inp, targ_inp, enc_padding_mask, combined_mask, dec_padding_mask)
        # [b, targ_seq_len, target_vocab_size]
        # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
        #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}

        val_loss = mask_loss_func(targ_real, prediction)
        val_metric = mask_accuracy_func(targ_real, prediction)

    return val_loss.item(), val_metric.item()


EPOCHS = 50 # 50 # 30  # 20

print_trainstep_every = 50  # 

metric_name = 'acc'

df_history = pd.DataFrame(columns=['epoch', 'loss', metric_name, 'val_loss', 'val_' + metric_name])


def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m_%d %H:%M:%S')
    print('\n' + "=========="*8 + '%s'%nowtime)


def train_model(model, epochs, train_dataloader, val_dataloader, print_every):
    starttime = time.time()
    print('*' * 27, 'start training...')
    printbar()

    best_acc = 0.
    for epoch in range(1, epochs + 1):

        # lr_scheduler.step() # 

        loss_sum = 0.
        metric_sum = 0.

        for step, (inp, targ) in enumerate(train_dataloader, start=1):
            # inp [64, 10] , targ [64, 10]
            loss, metric = train_step(model, inp, targ)

            loss_sum += loss
            metric_sum += metric


            if step % print_every == 0:
                print('*' * 8, f'[step = {step}] loss: {loss_sum / step:.3f}, {metric_name}: {metric_sum / step:.3f}')

            lr_scheduler.step()  # 

        # test(model, train_dataloader)
        val_loss_sum = 0.
        val_metric_sum = 0.
        for val_step, (inp, targ) in enumerate(val_dataloader, start=1):
            # inp [64, 10] , targ [64, 10]
            loss, metric = validate_step(model, inp, targ)

            val_loss_sum += loss
            val_metric_sum += metric


        # record = (epoch, loss_sum/step, metric_sum/step)
        record = (epoch, loss_sum/step, metric_sum/step, val_loss_sum/val_step, val_metric_sum/val_step)
        df_history.loc[epoch - 1] = record

        # print('*'*8, 'EPOCH = {} loss: {:.3f}, {}: {:.3f}'.format(
        #        record[0], record[1], metric_name, record[2]))
        print('EPOCH = {} loss: {:.3f}, {}: {:.3f}, val_loss: {:.3f}, val_{}: {:.3f}'.format(
            record[0], record[1], metric_name, record[2], record[3], metric_name, record[4]))
        printbar()

        # current_acc_avg = metric_sum / step
        current_acc_avg = val_metric_sum / val_step #
        if current_acc_avg > best_acc:  #
            best_acc = current_acc_avg
            # checkpoint = save_dir + '{:03d}_{:.2f}_ckpt.tar'.format(epoch, current_acc_avg)
            if device.type == 'cuda' and ngpu > 1:
                # model_sd = model.module.state_dict()  ##################
                model_sd = copy.deepcopy(model.module.state_dict())
            else:
                # model_sd = model.state_dict(),  ##################
                model_sd = copy.deepcopy(model.state_dict())  ##################
            torch.save({
                'loss': loss_sum / step,
                'epoch': epoch,
                'net': model_sd,
                'opt': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }, "project2_model.pt")

    print('finishing training...')
    endtime = time.time()
    time_elapsed = endtime - starttime
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return df_history

df_history = train_model(transformer, EPOCHS, train_dataloader, val_dataloader, print_trainstep_every)
print(df_history)



def plot_metric(df_history, metric):
    plt.figure()

    train_metrics = df_history[metric]
    val_metrics = df_history['val_' + metric]  #

    epochs = range(1, len(train_metrics) + 1)

    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')  #

    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.savefig('./imgs/' + metric + 'pair.png')  # 保存图片
    plt.show()


plot_metric(df_history, 'loss')
plot_metric(df_history, metric_name)


checkpoint = "project2_model.pt"
print('checkpoint:', checkpoint)
# ckpt = torch.load(checkpoint, map_location=device) 
ckpt = torch.load(checkpoint) 
# print('ckpt', ckpt)
transformer_sd = ckpt['net']
# optimizer_sd = ckpt['opt'] 
# lr_scheduler_sd = ckpt['lr_scheduler']

reload_model = Transformer(num_layers,
                           d_model,
                           num_heads,
                           dff,
                           input_vocab_size,
                           target_vocab_size,
                           pe_input=input_vocab_size,
                           pe_target=target_vocab_size,
                           rate=dropout_rate)

reload_model = reload_model.to(device)
if ngpu > 1:
    reload_model = torch.nn.DataParallel(reload_model,  device_ids=list(range(ngpu))) 


print('Loading model ...')
if device.type == 'cuda' and ngpu > 1:
   reload_model.module.load_state_dict(transformer_sd)
else:
   reload_model.load_state_dict(transformer_sd)
print('Model loaded ...')


def test(model, dataloader):
    # model.eval()

    test_loss_sum = 0.
    test_metric_sum = 0.
    for test_step, (inp, targ) in enumerate(dataloader, start=1):
        # inp [64, 10] , targ [64, 10]
        loss, metric = validate_step(model, inp, targ)
        # print('*'*8, loss, metric)

        test_loss_sum += loss
        test_metric_sum += metric

    print('*' * 8,
          'Test: loss: {:.3f}, {}: {:.3f}'.format(test_loss_sum / test_step, 'test_acc', test_metric_sum / test_step))


print('*' * 8, 'final test...')
test(reload_model, val_dataloader)



def tokenizer_encode(tokenize, sentence, vocab):
    # print(type(vocab)) # torchtext.vocab.Vocab
    # print(len(vocab))
    sentence = normalizeString(sentence)
    # print(type(sentence)) # str
    sentence = tokenize(sentence)  # list
    sentence = ['<start>'] + sentence + ['<end>']
    sentence_ids = [vocab.stoi[token] for token in sentence]
    # print(sentence_ids, type(sentence_ids[0])) # int
    return sentence_ids


def tokenzier_decode(sentence_ids, vocab):
    sentence = [vocab.itos[id] for id in sentence_ids if id<len(vocab)]
    # print(sentence)
    return " ".join(sentence)

s = 'je pars en vacances pour quelques jours .'
print(tokenizer_encode(tokenizer, s, SRC_TEXT.vocab))


s_ids = [3, 5, 251, 17, 365, 35, 492, 390, 4, 2]
print(tokenzier_decode(s_ids, SRC_TEXT.vocab))
print(tokenzier_decode(s_ids, TARG_TEXT.vocab))



def evaluate(model, inp_sentence):
    model.eval() 

    inp_sentence_ids = tokenizer_encode(tokenizer, inp_sentence, SRC_TEXT.vocab)  
    # print(tokenzier_decode(inp_sentence_ids, SRC_TEXT.vocab))
    encoder_input = torch.tensor(inp_sentence_ids).unsqueeze(dim=0)  # =>[b=1, inp_seq_len=10]
    # print(encoder_input.shape)

    decoder_input = [TARG_TEXT.vocab.stoi['<start>']]
    decoder_input = torch.tensor(decoder_input).unsqueeze(0)  # =>[b=1,seq_len=1]
    # print(decoder_input.shape)

    with torch.no_grad():
        for i in range(MAX_LENGTH + 2):
            enc_padding_mask, combined_mask, dec_padding_mask = create_mask(encoder_input.cpu(), decoder_input.cpu()) ################
            # [b,1,1,inp_seq_len], [b,1,targ_seq_len,inp_seq_len], [b,1,1,inp_seq_len]

            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            enc_padding_mask = enc_padding_mask.to(device)
            combined_mask = combined_mask.to(device)
            dec_padding_mask = dec_padding_mask.to(device)

            # forward
            predictions, attention_weights = model(encoder_input,
                                                   decoder_input,
                                                   enc_padding_mask,
                                                   combined_mask,
                                                   dec_padding_mask)
            # [b=1, targ_seq_len, target_vocab_size]
            # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
            #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}


            prediction = predictions[:, -1:, :]  # =>[b=1, 1, target_vocab_size]
            prediction_id = torch.argmax(prediction, dim=-1)  # => [b=1, 1]
            # print('prediction_id:', prediction_id, prediction_id.dtype) # torch.int64
            if prediction_id.squeeze().item() == TARG_TEXT.vocab.stoi['<end>']:
                return decoder_input.squeeze(dim=0), attention_weights

            decoder_input = torch.cat([decoder_input, prediction_id],
                                      dim=-1)  # [b=1,targ_seq_len=1]=>[b=1,targ_seq_len=2]


    return decoder_input.squeeze(dim=0), attention_weights
    # [targ_seq_len],
    # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
    #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}



# s = 'Споразумение между и Русия за визите за краткосрочно пребиваване .'
# evaluate(s)

# s = 'Одобряване на протокола от предишното заседание.'
# s_targ = '	Approval of Minutes of previous sitting .'
# pred_result, attention_weights = evaluate(reload_model, s)
# pred_sentence = tokenzier_decode(pred_result, TARG_TEXT.vocab)
# print('real target:', s_targ)
# print('pred_sentence:', pred_sentence)



sentence_pairs = [
    ['je pars en vacances pour quelques jours .', 'i m taking a couple of days off .'],
    ['je ne me panique pas .', 'i m not panicking .'],
    ['je recherche un assistant .', 'i am looking for an assistant .'],
    ['je suis loin de chez moi .', 'i m a long way from home .'],
    ['vous etes en retard .', 'you re very late .'],
    ['j ai soif .', 'i am thirsty .'],
    ['je suis fou de vous .', 'i m crazy about you .'],
    ['vous etes vilain .', 'you are naughty .'],
    ['il est vieux et laid .', 'he s old and ugly .'],
    ['je suis terrifiee .', 'i m terrified .'],
]


def batch_translate(sentence_pairs):
    for pair in sentence_pairs:
        print('input:', pair[0])
        print('target:', pair[1])
        pred_result, _ = evaluate(reload_model, pair[0])
        pred_sentence = tokenzier_decode(pred_result, TARG_TEXT.vocab)
        print('pred:', pred_sentence)
        print('')

batch_translate(sentence_pairs)



def evaluateRandomly(n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('input:', pair[0])
        print('target:', pair[1])
        pred_result, attentions = evaluate(reload_model, pair[0])
        pred_sentence = tokenzier_decode(pred_result, TARG_TEXT.vocab)
        print('pred:', pred_sentence)
        print('')


evaluateRandomly(2)





def plot_attention_weights(attention, sentence, pred_sentence, layer):
    sentence = sentence.split()
    pred_sentence = pred_sentence.split()

    fig = plt.figure(figsize=(16, 8))

    # block2 attention[layer] => [b=1, num_heads, targ_seq_len, inp_seq_len]
    attention = torch.squeeze(attention[layer], dim=0) # => [num_heads, targ_seq_len, inp_seq_len]

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)  #

        cax = ax.matshow(attention[head].cpu(), cmap='viridis')  #

        fontdict = {'fontsize': 10}


        ax.set_xticks(range(len(sentence)+2))  #
        ax.set_yticks(range(len(pred_sentence)))

        ax.set_ylim(len(pred_sentence) - 1.5, -0.5)  

        ax.set_xticklabels(['<start>']+sentence+['<end>'], fontdict=fontdict, rotation=90)  
        ax.set_yticklabels(pred_sentence, fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1))
    plt.tight_layout()
    plt.show()


def translate(sentence_pair, plot=None):
    print('input:', sentence_pair[0])
    print('target:', sentence_pair[1])
    pred_result, attention_weights = evaluate(reload_model, sentence_pair[0])
    print('attention_weights:', attention_weights.keys())
    pred_sentence = tokenzier_decode(pred_result, TARG_TEXT.vocab)
    print('pred:', pred_sentence)
    print('')

    if plot:
        plot_attention_weights(attention_weights, sentence_pair[0], pred_sentence, plot)

translate(sentence_pairs[0], plot='decoder_layer4_block2')