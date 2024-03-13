#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from comet_ml import Experiment
#experiment = Experiment()


# In[ ]:


import os
import math
import random
import re
import string
import argparse

from tqdm import tqdm
from PIL import Image, ImageFile
from pickle import load, dump

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

#import collections
import pandas as pd
from janome.tokenizer import Tokenizer

from torch import einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# In[ ]:


class TextData:
    texts1 = []
    texts2 = []
    
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        
        eng = df.iloc[:, 1]
        jp = df.iloc[:, 2]

        print('Load text1 from datasets.')
        tokenizer = Tokenizer()
        self.texts1 = [list(tokenizer.tokenize(t.strip(), wakati=True)) for t in tqdm(jp)]
        
        print('Load text2 from datasets.')
        self.texts2 = [s.split() for s in
                       [' '.join(re.split('([' + string.punctuation + '])', t.strip())) for t in tqdm(eng)]]
    
    def __getitem__(self, index):
        return list(self.texts1[index]), list(self.texts2[index])
        
    def __len__(self):
        return len(self.texts1)
    
    def tolist(self):
        return sum(self.texts1, []) + sum(self.texts2, [])


# In[ ]:


class TextDataset(Dataset):
    @staticmethod
    def make_vocab(text_data: TextData, vocab_size=None):
        print('Generate word-ids.')
        word2id = {}
        word2id['<pad>'] = 0
        word2id['<unk>'] = 1
        word2id['<s>'] = 2
        
        #wc = collections.Counter(text_data.tolist())
        #for i, (w, _) in enumerate(wc.most_common(vocab_size), 3):
        #    word2id[w] = i
        
        id2word = {v: k for k, v in word2id.items()}
        
        for words1, words2 in tqdm(text_data):
            for word in (words1 + words2):
                if word not in word2id:
                    id = len(word2id)
                    word2id[word] = id
                    id2word[id] = word
        
        return word2id, id2word
    
    def to_string(self, data):
        text = ''
        for d in data:
            try:
                text += self.id2word[int(d)] + ' '
            except:
                text += '<unk>'
        return text
    
    def to_tokens(self, data):
        tokens = []
        for d in data:
            try:
                tokens += [self.word2id[d]]
            except:
                tokens += [self.word2id['<unk>']]
        return tokens
    
    def __init__(self, csv_path, sentence_size, vocab_size=None):
        self.sentence_size = sentence_size
        
        if os.path.exists('textdata.dat'):
            with open(os.path.join('.', 'textdata.dat'), 'rb') as f:
                self.text_data = load(f)
                print('Loaded textdata.dat.')
        else:
            self.text_data = TextData(csv_path)
            with open(os.path.join('.', 'textdata.dat'), 'wb') as f:
                dump(self.text_data, f)
                print('Saved textdata.dat.')
        
        if os.path.exists('word2id.dat') and os.path.exists('id2word.dat'):
            with open(os.path.join('.', 'word2id.dat'), 'rb') as f:
                self.word2id = load(f)
                print('Loaded word2id.dat.')
            with open(os.path.join('.', 'id2word.dat'), 'rb') as f:
                self.id2word = load(f)
                print('Loaded id2word.dat.')
        else:
            self.word2id, self.id2word = TextDataset.make_vocab(self.text_data, vocab_size)
            with open(os.path.join('.', 'word2id.dat'), 'wb') as f:
                dump(self.word2id, f)
                print('Saved word2id.dat.')
            with open(os.path.join('.', 'id2word.dat'), 'wb') as f:
                dump(self.id2word, f)
                print('Saved id2word.dat.')
    
    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, index):
        text1, text2 = self.text_data[index]
        tokens1 = self.to_tokens(text1)
        tokens2 = self.to_tokens(text2)
        
        tokens1 = [self.word2id['<s>']] + tokens1
        tokens2 = [self.word2id['<s>']] + tokens2
        
        tokens1 = tokens1[:self.sentence_size]
        tokens2 = tokens2[:self.sentence_size]
        
        tokens1.extend([self.word2id['<pad>'] for _ in range(self.sentence_size - len(tokens1))])
        tokens2.extend([self.word2id['<pad>'] for _ in range(self.sentence_size - len(tokens2))])
        
        tokens1 = torch.LongTensor(tokens1)
        tokens2 = torch.LongTensor(tokens2)
        
        return tokens1, tokens2


# In[ ]:


class Mish(nn.Module):
    @staticmethod
    def mish(x):
        return x * torch.tanh(F.softplus(x))
    
    def forward(self, x):
        return Mish.mish(x)


# In[ ]:


class PositionalEncoder(nn.Module):
    def __init__(self, vocab_size, sentence_size):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.pe = torch.Tensor(sentence_size, vocab_size)
        for pos in range(sentence_size):
            for i in range(0, vocab_size, 2):
                self.pe[pos, i] = math.sin(pos / (10000**((2*i)/vocab_size)))
                self.pe[pos, i+1] = math.cos(pos / (10000**((2*(i+1))/vocab_size)))
        self.pe = self.pe.unsqueeze(0)
        self.pe.requires_grad = False
    
    def to(self, device):
        self.pe = self.pe.to(device)
        return super().to(device)
    
    def forward(self, x):
        return math.sqrt(self.vocab_size) * x + self.pe


# In[ ]:


class Embedding(nn.Module):
    def __init__(self, vocab_size, sentence_size, dim, dropout=0.):
        super().__init__()
        
        self.word_embedding = nn.Embedding(vocab_size, dim)
        #self.position_embedding = nn.Embedding(sentence_size, dim)
        self.position_embedding = PositionalEncoder(dim, sentence_size)
        self.LayerNorm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids):
        word_embedding = self.word_embedding(input_ids)
        
        #position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        #position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        #position_embedding = self.position_embedding(position_ids)
        
        #embedding = word_embedding + position_embedding
        embedding = self.position_embedding(word_embedding)
        embedding = self.LayerNorm(embedding)
        embedding = self.dropout(embedding)
        
        return embedding


# In[ ]:


class Attention(nn.Module):
    def __init__(self, dim, cross=False, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        inner_dim = dim_head * heads
        self.to_query = nn.Linear(dim, inner_dim, bias = False)
        if not cross:
            self.to_key = nn.Linear(dim, inner_dim, bias = False)
            self.to_value = nn.Linear(dim, inner_dim, bias = False)
        
        self.dropout = nn.Dropout(dropout)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if not (heads == 1 and dim_head == dim) else nn.Identity()
        
    def forward(self, x, kv=None, mask=None, return_attention=False):
        query = self.to_query(x)
        if kv != None:
            key = value = kv
        else:
            key = self.to_key(x)
            value = self.to_value(x)
        
        query = rearrange(query, 'b n (h d) -> b h n d', h = self.heads)
        key = rearrange(key, 'b n (h d) -> b h n d', h = self.heads)
        value = rearrange(value, 'b n (h d) -> b h n d', h = self.heads)
        
        attention_score = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, float('-inf'))
        
        attention_prob = F.softmax(attention_score, dim=-1)
        attention_prob = self.dropout(attention_prob)
        
        context = torch.matmul(attention_prob, value)
        context = rearrange(context, 'b h n d -> b n (h d)')
        
        if return_attention:
            return self.to_out(context), attention_prob
        else:
            return self.to_out(context)


# In[ ]:


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


# In[ ]:


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# In[ ]:


class Encoder(nn.Module):
    def __init__(self, dim, mlp_dim, depth=4, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
            
    def forward(self, x):
        for attention, feedforward in self.layers:
            x = attention(x) + x
            x = feedforward(x) + x
        return x


# In[ ]:


class Decoder(nn.Module):
    def __init__(self, dim, mlp_dim, vocab_size, depth=4, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, Attention(dim, cross=True, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x, kv):
        tri_mask = torch.triu(torch.ones(x.size(-2), x.size(-2))).transpose(1, 0).to(x.device)
        for attention, cross_attention, feedforward in self.layers:
            x = attention(x, mask=tri_mask) + x
            x = cross_attention(x, kv=kv) + x
            x = feedforward(x) + x
        return x


# In[ ]:


class Transformer(nn.Module):
    def __init__(self, vocab_size, sentence_size, dim=512, mlp_dim=1024, dropout=0., emb_dropout=0.):
        super().__init__()
        
        self.embedding = Embedding(vocab_size, sentence_size, dim, emb_dropout)
        self.encoder = Encoder(dim, mlp_dim, dropout=dropout)
        self.decoder = Decoder(dim, mlp_dim, vocab_size, dropout=dropout)
        self.linear = nn.Linear(dim, vocab_size) # トークン出力分布
    
    def to(self, *args, **kwargs):
        self.embedding.position_embedding.to(args[0])
        return super().to(*args, **kwargs)
    
    def forward(self, x, y):
        x = self.embedding(x)
        y = self.embedding(y)
        enc = self.encoder(x)
        dec = self.decoder(y, enc)
        out = self.linear(dec)
        return out


# In[ ]:


class Solver:
    def __init__(self, args):
        use_cuda = torch.cuda.is_available() if not args.cpu else False
        self.device = torch.device("cuda" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True
        print(f'Use Device: {self.device}')
        
        self.args = args
        
        self.dataset = TextDataset(self.args.csv_path, self.args.sentence_size)
        self.dataloader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        self.vocab_size = len(self.dataset.word2id)
        
        self.transformer = Transformer(self.vocab_size, self.args.sentence_size,
                                       dropout=self.args.dropout_prob).to(self.device)
        self.transformer.apply(self.weights_init)
        self.optimizer = optim.Adam(self.transformer.parameters(), lr=self.args.lr)
        
        self.epoch = 0
    
    def weights_init(self, module):
        if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.ConvTranspose2d:
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0)
    
    def save_state(self, epoch):
        self.transformer.cpu()
        torch.save(self.transformer.state_dict(), os.path.join(self.args.weight_dir, f'weight.{epoch}.pth'))
        self.transformer.to(self.device)
        
    def load_state(self):
        if os.path.exists('weight.pth'):
            self.transformer.load_state_dict(torch.load('weight.pth', map_location=self.device))
            print('Loaded network state.')
    
    def save_resume(self):
        with open(os.path.join('.', f'resume.pkl'), 'wb') as f:
            dump(self, f)
    
    def load_resume(self):
        if os.path.exists('resume.pkl'):
            with open(os.path.join('.', 'resume.pkl'), 'rb') as f:
                print('Load resume.')
                return load(f)
        else:
            return self
    
    def trainTransformer(self, epoch, iters, max_iters, text1, text2):
        softmax_crossentropy = nn.CrossEntropyLoss()
        
        # Compute loss.
        text_evals = self.transformer(text1, text2)
        text_evals_reshape = text_evals.reshape(-1, text_evals.size(-1))
        
        pad = torch.LongTensor([self.dataset.word2id['<pad>']]).unsqueeze(0).repeat(text2.size(0), 1).to(self.device)
        text_study = torch.cat((text2[:, 1:], pad), dim=1)
        text_study_reshape = text_study.reshape(-1)
        
        loss = softmax_crossentropy(text_evals_reshape, text_study_reshape)
        
        # Backward and optimize.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Logging.
        losses = {}
        losses['Loss'] = loss.item()
        
        # Save.
        if iters == max_iters:
            text_prob = torch.softmax(text_evals[0], dim=-1)
            text = text_prob.argmax(dim=-1)
            print(self.dataset.to_string(text1[0]))
            print('================>>>>')
            print(self.dataset.to_string(text))
            self.save_state(epoch)
        
        return losses
    
    def train(self):
        self.transformer.train()
        
        hyper_params = {}
        hyper_params['CSV Path'] = args.csv_path
        hyper_params['Weight Dir'] = args.weight_dir
        hyper_params['Sentence Size'] = args.sentence_size
        hyper_params['Dropout Prob.'] = args.dropout_prob
        hyper_params['Learning Rate'] = args.lr
        hyper_params['Batch Size'] = args.batch_size
        hyper_params['Num Train'] = args.num_train

        for key in hyper_params.keys():
            print(f'{key}: {hyper_params[key]}')
        #experiment.log_parameters(hyper_params)
        
        while self.args.num_train > self.epoch:
            self.epoch += 1
            epoch_loss = 0.0
            
            max_iters = len(self.dataloader)
            for iters, (text1, text2) in enumerate(tqdm(self.dataloader)):
                iters += 1
                
                text1 = text1.to(self.device)
                text2 = text2.to(self.device)
                
                losses = self.trainTransformer(self.epoch, iters, max_iters, text1, text2)
                
                epoch_loss += losses['Loss']
                #experiment.log_metrics(losses)
            
            print(f'Epoch[{self.epoch}] Loss({epoch_loss})')
                    
            if not self.args.noresume:
                self.save_resume()
    
    def translate(self, text):
        tokenizer = Tokenizer()
        text = list(tokenizer.tokenize(text.strip(), wakati=True))
        
        tokens1 = []
        for t in text:
            try:
                tokens1 += [self.dataset.word2id[t]]
            except:
                tokens1 += [self.dataset.word2id['<unk>']]

        tokens1 = [self.dataset.word2id['<s>']] + tokens1
        tokens2 = [self.dataset.word2id['<s>']]
        tokens1 = tokens1[:self.args.sentence_size]
        tokens2 = tokens2[:self.args.sentence_size]
        tokens1.extend([self.dataset.word2id['<pad>'] for _ in range(self.args.sentence_size - len(tokens1))])
        tokens2.extend([self.dataset.word2id['<pad>'] for _ in range(self.args.sentence_size - len(tokens2))])
        tokens1 = torch.LongTensor(tokens1).unsqueeze(0).to(self.device)
        tokens2 = torch.LongTensor(tokens2).unsqueeze(0).to(self.device)

        text_evals = self.transformer(tokens1, tokens2)
        text_prob = torch.softmax(text_evals[0], dim=-1)
        text = text_prob.argmax(dim=-1)
            
        print(self.dataset.to_string(text))


# In[ ]:


def main(args):
    solver = Solver(args)
    solver.load_state()
    
    if not args.noresume:
        solver = solver.load_resume()
        solver.args = args

    if args.translate != '':
        solver.translate(args.translate)
        return
    
    solver.train()
    
    #experiment.end()


# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='')
    parser.add_argument('--weight_dir', type=str, default='weights')
    parser.add_argument('--sentence_size', type=int, default=128)
    parser.add_argument('--dropout_prob', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_train', type=int, default=10)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--noresume', action='store_true')
    parser.add_argument('--translate', type=str, default='')
    
    args, unknown = parser.parse_known_args()
    
    if not os.path.exists(args.weight_dir):
        os.mkdir(args.weight_dir)
    
    main(args)

