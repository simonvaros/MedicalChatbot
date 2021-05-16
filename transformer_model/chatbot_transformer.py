"""
Language Translation with Transformer
=====================================
This tutorial shows, how to train a translation model from scratch using
Transformer. We will be using Multi30k dataset to train a German to English translation model.
"""

######################################################################
# Data Processing
# ---------------
# 
# torchtext has utilities for creating datasets that can be easily
# iterated through for the purposes of creating a language translation
# model. In this example, we show how to tokenize a raw text sentence,
# build vocabulary, and numericalize tokens into tensor.
# 
# To run this tutorial, first install spacy using pip or conda. Next,
# download the raw data for the English and German Spacy tokenizers from
# https://spacy.io/usage/models


import math
import torchtext
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive
from torch import Tensor
import io
import time
import pandas as pd
import random
import os

torch.manual_seed(0)
# torch.use_deterministic_algorithms(True)

# data_path = "./healthtap_1000_qa.csv"
data_path = "./healthtap_full_qa_processed_20k_words.csv"

medical_data = pd.read_csv(data_path)
medical_data_questions = medical_data[['question']]
medical_data_answers = medical_data[['answer']]
input_data = medical_data_questions.question.values.tolist()
target_data = medical_data_answers.answer.values.tolist()

en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

print('Tokenizers set')
print('Start building vocab')

train = False

def build_vocab(tokenizer):
    counter = Counter()
    for string_ in input_data:
        counter.update(tokenizer(string_))
    for string_ in target_data:
        counter.update(tokenizer(string_))
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


if os.path.exists('vocab.pth'):
    vocab = torch.load('vocab.pth')
    print('vocab loaded')
else:
    vocab = build_vocab(en_tokenizer)
    print('Vocab built')
    torch.save(vocab, 'vocab.pth')

print('Start processing data')


def data_process():
    data = []
    for (raw_questions, raw_answers) in zip(input_data, target_data):
        question_tensor_ = torch.tensor([vocab[token] for token in en_tokenizer(raw_questions.rstrip("\n"))],
                                  dtype=torch.long)
        answer_tensor_ = torch.tensor([vocab[token] for token in en_tokenizer(raw_answers.rstrip("\n"))],
                                  dtype=torch.long)
        data.append((question_tensor_, answer_tensor_))
    return data

if train:
    if os.path.exists('all_data.pth'):
        all_data = torch.load('all_data.pth')
        print('data loaded')
    else:
        all_data = data_process()
        print('data processed')
        torch.save(all_data, 'all_data.pth')

    random.shuffle(all_data)
    split_val = int(len(all_data) * 0.8)
    train_data = all_data[:split_val]
    val_data = all_data[split_val:]

print('Data prepared')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is: " + str(device))

BATCH_SIZE = 64
PAD_IDX = vocab['<pad>']
BOS_IDX = vocab['<bos>']
EOS_IDX = vocab['<eos>']

######################################################################
# DataLoader
# ----------
# 
# The last torch specific feature well use is the DataLoader, which is
# easy to use since it takes the data as its first argument. Specifically,
# as the docs say: DataLoader combines a dataset and a sampler, and
# provides an iterable over the given dataset. The DataLoader supports
# both map-style and iterable-style datasets with single- or multi-process
# loading, customizing loading order and optional automatic batching
# (collation) and memory pinning.
# 
# Please pay attention to collate_fn (optional) that merges a list of
# samples to form a mini-batch of Tensor(s). Used when using batched
# loading from a map-style dataset.
# 

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

if train:
    print('Start preparing batches')

    def generate_batch(data_batch):
        de_batch, en_batch = [], []
        for (de_item, en_item) in data_batch:
            de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
            en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
        de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
        en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
        return de_batch, en_batch


    train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=generate_batch)
    valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=generate_batch)

    print('Batches generated')

######################################################################
# Transformer!
# ------------
# 
# Transformer is a Seq2Seq model introduced in `“Attention is all you
# need” <https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>`__
# paper for solving machine translation task. Transformer model consists
# of an encoder and decoder block each containing fixed number of layers.
# 
# Encoder processes the input sequence by propogating it, through a series
# of Multi-head Attention and Feed forward network layers. The output from
# the Encoder referred to as ``memory``, is fed to the decoder along with
# target tensors. Encoder and decoder are trained in an end-to-end fashion
# using teacher forcing technique.
# 

from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, vocab_size: int,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.generator = nn.Linear(emb_size, vocab_size)
        self.tok_emb = TokenEmbedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor,
                tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        src_embedding = self.positional_encoding(self.tok_emb(src))
        tgt_embedding = self.positional_encoding(self.tok_emb(trg))
        memory = self.transformer_encoder(src_embedding, src_mask, src_padding_mask)
        outs = self.transformer_decoder(tgt_embedding, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(
            self.tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer_decoder(self.positional_encoding(
            self.tok_emb(tgt)), memory,
            tgt_mask)


######################################################################
# Text tokens are represented by using token embeddings. Positional
# encoding is added to the token embedding to introduce a notion of word
# order.
# 

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


######################################################################
# We create a ``subsequent word`` mask to stop a target word from
# attending to its subsequent words. We also create masks, for masking
# source and target padding tokens
# 

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


######################################################################
# Define model parameters and instantiate model 
#

VOCAB_SIZE = len(vocab)
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 64
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
NUM_EPOCHS = 16

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                                 EMB_SIZE, VOCAB_SIZE,
                                 FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(device)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(
    transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
)


######################################################################
#

def train_epoch(model, train_iter, optimizer):
    model.train()
    losses = 0
    iters_len = len(train_iter)
    start_time = time.time()
    iteration_time = 0
    for idx, (src, tgt) in enumerate(train_iter):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        iteration_time = (time.time() - start_time) / (idx + 1)
        remaining_time = (iters_len - idx + 1) * iteration_time
        print('training iter ' + str(idx + 1) + ' / ' + str(iters_len) + " remaining time: " + str(remaining_time))

    return losses / len(train_iter)


def evaluate(model, val_iter):
    model.eval()
    losses = 0
    for idx, (src, tgt) in (enumerate(valid_iter)):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(val_iter)


######################################################################
# Train model 
#

directory = os.path.join("save", "checkpoint")

if train:
    print('Start training')

    if not os.path.exists(directory):
        os.makedirs(directory)

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        train_loss = train_epoch(transformer, train_iter, optimizer)
        end_time = time.time()
        torch.save({
            'epoch': epoch,
            'en': transformer.transformer_encoder.state_dict(),
            'de': transformer.transformer_decoder.state_dict(),
            'opt': optimizer.state_dict(),
            'voc_dict': vocab.__dict__,
            'embedding': transformer.tok_emb.state_dict()
        }, os.path.join(directory, '{}_{}.tar'.format(epoch, 'checkpoint')))

        val_loss = evaluate(transformer, valid_iter)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
               f"Epoch time = {(end_time - start_time):.3f}s"))

checkpoint = torch.load(os.path.join(directory, '16_checkpoint.tar'), map_location=torch.device('cpu'))
transformer.transformer_encoder.load_state_dict(checkpoint['en'])
transformer.transformer_decoder.load_state_dict(checkpoint['de'])
transformer.tok_emb.load_state_dict(checkpoint['embedding'])
optimizer.load_state_dict(checkpoint['opt'])
vocab.__dict__ = checkpoint['voc_dict']

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len - 1):
        memory = memory.to(device)
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


def translate(model, src, src_vocab, tgt_vocab, src_tokenizer):
    model.eval()
    tokens = [BOS_IDX] + [src_vocab.stoi[tok] for tok in src_tokenizer(src)] + [EOS_IDX]
    num_tokens = len(tokens)
    src = (torch.LongTensor(tokens).reshape(num_tokens, 1))
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join([tgt_vocab.itos[tok] for tok in tgt_tokens]).replace("<bos>", "").replace("<eos>", "")

def evaluateInput():
    input_sentence = ''
    while (1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            print(translate(transformer, input_sentence, vocab, vocab, en_tokenizer))

        except KeyError:
            print("Error: Encountered unknown word.")


######################################################################
#


evaluateInput()