# -*- coding: utf-8 -*-

"""
Chatbot Tutorial
================
**Author:** `Matthew Inkawhich <https://github.com/MatthewInkawhich>`_
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch import optim
import random
import os
from model import EncoderRNN, LuongAttnDecoderRNN
import config
import train
import prepare_data
import evaluation
from evaluation import BeamSearchDecoder, GreedySearchDecoder, SamplingDecoder
# import flask
# from flask import request

# Load/Assemble voc and pairs
save_dir = os.path.join("data", "save")
voc, pairs = prepare_data.loadPrepareData()
# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)


# Trim voc and pairs
pairs = prepare_data.trimRareWords(voc, pairs, config.MIN_WORD_COUNT)


# Example for validation
small_batch_size = 5
batches = prepare_data.batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)

#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
checkpoint = None
if config.loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(config.loadFilename, map_location=torch.device('cpu'))
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, config.hidden_size)
if config.loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(config.hidden_size, embedding, config.encoder_n_layers, config.dropout)
decoder = LuongAttnDecoderRNN(config.attn_model, embedding, config.hidden_size, voc.num_words, config.decoder_n_layers, config.dropout)
if config.loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(config.device)
decoder = decoder.to(config.device)
print('Models built and ready to go!')


######################################################################
# Run Training
# ~~~~~~~~~~~~
#

# Configure training/optimization


# Ensure dropout layers are in train mode
# encoder.train()
# decoder.train()
#
# # Initialize optimizers
# print('Building optimizers ...')
# encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.learning_rate)
# decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.learning_rate * config.decoder_learning_ratio)
# if config.loadFilename:
#     encoder_optimizer.load_state_dict(encoder_optimizer_sd)
#     decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# If you have cuda, configure cuda to call
# for state in encoder_optimizer.state.values():
#     for k, v in state.items():
#         if isinstance(v, torch.Tensor):
#             state[k] = v.cuda()

# for state in decoder_optimizer.state.values():
#     for k, v in state.items():
#         if isinstance(v, torch.Tensor):
#             state[k] = v.cuda()
    
# Run training iterations
# print("Starting Training!")
# for epoch in range(1, config.n_epoch + 1):
#     print("Starting epoch number " + str(epoch))
#     train.trainIters(config.model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
#                embedding, config.encoder_n_layers, config.decoder_n_layers, save_dir, config.n_iteration, config.batch_size,
#                config.print_every, config.save_every, config.clip, config.corpus_name, config.loadFilename, checkpoint, epoch)
#

######################################################################
# Run Evaluation
# ~~~~~~~~~~~~~~
#
# To chat with your model, run the following block.
#

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)
searcher2 = BeamSearchDecoder(encoder, decoder)
searcher3 = SamplingDecoder(encoder, decoder)

evaluation.evaluateInput(encoder, decoder, searcher, voc, searcher2=searcher2, searcher3=searcher3)

