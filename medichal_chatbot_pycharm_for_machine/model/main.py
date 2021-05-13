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
import time

from evaluation import BeamSearchDecoder, GreedySearchDecoder

# Load/Assemble voc and pairs
save_dir = os.path.join("data", "save")
voc, pairs = prepare_data.loadPrepareData()
# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)



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




# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.learning_rate * config.decoder_learning_ratio)
if config.loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# If you have cuda, configure cuda to call
for state in encoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

for state in decoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

random.shuffle(pairs)
split_val = int(len(pairs) * 0.8)
train_pairs = pairs[:split_val]
val_pairs = pairs[split_val:]

print("Training dataset size: {}, Validation dataset size: {}".format(len(train_pairs), len(val_pairs)))
    
# Run training iterations
print("Starting Training!")
total_time = 0
epoch_val_loss = evaluation.validate_batches(voc, val_pairs, encoder, decoder)
print("Initial validation loss: {:.4f}".format(epoch_val_loss))
for epoch in range(1, config.n_epoch + 1):
    s = time.time()
    print("Starting epoch number " + str(epoch))
    epoch_train_loss = train.train_iters(voc, train_pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
                                         embedding, save_dir, checkpoint, epoch)

    # epoch_val_loss2 = evaluation.validate_batches(voc, train_pairs, encoder, decoder)

    epoch_val_loss = evaluation.validate_batches(voc, val_pairs, encoder, decoder)
    epoch_time = time.time() - s

    print("Epoch: {}, Train loss: {:.4f}, Validation loss: {:.4f}, Duration: {}".format(epoch, epoch_train_loss,
                                                                                        epoch_val_loss, epoch_time))
    total_time += epoch_time
    print(f"Total time elapsed: {total_time}")

######################################################################
# Run Evaluation
# ~~~~~~~~~~~~~~
#
# To chat with your model, run the following block.
#

# Set dropout layers to eval mode
# encoder.eval()
# decoder.eval()
#
# # Initialize search module
# searcher = GreedySearchDecoder(encoder, decoder)
# searcher2 = BeamSearchDecoder(encoder, decoder)
#
# evaluation.evaluateInput(encoder, decoder, searcher, voc, searcher2=searcher2)