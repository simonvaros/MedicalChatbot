# -*- coding: utf-8 -*-

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
import flask
from flask import request
from flask_cors import CORS, cross_origin
import time

save_dir = os.path.join("data", "save")
voc, pairs = prepare_data.loadPrepareData()

print("\npairs:")
for pair in pairs[:10]:
    print(pair)


checkpoint = None
if config.loadFilename:
    checkpoint = torch.load(config.loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
embedding = nn.Embedding(voc.num_words, config.hidden_size)

if config.loadFilename:
    embedding.load_state_dict(embedding_sd)

encoder = EncoderRNN(config.hidden_size, embedding, config.encoder_n_layers, config.dropout)
decoder = LuongAttnDecoderRNN(config.attn_model, embedding, config.hidden_size, voc.num_words, config.decoder_n_layers, config.dropout)
if config.loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

encoder = encoder.to(config.device)
decoder = decoder.to(config.device)
print('Models built and ready to go!')


if config.training:

    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.learning_rate * config.decoder_learning_ratio)
    if config.loadFilename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    if config.USE_CUDA:
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

    print("Starting Training!")
    total_time = 0
    epoch_val_loss = evaluation.validate_batches(voc, val_pairs, encoder, decoder)
    print("Initial validation loss: {:.4f}".format(epoch_val_loss))

    for epoch in range(1, config.n_epoch + 1):
        s = time.time()
        print("Starting epoch number " + str(epoch))
        epoch_train_loss = train.train_iters(voc, train_pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
                                             embedding, save_dir, checkpoint, epoch)


        epoch_val_loss = evaluation.validate_batches(voc, val_pairs, encoder, decoder)
        epoch_time = time.time() - s

        print("Epoch: {}, Train loss: {:.4f}, Validation loss: {:.4f}, Duration: {}".format(epoch, epoch_train_loss,
                                                                                            epoch_val_loss, epoch_time))
        total_time += epoch_time
        print(f"Total time elapsed: {total_time}")


if config.evaluation:
    encoder.eval()
    decoder.eval()

    searcher = GreedySearchDecoder(encoder, decoder)
    searcher2 = BeamSearchDecoder(encoder, decoder)
    searcher3 = SamplingDecoder(encoder, decoder)

    evaluation.evaluateInput(searcher, voc, searcher2=searcher2, searcher3=searcher3)


app = flask.Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['GET'])
@cross_origin()
def get_answer():
    if 'question' in request.args:
        question = request.args['question']
    else:
        return "No question provided"

    decoder_gen = searcher

    if 'decoder' in request.args:
        decoder_type = request.args['decoder']

        if decoder_type == 'beamsearch':
            decoder_gen = searcher2
        if decoder_type == 'sampling':
            decoder_gen = searcher3

    answer = evaluation.generateAnswer(question, decoder_gen, voc)

    return answer
