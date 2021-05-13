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

# Load/Assemble voc and pairs
save_dir = os.path.join("data", "save")
voc, pairs = prepare_data.loadPrepareData()
# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)


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

# evaluation.evaluateInput(encoder, decoder, searcher, voc, searcher2=searcher2, searcher3=searcher3)

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

    answer = evaluation.generateAnswer(question, encoder, decoder, decoder_gen, voc)

    return answer

