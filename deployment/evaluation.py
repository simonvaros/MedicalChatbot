import config
import torch
import torch.nn as nn
import operator
import prepare_data
import torch.nn.functional as F


from queue import PriorityQueue


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

class BeamSearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(BeamSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:config.decoder_n_layers]
        decoder_input = torch.ones(1, 1, device=config.device, dtype=torch.long) * config.SOS_token

        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()
        nodes.put((-node.eval(), node))
        qsize = 1

        beam_width = 10
        topk = 1
        endnodes = []
        number_required = 1

        while True:
            if qsize > 2000: break

            score, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h

            if n.wordid.item() == config.EOS_token and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)

            utterance = utterance[::-1]
            utterances.append(utterance)

        return utterances[0]

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:config.decoder_n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=config.device, dtype=torch.long) * config.SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=config.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=config.device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)

            if (decoder_input == config.EOS_token):
                break
            
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


def top_k_filtering(logits, top_k=0, filter_value=-float('Inf')):
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    return logits


class SamplingDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(SamplingDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:config.decoder_n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=config.device, dtype=torch.long) * config.SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=config.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=config.device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

            temperature = 0.7
            top_k = 5

            # Keep only the last token predictions of the first batch item (batch size 1), apply a temperature coefficient and filter
            decoder_output = decoder_output[-1] / temperature
            filtered_output = top_k_filtering(decoder_output, top_k=top_k)

            # Sample from the filtered distribution
            probabilities = F.softmax(filtered_output, dim=-1)
            decoder_input = torch.multinomial(probabilities, num_samples=1)

            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)

            if (decoder_input == config.EOS_token):
                break

            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens


######################################################################
# Evaluate my text
# ~~~~~~~~~~~~~~~~

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=config.MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [prepare_data.indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(config.device)
    lengths = lengths.to(config.device)
    # Decode sentence with searcher

    if (type(searcher) == BeamSearchDecoder):
        tokens = searcher(input_batch, lengths, max_length)
        decoded_words = [voc.index2word[token.item()] for token in tokens]
        return decoded_words

    if (type(searcher) == SamplingDecoder):
        tokens = searcher(input_batch, lengths, 100)
        decoded_words = [voc.index2word[token.item()] for token in tokens]
        return decoded_words

    tokens, scores = searcher(input_batch, lengths, max_length)
    # tokens = searcher(input_batch, lengths, max_length)
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc, searcher2 = None, searcher3 = None):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = prepare_data.normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)

            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot (greedy search):', ' '.join(output_words))

            if (searcher2 != None): 
                output_words2 = evaluate(encoder, decoder, searcher2, voc, input_sentence)
                output_words2[:] = [x for x in output_words2 if not (x == 'EOS' or x == 'PAD')]
                print('Bot (beam search):', ' '.join(output_words2))

            if (searcher3 != None):
                output_words3 = evaluate(encoder, decoder, searcher3, voc, input_sentence)
                output_words3[:] = [x for x in output_words3 if not (x == 'EOS' or x == 'PAD')]
                print('Bot (sampling):', ' '.join(output_words3))

        except KeyError:
            print("Error: Encountered unknown word.")

def generateAnswer(question, encoder, decoder, searcher, voc):
    try:
        # Normalize sentence
        input_sentence = prepare_data.normalizeString(question)
        # Evaluate sentence
        output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)

        # Format and print response sentence
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        return ' '.join(output_words)

    except KeyError:
        return "Error: Encountered unknown word."

