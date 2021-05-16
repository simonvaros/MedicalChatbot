import config
import torch
import torch.nn as nn
import operator
import prepare_data
import torch.nn.functional as F
from queue import PriorityQueue
import random
import train


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
        self.encoder.eval()
        self.decoder.eval()

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
        self.encoder.eval()
        self.decoder.eval()

        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:config.decoder_n_layers]
        decoder_input = torch.ones(1, 1, device=config.device, dtype=torch.long) * config.SOS_token
        all_tokens = torch.zeros([0], device=config.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=config.device)

        with torch.no_grad():
            for _ in range(max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
                all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
                all_scores = torch.cat((all_scores, decoder_scores), dim=0)

                if (decoder_input == config.EOS_token):
                    break

                decoder_input = torch.unsqueeze(decoder_input, 0)

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
        self.encoder.eval()
        self.decoder.eval()

        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:config.decoder_n_layers]
        decoder_input = torch.ones(1, 1, device=config.device, dtype=torch.long) * config.SOS_token
        all_tokens = torch.zeros([0], device=config.device, dtype=torch.long)

        with torch.no_grad():
            for _ in range(max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

                temperature = 0.7
                top_k = 5

                decoder_output = decoder_output[-1] / temperature
                filtered_output = top_k_filtering(decoder_output, top_k=top_k)

                probabilities = F.softmax(filtered_output, dim=-1)
                decoder_input = torch.multinomial(probabilities, num_samples=1)

                all_tokens = torch.cat((all_tokens, decoder_input), dim=0)

                if decoder_input == config.EOS_token:
                    break

                decoder_input = torch.unsqueeze(decoder_input, 0)

        return all_tokens


def evaluate(searcher, voc, sentence, max_length=config.MAX_LENGTH):
    indexes_batch = [prepare_data.indexesFromSentence(voc, sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(config.device)
    lengths = lengths.to(config.device)

    if type(searcher) == BeamSearchDecoder:
        tokens = searcher(input_batch, lengths, max_length)
        decoded_words = [voc.index2word[token.item()] for token in tokens]
        return decoded_words

    if type(searcher) == SamplingDecoder:
        tokens = searcher(input_batch, lengths, 100)
        decoded_words = [voc.index2word[token.item()] for token in tokens]
        return decoded_words

    tokens, scores = searcher(input_batch, lengths, max_length)
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(searcher, voc, searcher2=None, searcher3=None):
    while 1:
        try:
            input_sentence = input('> ')
            if input_sentence == 'q' or input_sentence == 'quit': break
            input_sentence = prepare_data.normalizeString(input_sentence)
            output_words = evaluate(searcher, voc, input_sentence)

            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot (greedy search):', ' '.join(output_words))

            if (searcher2 != None): 
                output_words2 = evaluate(searcher2, voc, input_sentence)
                output_words2[:] = [x for x in output_words2 if not (x == 'EOS' or x == 'PAD' or x == 'SOS')]
                print('Bot (beam search):', ' '.join(output_words2))

            if (searcher3 != None):
                output_words3 = evaluate(searcher3, voc, input_sentence)
                output_words3[:] = [x for x in output_words3 if not (x == 'EOS' or x == 'PAD')]
                print('Bot (sampling):', ' '.join(output_words3))

        except KeyError:
            print("Error: Encountered unknown word.")

def generateAnswer(question, searcher, voc):
    try:
        input_sentence = prepare_data.normalizeString(question)
        output_words = evaluate(searcher, voc, input_sentence)

        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD' or x == 'SOS')]
        return ' '.join(output_words)

    except KeyError:
        return "Error: Encountered unknown word."

def validate(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, batch_size):
    input_variable = input_variable.to(config.device)
    target_variable = target_variable.to(config.device)
    mask = mask.to(config.device)
    lengths = lengths.to("cpu")

    loss = 0
    print_losses = []
    n_totals = 0

    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    decoder_input = torch.LongTensor([[config.SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(config.device)

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    with torch.no_grad():
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            decoder_input = target_variable[t].view(1, -1)

            mask_loss, nTotal = train.maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    return sum(print_losses) / n_totals


def validate_batches(voc, val_pairs, encoder, decoder):
    encoder.eval()
    decoder.eval()

    random.shuffle(val_pairs)
    batches = [val_pairs[i:i + config.batch_size] for i in range(0, len(val_pairs), config.batch_size)]

    val_batches = [prepare_data.batch2TrainData(voc, batches[i])
                   for i in range(len(batches))]
    val_loss = 0

    for iteration in range(1, config.n_val_iteration + 1):
        val_batch = val_batches[iteration - 1]
        input_variable, lengths, target_variable, mask, max_target_len = val_batch

        loss = validate(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, config.batch_size)
        val_loss += loss

    return val_loss / config.n_val_iteration