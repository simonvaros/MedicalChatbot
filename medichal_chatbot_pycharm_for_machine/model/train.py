import config
import random
import os
import torch
import torch.nn as nn
import prepare_data
import time

device = config.device

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=config.MAX_LENGTH):
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    # Lengths for rnn packing should always be on the cpu
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[config.SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < config.teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)

            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


######################################################################
# Training iterations
# ~~~~~~~~~~~~~~~~~~~
#

def train_iters(voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, save_dir, checkpoint,
                epoch):
    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()
    # Load batches for each iteration
    random.shuffle(pairs)
    batches = [pairs[i:i + config.batch_size] for i in range(0, len(pairs), config.batch_size)]

    training_batches = [prepare_data.batch2TrainData(voc, batches[i])
                        for i in range(len(batches))]

    # Initializations
    start_iteration = 1
    print_loss = 0
    epoch_loss = 0
    if config.loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    for iteration in range(start_iteration, config.n_iteration + 1):
        s = time.time()

        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, config.batch_size, config.clip)
        print_loss += loss
        epoch_loss += loss
        config.train_time += time.time() - s

        # Print progress
        if iteration % config.print_every == 0:
            print_loss_avg = print_loss / config.print_every
            print("Epoch: {} Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}; Train time: {}"
                  .format(epoch, iteration, iteration / config.n_iteration * 100, print_loss_avg, config.train_time))

            print_loss = 0

        # Save checkpoint
        if iteration % config.save_every == 0:
            directory = os.path.join(save_dir, config.model_name, config.corpus_name,
                                     '{}-{}_{}'.format(config.encoder_n_layers, config.decoder_n_layers, config.hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'epoch': epoch,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}_{}.tar'.format(epoch, iteration, 'checkpoint')))

    return epoch_loss / config.n_iteration
