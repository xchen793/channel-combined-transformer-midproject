import random
import time
import os
import torch
import torch.nn as nn
from torch import optim
from seq2seq_att_model import prepareData, BLEU
from seq2seq_att_model import MAX_LENGTH, SOS_token, EOS_token
from util import timeSince, showPlot
from seq2seq_att_model import EncoderRNN, AttnDecoderRNN
from seq2seq_att_model import evaluateRandomly, evaluate, showAttention, tensorsFromPair


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

teacher_forcing_ratio = 0.5



input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0



    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)


    decoder_hidden = encoder_hidden


    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            
            if decoder_input.item() == EOS_token:
                break


    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length



def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    former_loss = 9999
    # use SGD as optimizer
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs))
                      for i in range(n_iters)]

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss



        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('epoch:%d  %s (%d%%) loss:%.4f' % (iter, timeSince(start, iter / n_iters),
                                          iter / n_iters * 100, print_loss_avg))

            # save the best model,and cover the original model
            if print_loss_avg < former_loss:
                former_loss = print_loss_avg
                if os.path.exists("models") == False:
                    os.mkdir("models")
                torch.save(encoder.state_dict(), 'models\encoder.pkl')
                torch.save(decoder.state_dict(), 'models\decoder.pkl')
                print("save the best model successful!")

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(input_lang, output_lang, encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


if __name__ == '__main__':
    hidden_size = 256  
    encoder1 = EncoderRNN(input_lang.wordIndex, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.wordIndex, dropout_p=0.1).to(device)

    trainIters(encoder1, attn_decoder1, 100, print_every=10)

    evaluateRandomly(input_lang, output_lang, pairs, encoder1, attn_decoder1)


    evaluateAndShowAttention("elle a cinq ans de moins que moi .")
    evaluateAndShowAttention("elle est trop petit .")
    evaluateAndShowAttention("je ne crains pas de mourir .")
    evaluateAndShowAttention("c est un jeune directeur plein de talent .")
