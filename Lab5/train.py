import random
import time
import gc
import math
import torch
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from datahelper import *
from models import *
from evaluate import evaluate
SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def show_result(scores, losses):  
    bleu_score, gaussian_score = scores
    c_loss, kl_loss = losses
    
    plt.figure(figsize=(10, 6))
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title("CrossEntropy Loss Curve", fontsize=18)
    plt.plot(c_loss)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title("KL Loss Curve", fontsize=18)
    plt.plot(kl_loss)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.ylabel("Score")
    plt.xlabel("Epochs")
    plt.title("BLEU Score Curve", fontsize=18)
    plt.plot(bleu_score)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.ylabel("Score")
    plt.xlabel("Epochs")
    plt.title("Gaussian Score Curve", fontsize=18)
    plt.plot(gaussian_score)
    plt.show()


def condition_embedding(condition_size, batch_size=1, condition=None):
    order = {'sp':0, 'tp':1, 'pg':2, 'p':3}
    one_hot = {0:[1, 0, 0, 0], 1:[0, 1, 0, 0], 2:[0, 0, 1, 0], 3:[0, 0, 0, 1]}
    
    if condition != None:
        embedded_tense = torch.tensor(one_hot[condition], dtype=torch.float).view(1, -1)
    else: 
        embedded_tense = []
        for o in order.values():
            embedded = torch.tensor(one_hot[o], dtype=torch.float).view(1, -1)
            embedded_cp = embedded
            
            # expand batch dim, (4, condition_size) to (4, batch_size, condition_size)
            for i in range(batch_size-1):
                embedded = torch.cat((embedded, embedded_cp), 0)
            embedded_tense.append(embedded)
        
    return embedded_tense
                     

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

def kl_annealing(epochs, mode):
    assert mode == "monotonic" or mode == "cyclical"
    if mode == "monotonic":
        if epochs > 500:
            KLD_weight = 0.1
        else:
            KLD_weight = 0.0002 * epochs
    else:
        if epochs%500 > 250: 
            KLD_weight = 0.1
        else:
            KLD_weight = 0.0004 * (epochs%500) 
    return KLD_weight

def compute_kl_loss(mu, logvar):
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


# save model every epoch
def train(input_tensors, encoder, decoder, encoder_optimizer, decoder_optimizer, 
          criterion, epochs, condition_size, teacher_forcing_ratio):
    batch_size = input_tensors[0].size(0)
    # get 4 tense embedding tensor
    embedded_tenses = condition_embedding(condition_size, batch_size)
    
    # loss for 4 tense
    kl_loss_total = 0
    ce_loss_total = 0
    
    
    # 4 tense iteration
    for index, embedded_tense in enumerate(embedded_tenses):
        # embedded_tense.to(device)
        input_tensor = input_tensors[index].to(device) 
        input_tensor = input_tensor.transpose(0, 1) # (seq_len, batch_size)
    
        # init encoder hidden state and cat condition
        encoder_hidden = encoder.initHidden(embedded_tense, batch_size)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # calculate number of time step
        input_length = input_tensor.size(0)

        loss = 0
        ce_loss = 0

        encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
            
        # reparameterization trick
        mu, logvar = encoder.variational(encoder_hidden)
        reparameterized_state = reparameterize(mu, logvar)
        reparameterized_state = decoder.in_layer(reparameterized_state)
        
        # calculate kl loss
        kl_loss = compute_kl_loss(mu, logvar) / batch_size
        kl_loss_total += kl_loss
        loss += kl_annealing(epochs, "cyclical") * kl_loss
        
        # init decoder hidden state and cat condition
        decoder_hidden = decoder.initHidden(reparameterized_state, embedded_tense, batch_size)
        
        decoder_input = torch.tensor([[SOS_token] for i in range(batch_size)], device=device)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        #----------sequence to sequence part for decoder----------#
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(input_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden) 
                _, indx = torch.max(decoder_output, 1)
                ce_loss += criterion(decoder_output, input_tensor[di])
                decoder_input = input_tensor[di]  # Teacher forcing
                
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(input_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden) 
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                ce_loss += criterion(decoder_output, input_tensor[di])

        loss += ce_loss
        ce_loss_total += (ce_loss.item()/input_length)
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

    return ce_loss_total/4, kl_loss_total/4


def trainIters(encoder, decoder, vocab, n_iters, print_every=1000, plot_every=100, 
               batch_size=64, learning_rate=0.01, laten_size=32, condition_size=8, teacher_forcing_ratio=1.0):
    start = time.time()

    # Reset every print_every, for print log
    print_ce_loss_total = 0  
    print_kl_loss_total = 0
    # Reset every plot_every, for plot curve
    crossentropy_losses = []
    kl_losses = []
    plot_ce_loss_total = 0
    plot_kl_loss_total = 0
    # scores
    gaussian_scores = []
    bleu_scores = []
    

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # create dataloader
    trainset = TenseLoader('train', vocab)
    trainloader = data.DataLoader(trainset, batch_size, shuffle = True)

    criterion = nn.CrossEntropyLoss()

    for iter in range(1, n_iters + 1):
        for input_tensors in trainloader:

            ce_loss, kl_loss = train(input_tensors, encoder, decoder, encoder_optimizer, decoder_optimizer, 
                                     criterion, (iter-1), condition_size, teacher_forcing_ratio)
            print_ce_loss_total += ce_loss
            print_kl_loss_total += kl_loss
            plot_ce_loss_total += ce_loss
            plot_kl_loss_total += kl_loss
            
        # evaluate and save model
        bleu_score, gaussian_score = evaluate(encoder, decoder, vocab, laten_size=laten_size, condition_size=condition_size, plot_pred=False)
        bleu_scores.append(bleu_score)
        gaussian_scores.append(gaussian_score)
        if bleu_score > 0.8 and gaussian_score > 0.4:
            print ("Model save...")
            torch.save(encoder, "./models/encoder_{:.4f}_{:.4f}_c.ckpt".format(gaussian_score, bleu_score))
            torch.save(decoder, "./models/decoder_{:.4f}_{:.4f}_c.ckpt".format(gaussian_score, bleu_score))

        if iter % print_every == 0:
            print_ce_loss_avg = print_ce_loss_total / print_every
            print_kl_loss_avg = print_kl_loss_total / print_every
            print('%s (%d %d%%) CE Loss: %.4f, KL Loss: %.4f, BLEU score: %.2f, Gaussian score: %.2f' % 
                  (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_ce_loss_avg, print_kl_loss_avg, bleu_score, gaussian_score))
            print_ce_loss_total = 0
            print_kl_loss_total = 0

        crossentropy_losses.append(plot_ce_loss_total)
        plot_ce_loss_total = 0
        kl_losses.append(plot_kl_loss_total)
        plot_kl_loss_total = 0
        
        collected = gc.collect()
            
    return (bleu_scores, gaussian_scores), (crossentropy_losses, kl_losses)