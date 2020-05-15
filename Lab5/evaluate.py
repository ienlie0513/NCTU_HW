import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from datahelper import *
from models import *
SOS_token = 0
EOS_token = 1

#compute BLEU-4 score
def compute_bleu(output, reference):
    """
    reference = 'accessed'
    output = 'access'
    return BLEU score
    """
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)

def BLEU_predict(encoder, decoder, vocab, batch_size=1, condition_size=8, plot_pred=False):
    from train import condition_embedding, reparameterize
    testset = TenseLoader('test', vocab)
    outputs = []
    
    with torch.no_grad():
    
        for idx in range(len(testset)):
            input_tense = testset[idx][0][0]
            input_embedded_tense = condition_embedding(condition_size, condition=input_tense) # (1, condition_szie)
            input_tensor = testset[idx][0][1].to(device) # (seq_len)

            target_tense = testset[idx][1][0]
            target_embedded_tense = condition_embedding(condition_size, condition=target_tense)
            target_tensor = testset[idx][1][1].to(device)

            batch_size = 1
            
            # transpose tensor from (batch_size, tense, seq_len) to (tense, seq_len, batch_size)
            input_tensor = input_tensor.view(-1, 1) # (seq_len, batch_size)
            target_tensor = target_tensor.view(-1, 1)

            # init encoder hidden state and cat condition
            encoder_hidden = encoder.initHidden(input_embedded_tense, batch_size)

            # calculate number of time step
            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)
            

            encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

            # reparameterization trick
            mu, logvar = encoder.variational(encoder_hidden)
            reparameterized_state = reparameterize(mu, logvar)
            reparameterized_state = decoder.in_layer(reparameterized_state)

            # init decoder hidden state and cat condition
            decoder_hidden = decoder.initHidden(reparameterized_state, target_embedded_tense, batch_size)
            
            decoder_input = torch.tensor([[SOS_token] for i in range(batch_size)], device=device)
            
            output = torch.zeros(target_length, batch_size)

            #----------sequence to sequence part for decoder----------#
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden) 
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                output[di] = decoder_input
            
            # transpose tensor from (target_length, batch_size) to (batch_size, target_length)
            output = output.transpose(0, 1).view(-1)
         
            outputs.append(vocab.indices2word(output.data.numpy()))

    return outputs

# print the prediction and return the bleu score
def BLEU_score(prediction, plot_pred=False):
    data = getData('test')

    bleu_total = 0
    for idx in range(len(prediction)):
        bleu_total += compute_bleu(prediction[idx], data[idx][1])

        if plot_pred:
            output = "\ninput:  {}\ntarget: {}\npred:   {}".format(data[idx][0], data[idx][1], prediction[idx])
            print ("="*30+output)

    return bleu_total/len(prediction)


def Gaussian_predict(encoder, decoder, vocab, batch_size=64, laten_size=32, condition_size=8, plot_pred=False):
    from train import condition_embedding
    outputs = []

    with torch.no_grad():    
        batch_size = 100

        # sample 100 Gaussian
        laten_variable = torch.randn((batch_size, laten_size), device=device).view(1, batch_size, -1)

        # get 4 tense embedding tensor
        embedded_tenses = condition_embedding(condition_size , batch_size)

        # record outputs
        output_tensors = torch.zeros(4, vocab.max_length, batch_size)# (tense, seq_len, batch_size)

        # 4 tense iteration
        for index, embedded_tense in enumerate(embedded_tenses):

            decoder_input = torch.tensor([[SOS_token] for i in range(batch_size)], device=device)

            output = torch.zeros(vocab.max_length, batch_size)

            # init decoder hidden state and cat condition
            decoder_hidden = decoder.initHidden(decoder.in_layer(laten_variable), embedded_tense, batch_size)

            #----------sequence to sequence part for decoder----------#
            for di in range(vocab.max_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden) 
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                output[di] = decoder_input

            # get predict tensors
            output_tensors[index] = output

        # transpose tensor from (tense, seq_len, batch_size) to (batch_size, tense, seq_len)
        output_tensors = output_tensors.permute(2, 0, 1)

        # convert input into string
        for idx in range(batch_size):
            outputs.append([vocab.indices2word(tense.data.numpy()) for tense in output_tensors[idx]])

    return outputs

# compute generation score
def Gaussian_score(predictions, plot_pred=False):
    """
    the order should be : simple present, third person, present progressive, past
    predictions = [['consult', 'consults', 'consulting', 'consulted'],...]
    return Gaussian_score score
    """
    score = 0
    words_list = []
    with open('./data/train.txt','r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for idx, t in enumerate(predictions):
            if plot_pred:
                print (t)
            for idxj, i in enumerate(words_list):
                if t == i:
                    score += 1
    return score/len(predictions)


def evaluate(encoder, decoder, vocab, batch_size=64, laten_size=32, condition_size=8, plot_pred=False):
    # predict train.txt for gaussian score
    predictions = Gaussian_predict(encoder, decoder, vocab, batch_size=batch_size, laten_size=laten_size, condition_size=condition_size, plot_pred=plot_pred)
    
    # compute Gaussian score
    gaussian_score = Gaussian_score(predictions, plot_pred=plot_pred)
    if plot_pred:
        print ("Gaussian score: %.2f"%gaussian_score)

    # predict test.txt for bleu score
    predictions = BLEU_predict(encoder, decoder, vocab, batch_size=1, condition_size=condition_size, plot_pred=False)
            
    # compute BLEU score
    bleu_score = BLEU_score(predictions, plot_pred=plot_pred)
    if plot_pred:
        print ("BLEU score: %.2f"%bleu_score)
    
    return bleu_score, gaussian_score