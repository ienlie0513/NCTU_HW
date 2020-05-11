import torch
from datahelper import *
from train import *
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
SOS_token = 0
EOS_token = 1
#----------Hyper Parameters----------#
hidden_size = 256
laten_size = 32
condition_size = 8
#The number of vocabulary
batch_size = 64
vocab_size = 30
teacher_forcing_ratio = 1.0
empty_input_ratio = 0.1
KLD_weight = 0.0
learning_rate = 0.02

if __name__ == '__main__':
	# train
	encoder = EncoderRNN(vocab_size, hidden_size, laten_size, condition_size).to(device)
	decoder = DecoderRNN(laten_size, vocab_size, condition_size).to(device)
	vocab = Vocabuary()
	scores, losses = trainIters(encoder, decoder, vocab, 1000, print_every=10, plot_every=1, 
                            batch_size=batch_size, learning_rate=learning_rate, laten_size=laten_size, condition_size=condition_size, teacher_forcing_ratio=teacher_forcing_ratio)
	show_result(scores, losses)

	# evaluate
	# encoder = torch.load("./models/encoder_0.8785.ckpt")
	# decoder = torch.load("./models/decoder_0.8785.ckpt")
	_, _  = evaluate(encoder, decoder, vocab, batch_size=64, plot_pred=True)