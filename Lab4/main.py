import torch
from datahelper import *
from train import *
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#----------Hyper Parameters----------#
hidden_size = 256
#The number of vocabulary
vocab_size = 30 
learning_rate = 0.05
teacher_forcing_ratio = 1.0

if __name__ == '__main__':
	# train
	encoder1 = EncoderRNN(vocab_size, hidden_size).to(device)
	decoder1 = DecoderRNN(hidden_size, vocab_size).to(device)
	vocab = Vocabuary()
	scores, losses = trainIters(encoder1, decoder1, vocab, 100, print_every=10, plot_every=1, teacher_forcing_ratio=0.7)
	show_result(scores, losses)

	# evaluate
	encoder = torch.load("./models/encoder_0.8785.ckpt")
	decoder = torch.load("./models/decoder_0.8785.ckpt")
	predictions, avg_bleu = evaluate(encoder, decoder, vocab, plot_pred=True)
	print ("BLEU-4 score: %.4f"%avg_bleu)