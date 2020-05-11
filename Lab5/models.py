import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, laten_size, condition_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.condition_size = condition_size
        
        self.fc1 = nn.Linear(hidden_size+condition_size, laten_size)
        self.fc2 = nn.Linear(hidden_size+condition_size, laten_size)

        self.embedding = nn.Embedding(input_size, hidden_size+condition_size)
        self.lstm = nn.LSTM(hidden_size+condition_size, hidden_size+condition_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, -1, self.hidden_size+self.condition_size)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden
    
    def variational(self, hidden):
        return self.fc1(hidden[0]), self.fc2(hidden[0])

    def initHidden(self, embedded_tense, batch_size=64):
        embedded_tense = embedded_tense.to(device).view(1, batch_size, -1)
        zeros = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (torch.cat((zeros, embedded_tense), 2),
                torch.cat((zeros, embedded_tense), 2))

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, condition_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.condition_size = condition_size

        self.embedding = nn.Embedding(output_size, hidden_size+condition_size)
        self.lstm = nn.LSTM(hidden_size+condition_size, hidden_size+condition_size)
        self.out = nn.Linear(hidden_size+condition_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, -1, self.hidden_size+self.condition_size)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self, hidden_state, embedded_tense, batch_size):
        embedded_tense = embedded_tense.to(device).view(1, batch_size, -1)
        zeros = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (torch.cat((hidden_state, embedded_tense), 2),
                torch.cat((zeros, embedded_tense), 2))