import numpy as np
import pandas as pd
import torch
from torch.utils import data

def getData(mode):
    assert mode == 'train' or mode == 'test'
    if mode == 'train':
        data = pd.read_csv('./data/'+mode+'.txt', delimiter=' ', header=None)
    else:
        data = []
        with open('./data/test.txt','r') as fp:
            for line in fp:
                word = line.split(' ')
                word[1] = word[1].strip('\n')
                data.extend([word])
    return data


class Vocabuary():
    def __init__(self):
        self.word2index = {'SOS': 0, 'EOS': 1, 'PAD': 2, 'UNK': 3}
        self.index2word = {0: 'SOS', 1: 'EOS', 2: 'PAD', 3: 'UNK'}
        self.n_words = 4
        self.max_length = 0
        self.build_vocab(getData('train'))
        

    # input the training data and build vocabulary
    def build_vocab(self, corpus):        
        for idx in range(corpus.shape[0]):
            for word in corpus.iloc[idx,:]:
                if len(word) > self.max_length:
                    self.max_length = len(word)
                    
                for char in word:
                    if char not in self.word2index:
                        self.word2index[char] = self.n_words
                        self.index2word[self.n_words] = char
                        self.n_words += 1 
        self.max_length +=2
                    
    # convert word in indices
    def word2indices(self, word, add_eos=True, add_sos=True):
        indices = [self.word2index[char] if char in self.word2index else 3 for char in word]

        if add_sos:
            indices.insert(0, 0)
        if add_eos:
            indices.append(1)
            
        # padding input of same target into same length
        indices.extend([2]*(self.max_length-len(word)))
            
        return np.array(indices)
    
    # convert indices to word
    def indices2word(self, indices):
        # ignore indices after EOS
        new_indices = []
        for idx in indices:
            new_indices.append(idx)
            if idx == self.word2index['EOS']:
                break
                    
        word = [self.index2word[idx] for idx in new_indices if idx > 2 ]
        return ''.join(word)


class TenseLoader(data.Dataset):
    def __init__(self, mode, vocab):
        self.mode = mode   
        self.data = getData(self.mode)
        self.vocab = vocab
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_tensor = []
        if self.mode == 'train':
            for word in self.data.iloc[index,:]:
                data_tensor.append(torch.tensor(self.vocab.word2indices(word)))
        else:
            condition = [["sp", "p"], ["sp", "pg"], ["sp", "tp"], ["sp", "tp"], ["p", "tp"], 
                        ["sp", "pg"], ["p", "sp"], ["pg", "sp"], ["pg", "p"], ["pg", "tp"]]
            order = {'sp':0, 'tp':1, 'pg':2, 'p':3}
            
            input_tense = order[condition[index][0]]
            input_tensor = torch.tensor(self.vocab.word2indices(self.data[index][0]))
            target_tense = order[condition[index][1]]
            target_tensor = torch.tensor(self.vocab.word2indices(self.data[index][1]))
            
            data_tensor = [(input_tense, input_tensor), (target_tense, target_tensor)]

        return data_tensor                   