import numpy as np
import torch
from torch.utils import data
import json

def getData(mode):
    assert mode == 'train' or mode == 'test' or mode == 'new_test'
    dataset = json.load(open('./data/'+mode+'.json', 'r'))
    inputs = []
    labels = []
    for data in dataset:
        inputs.append(data['input'])
        labels.append(data['target'])
    return inputs, labels

class Vocabuary():
    def __init__(self):
        self.word2index = {'SOS': 0, 'EOS': 1, 'PAD': 2, 'UNK': 3}
        self.index2word = {0: 'SOS', 1: 'EOS', 2: 'PAD', 3: 'UNK'}
        self.n_words = 4
        self.max_length = 0
        self.build_vocab(getData('train')[0])
        

    # input the training data and build vocabulary
    def build_vocab(self, corpus):
        for words in corpus:
            for word in words:
                if len(word) > self.max_length:
                    self.max_length = len(word)
                    
                for char in word:
                    if char not in self.word2index:
                        self.word2index[char] = self.n_words
                        self.index2word[self.n_words] = char
                        self.n_words += 1                      
                    
    # convert word in indices
    def word2indices(self, word, add_eos=False, add_sos=False):
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
        word = [self.index2word[idx] for idx in indices if idx > 2 ]
        return ''.join(word)


class SpellingLoader(data.Dataset):
    def __init__(self, mode, vocab):
        self.mode = mode   
        self.inputs, self.targets = self.convert_pair()
        self.vocab = vocab
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input = torch.tensor(self.vocab.word2indices(self.inputs[index]))
        target = torch.tensor(self.vocab.word2indices(self.targets[index]))
        return input, target
    
    # convert (multi-input)+target into multi-(input+target) pair
    def convert_pair(self):
        input_data, label_data = getData(self.mode)
        inputs_list = []
        labels_list = []
        for inputs, label in zip(input_data, label_data):
            for input in inputs:
                inputs_list.append(input)
                labels_list.append(label)
        return inputs_list, labels_list                              