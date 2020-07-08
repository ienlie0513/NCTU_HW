import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
from dataloader import read_bci_data
from models import EEGModel, DeepConvModel

def show_result(accuracy, model):            
    label = ["train_relu", "train_leakyrelu", "train_elu", "test_relu", "test_leakyrelu", "test_elu"]
    
    plt.figure(figsize=(10,6))
    
    plt.ylabel("Accuracy(%)")
    plt.xlabel("Epochs")
    plt.title("Activation function comparison("+model+")", fontsize=18)
    
    for idx, acc in enumerate(accuracy):
        plt.plot(acc, label=label[idx])
    
    plt.legend()
    plt.show()



if __name__ == '__main__':
	# set cuda
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("device:", device)

	# load data
	train_data, train_label, test_data, test_label = read_bci_data()
	
	# init trainset
	train_data_tensors = torch.tensor(train_data, device=device)
	train_label_tensors = torch.tensor(train_label, dtype=torch.long, device=device)
	trainset = TensorDataset(train_data_tensors, train_label_tensors)

	# init testset
	test_data_tensors = torch.tensor(test_data, device=device)
	test_label_tensors = torch.tensor(test_label, dtype=torch.long, device=device)
	testset = TensorDataset(test_data_tensors, test_label_tensors)


	# EEG ReLU model init
	eeg_relu = EEGModel(activation_function="relu")
	eeg_relu = eeg_relu.to(device)
	eeg_relu.double()
	eeg_relu_train_acc, eeg_relu_test_acc = eeg_relu.train(trainset, testset, batch_size=64, epochs=300, lr=0.001)

	# EEG LeakyReLU model init
	eeg_leakyrelu = EEGModel(activation_function="leakyrelu")
	eeg_leakyrelu = eeg_leakyrelu.to(device)
	eeg_leakyrelu.double()
	eeg_leakyrelu_train_acc, eeg_leakyrelu_test_acc = eeg_leakyrelu.train(trainset, testset, batch_size=12, epochs=300, lr=0.0004)

	# EEG ELU model init
	eeg_elu = EEGModel(activation_function="elu")
	eeg_elu = eeg_elu.to(device)
	eeg_elu.double()
	eeg_elu_train_acc, eeg_elu_test_acc = eeg_elu.train(trainset, testset, batch_size=12, epochs=300, lr=0.0005)

	# show EEG train and test result
	accuracy = [eeg_relu_train_acc, eeg_leakyrelu_train_acc, eeg_elu_train_acc, 
	            eeg_relu_test_acc, eeg_leakyrelu_test_acc, eeg_elu_test_acc]
	show_result(accuracy, model="EEGNet")


	# DeepConvNet ReLU model init
	deep_relu = DeepConvModel(activation_function="relu")
	deep_relu = deep_relu.to(device)
	deep_relu.double()
	deep_relu_train_acc, deep_relu_test_acc = deep_relu.train(trainset, testset, batch_size=64, epochs=300, lr=0.0003) # 0.001

	# DeepConvNet LeakyReLU model init
	deep_leakyrelu = DeepConvModel(activation_function="leakyrelu")
	deep_leakyrelu = deep_leakyrelu.to(device)
	deep_leakyrelu.double()
	deep_leakyrelu_train_acc, deep_leakyrelu_test_acc = deep_leakyrelu.train(trainset, testset, batch_size=64, epochs=300, lr=0.0003)

	# DeepConvNet ELU model init
	deep_elu = DeepConvModel(activation_function="elu")
	deep_elu = deep_elu.to(device)
	deep_elu.double()
	deep_elu_train_acc, deep_elu_test_acc = deep_elu.train(trainset, testset, batch_size=64, epochs=300, lr=0.0003)

	# show DeepConvNet train and test result
	accuracy = [deep_relu_train_acc, deep_leakyrelu_train_acc, deep_elu_train_acc, 
				deep_relu_test_acc, deep_leakyrelu_test_acc, deep_elu_test_acc]
	show_result(accuracy, model="DeepConvNet")
