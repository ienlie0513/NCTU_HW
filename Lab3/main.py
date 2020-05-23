import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from dataloader import RetinopathyLoader
from functions import *
import json

if __name__ == "__main__":
	# create RetinopathyLoader
	trainset = RetinopathyLoader("./data/", "train")
	testset = RetinopathyLoader("./data/", "test")

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("device:", device)


	# resnet 18 with pretraining
	model_ft_18 = models.resnet18(pretrained=True)
	model_ft_18.fc = nn.Linear(512, 5)
	model_ft_18.to(device)
	model_ft_18.float()
	train_ft_18, test_ft_18 = train(model_ft_18, trainset, testset, batch_size=8, epochs=12 , lr=1e-3, model_name="resnet18_ft") 
	# record accuracy
	with open('acc_list', 'a+') as outfile:
		json.dump({"train_ft_18": train_ft_18, "test_ft_18": test_ft_18}, outfile)
		outfile.write("\n")

	# resnet 18 with pretraining
	model_ft_50 = models.resnet50(pretrained=True)
	model_ft_50.fc = nn.Linear(2048, 5)
	model_ft_50.to(device)
	model_ft_50.float()
	train_ft_50, test_ft_50 = train(model_ft_50, trainset, testset, batch_size=4, epochs=5 , lr=1e-3, model_name="resnet50_ft") 
	# record accuracy
	with open('acc_list', 'a+') as outfile:
		json.dump( {"train_ft_50": train_ft_50, "test_ft_50": test_ft_50}, outfile)
		outfile.write("\n")

	# resnet 18 with pretraining
	model_18 = models.resnet18(pretrained=False)
	model_18.fc = nn.Linear(512, 5)
	model_18.to(device)
	model_18.float()
	train_18, test_18 = train(model_18, trainset, testset, batch_size=8, epochs=12 , lr=1e-3, model_name="resnet18") 
	# record accuracy
	with open('acc_list', 'a+') as outfile:
		json.dump({"train_18": train_18, "test_18": test_18}, outfile)
		outfile.write("\n")

	# resnet 18 with pretraining
	model_50 = models.resnet50(pretrained=False)
	model_50.fc = nn.Linear(2048, 5)
	model_50.to(device)
	model_50.float()
	train_50, test_50 = train(model_50, trainset, testset, batch_size=4, epochs=5 , lr=1e-3, model_name="resnet50") 
	# record accuracy
	with open('acc_list', 'a+') as outfile:
		json.dump({"train_50": train_50, "test_50": test_50}, outfile)
		outfile.write("\n")


	# plot result comparison figuire
	acc_list = []
	with open('acc_list', 'r') as outfile:
		for line in outfile:
			dic = json.loads(line)
			acc_list.extend([acc for acc in dic.values()])
		

	resnet18_acc = [acc_list[0][:10], acc_list[4][:10], acc_list[1][:10], acc_list[5][:10]]
	show_result(resnet18_acc, "ResNet18")
	resnet50_acc = [acc_list[2], acc_list[6], acc_list[3], acc_list[7]]
	show_result(resnet50_acc, "ResNet50")

	# load best model and plot the confusion matrix
	model_names = ["resnet18_ft", "resnet50_ft", "resnet18", "resnet50"]
	testloader = DataLoader(testset, batch_size=8)
	for model_name in model_names:
		model = torch.load("./models/"+model_name)
		_, acc = get_predict(model, testloader, True)
		print ("Accuracy of %s: %s"%(model_name, acc))

	