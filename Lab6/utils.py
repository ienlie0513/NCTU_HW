import numpy as np
import json
import time
import math
from PIL import Image
import torch
from torch.utils import data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
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

def show_result(G_losses, D_losses, save = False, path = './img/training_loss.png'):      
    plt.figure(figsize=(10, 6))
    x = range(len(G_losses))
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title("Training Loss Curve", fontsize=18)
    plt.plot(x, G_losses, label='G_loss')
    plt.plot(x, D_losses, label='D_loss')
    plt.legend()
    plt.show()
    
    if save:
        plt.savefig(path) 


def get_data(mode):
    assert mode == 'train' or mode == 'test' or mode == 'my_test'
    data = json.load(open('./data/'+mode+'.json', 'r'))
    if mode == 'train':
        data = [i for i in data.items()]
    return data

def get_objectDic():
    return json.load(open('./data/objects.json', 'r'))

class GANLoader(data.Dataset):
    def __init__(self, mode, image_size=64):
        self.mode = mode   
        self.data = get_data(mode)
        self.obj_dict = get_objectDic()
        self.transformation = transforms.Compose([
                                  transforms.Resize(image_size),
                                  transforms.CenterCrop(image_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == 'train': 
            img_name = self.data[index][0]
            objects = [self.obj_dict[obj] for obj in self.data[index][1]]

            # image preprocess
            img = np.array(Image.open('./data/iclevr/'+img_name))[...,:-1]
            img = self.transformation(Image.fromarray(img))
            
            # condition embedding - one hot
            condition = torch.zeros(24)
            condition = torch.tensor([v+1 if i in objects else v for i,v in enumerate(condition)])
            
            data = (img, condition)
        else:
            # condition embedding - one hot
            objects = [self.obj_dict[obj] for obj in self.data[index]]
            condition = torch.zeros(24)
            data = torch.tensor([v+1 if i in objects else v for i,v in enumerate(condition)])
        
        return data     