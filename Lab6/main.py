import numpy as np
import gc
import random
from PIL import Image
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from evaluator import evaluation_model
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# custom weights initialization called
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, latent_size=100, ngf=64):
        super(Generator, self).__init__()
        self.ngf = ngf
       
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d((latent_size+24), ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. 3 x 64 x 64
        )

    def forward(self, input, condition):
        input = torch.cat((input, condition.view(input.size(0), -1, 1, 1)), 1)
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.linear = nn.Linear(24, ndf*ndf)
        
        self.main = nn.Sequential(
            # input is 4 x 64 x 64
            nn.Conv2d(4, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, condition):
        condition = self.linear(condition).view(input.size(0), 1, self.ndf, self.ndf)
        input = torch.cat((input, condition), 1)
        return self.main(input)

def training(G, D, image_size, latent_size, lr_G, lr_D, batch_size, num_epochs):
    start = time.time()
    real_label = 1
    fake_label = 0
    
    # recording list
    G_losses = []
    D_losses = []
    
    # init dataloader 
    trainset = GANLoader('train', image_size=64)
    trainloader = data.DataLoader(trainset, batch_size, num_workers=2, shuffle=True)

    # init criterion & optimizer
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(D.parameters(), lr=lr_D, betas=(2e-4, 0.999))
    schedulerD = StepLR(optimizerD, step_size=50, gamma=0.5)
    optimizerG = optim.Adam(G.parameters(), lr=lr_G, betas=(2e-4, 0.999))
    schedulerG = StepLR(optimizerG, step_size=80, gamma=0.5)

    # init fixed noise
    fixed_noise = torch.randn(32, latent_size, 1, 1, device=device)
    
    for epoch in range(num_epochs):
        for idx, datas in enumerate(trainloader):
            b_size = datas[0].size(0)
            img = datas[0].to(device)
            condition = datas[1].to(device)
            
            #------part1 - train discriminator: maximize log(D(x)) + log(1 - D(G(z)))-----#
            ## all real batch
            D.zero_grad()
            label = torch.full((b_size,), real_label, device=device)
            output = D(img, condition).view(-1)
            
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item() 

            ## all fake batch
            noise = torch.randn(b_size, latent_size, 1, 1, device=device)
            fake = G(noise, condition)
            label.fill_(fake_label)
            
            output = D(fake.detach(), condition).view(-1)
            
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            errD = errD_real + errD_fake
            
            # Update D
            optimizerD.step()
            
            #------part2 - train generator: maximize log(D(G(z)))-----#
            G.zero_grad()
            label.fill_(real_label)
            output = D(fake, condition).view(-1)
            
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            
            # Update G
            optimizerG.step()

        # lr schedule
        schedulerG.step()
        schedulerD.step()

        acc,_ = testing(G, fixed_noise, latent_size, batch_size)
        if acc>0.5556 :
            print ("Model save...")
            torch.save(G, "./models/G_{:.4f}.ckpt".format(acc))
            torch.save(D, "./models/D_{:.4f}.ckpt".format(acc))
            
        if epoch % 1 == 0:
            print('%s (%d %d%%) Accuracy: %.4f Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                          % (timeSince(start, (epoch+1)/num_epochs), epoch, epoch/num_epochs * 100, 
                              acc, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        collected = gc.collect()
        torch.cuda.empty_cache()
    return G_losses, D_losses

def testing(G, noise=None, latent_size=100, batch_size=32):
    E = evaluation_model()
    
    img_list = []
    acc_list = []

    if noise is None:
        noise = torch.randn(batch_size, latent_size, 1, 1, device=device)
    
    # init dataloader 
    testset = GANLoader('test')
    testloader = data.DataLoader(testset, batch_size, num_workers=2)
    
    with torch.no_grad():
        for condition in testloader:
            condition = condition.to(device)
            # b_size = condition.size(0)
            fake = G(noise, condition).detach()
            
            acc_list.append(E.eval(fake, condition))
            img_list.append(make_grid(fake, nrow=8, padding=2, normalize=True).cpu())

    return sum(acc_list)/len(acc_list), img_list

if __name__ == '__main__':
    image_size = 64 
    latent_size = 100
    lr_G = 0.0002
    lr_D = 0.0001
    batch_size = 128
    num_epochs = 700

    manualSeed = 87
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    G = Generator(latent_size=100, ngf=64).to(device)
    G.apply(weights_init)
    D = Discriminator(ndf=64).to(device)
    D.apply(weights_init)
    G_losses, D_losses = training(G, D, image_size, latent_size, lr_G, lr_D, batch_size, num_epochs)
    show_result(G_losses, D_losses)

    G = torch.load('./models/G_0.6111.ckpt')
    acc, imgs = testing(G)
    fig = plt.figure(figsize=(15,15))
    plt.imshow(np.transpose(imgs[0],(1,2,0)))
    print ("Accuracy: %.4f"%(acc))
    plt.show()