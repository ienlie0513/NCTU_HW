import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# predict and calculate accuracy
def get_predict(model, loader):
    predictions = None
    total = 0
    correct = 0
    
    with torch.no_grad():
        for data, label in loader:
            # forward pass
            output = model.forward(data)
            output = nn.Softmax(dim=1)(output)
            
            # get output label
            _, pred = torch.max(output, 1)

            total += label.shape[0]
            correct += (pred == label).sum().item()
            
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
                
    acc = correct / total
            
    return predictions, acc

class EEGModel(nn.Module):
    def __init__(self, activation_function="relu"):
        # choose activation
        assert activation_function == "relu" or activation_function == "elu" or activation_function == "leakyrelu"
        self.activation_function = activation_function
        activation_dict = {"relu": nn.ReLU(), 
                           "elu": nn.ELU(alpha=1.0), 
                           "leakyrelu": nn.LeakyReLU(negative_slope=0.01)
                          }
        
        # init conv
        super(EEGModel, self).__init__()
        self.firstConv = nn.Sequential(
                            nn.Conv2d(1,16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
                            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        )
        self.depthwiseConv = nn.Sequential(
                                nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
                                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                activation_dict[activation_function],
                                nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
                                nn.Dropout(p=0.25)
                            )
        self.separableConv = nn.Sequential(
                                nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
                                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                activation_dict[activation_function],
                                nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
                                nn.Dropout(p=0.25)
                            )
        self.classify = nn.Sequential(
                            nn.Linear(in_features=736, out_features=2, bias=True)
                        )


    def forward(self, x):
        x = self.firstConv(x) 
        x = self.depthwiseConv(x) 
        x = self.separableConv(x) 
        x = x.view(x.size(0), -1) # flatten x, [B, 32, 1, 23] => [B, 736] 
        x = self.classify(x) 
        return x

    def train(self, trainset, testset, batch_size=64, epochs=300, lr=0.001):
        batch_size = batch_size
        epochs = epochs
        lr = lr
        
        # creare dataloaders
        trainloader = DataLoader(trainset, batch_size=batch_size)
        testloader = DataLoader(testset, batch_size=batch_size)
        
        # set optimizer, loss function
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # record loss and train accuracy
        train_acc = []
        test_acc = []

        for epoch in range(epochs):
            running_loss = 0.0

            for data, label in trainloader:
                # set parameter gradient into zero
                optimizer.zero_grad()

                # forward pass
                output = self.forward(data)

                # backward
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                # record the loss
                running_loss += loss.item()

            # record accuracy for each epochs
            _, train = get_predict(self, trainloader)
            train_acc.append(train)
            _, test = get_predict(self, testloader)
            test_acc.append(test)

            # print accuracy every 10 epochs
            if epoch % 10 == 0:
                print ("Epoch: %s, train accuracy: %s, test accuract: %s"%(epoch, train, test))
        
        # test 
        print ("Accuracy of EEG model with %s: %10s"%(self.activation_function, test_acc[-1]))
        
        return train_acc, test_acc


class DeepConvModel(nn.Module):
    def __init__(self, activation_function="relu"):
        # choose activation
        assert activation_function == "relu" or activation_function == "elu" or activation_function == "leakyrelu"
        self.activation_function = activation_function
        activation_dict = {"relu": nn.ReLU(), 
                           "elu": nn.ELU(alpha=1.0), 
                           "leakyrelu": nn.LeakyReLU(negative_slope=0.01)
                          }
        
        # init conv
        super(DeepConvModel, self).__init__()
        self.firstConv = nn.Sequential(
                            nn.Conv2d(1, 25, kernel_size=(1, 5)),
                            nn.Conv2d(25, 25, kernel_size=(2, 1)),
                            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
                            activation_dict[activation_function],
                            nn.MaxPool2d(kernel_size=(1, 2)),
                            nn.Dropout(p=0.5)
                        )
        self.secondConv = nn.Sequential(
                                nn.Conv2d(25, 50, kernel_size=(1, 5)),
                                nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
                                activation_dict[activation_function],
                                nn.MaxPool2d(kernel_size=(1, 2)),
                                nn.Dropout(p=0.5)
                            )
        self.thirdConv = nn.Sequential(
                                nn.Conv2d(50, 100, kernel_size=(1, 5)),
                                nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
                                activation_dict[activation_function],
                                nn.MaxPool2d(kernel_size=(1, 2)),
                                nn.Dropout(p=0.5)
                            )
        self.fourthConv = nn.Sequential(
                                nn.Conv2d(100,200, kernel_size=(1, 5)),
                                nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
                                activation_dict[activation_function],
                                nn.MaxPool2d(kernel_size=(1, 2)),
                                nn.Dropout(p=0.5)
                            )
        self.classify = nn.Sequential(
                            nn.Linear(in_features=8600, out_features=2, bias=True)
                        )


    def forward(self, x):
        x = self.firstConv(x)
        x = self.secondConv(x)
        x = self.thirdConv(x)
        x = self.fourthConv(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x

    def train(self, trainset, testset, batch_size=64, epochs=300, lr=0.001):
        batch_size = batch_size
        epochs = epochs
        lr = lr
        
        # creare dataloaders
        trainloader = DataLoader(trainset, batch_size=batch_size)
        testloader = DataLoader(testset, batch_size=batch_size)
        
        # set optimizer, loss function
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # record loss and train accuracy
        train_acc = []
        test_acc = []

        for epoch in range(epochs):
            running_loss = 0.0

            for data, label in trainloader:
                # set parameter gradient into zero
                optimizer.zero_grad()

                # forward pass
                output = self.forward(data)

                # backward
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                # record the loss
                running_loss += loss.item()

            # record accuracy for each epochs
            _, train = get_predict(self, trainloader)
            train_acc.append(train)
            _, test = get_predict(self, testloader)
            test_acc.append(test)

            # print accuracy every 10 epochs
            if epoch % 10 == 0:
                print ("Epoch: %s, train accuracy: %s, test accuract: %s"%(epoch, train, test))
        
        # test 
        print ("Accuracy of DeepConv model with %s: %10s"%(self.activation_function, test_acc[-1]))
        
        return train_acc, test_acc


if __name__ == '__main__':
    eegmodel = EEGModel(activation_function="relu")
    print(eegmodel) 

    deepmodel = DeepConvModel(activation_function="relu")
    print(deepmodel) 