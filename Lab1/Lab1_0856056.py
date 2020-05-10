import numpy as np
import matplotlib.pyplot as plt
import sys

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def derivate_sigmoid(x):
    return np.multiply(x, 1.0-x)


class GenData:
    @staticmethod
    def _gen_linear(n=100):
        """ Data generation (Linear)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data = np.random.uniform(0, 1, (n, 2))

        inputs = []
        labels = []

        for point in data:
            inputs.append([point[0], point[1]])

            if point[0] > point[1]:
                labels.append(0)
            else:
                labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

    @staticmethod
    def _gen_xor(n=100):
        """ Data generation (XOR)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data_x = np.linspace(0, 1, n // 2)

        inputs = []
        labels = []

        for x in data_x:
            inputs.append([x, x])
            labels.append(0)

            if x == 1 - x:
                continue

            inputs.append([x, 1 - x])
            labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

    @staticmethod
    def fetch_data(mode, n):
        """ Data gather interface

        Args:
            mode (str): 'Linear' or 'XOR', indicate which generator is used.
            n (int):    the number of data points generated in total.
        """
        assert mode == 'Linear' or mode == 'XOR'

        data_gen_func = {
            'Linear': GenData._gen_linear,
            'XOR': GenData._gen_xor
        }[mode]

        return data_gen_func(n)


class HiddenLayer:
    def __init__(self, input_size, hidden_size):
        self.W = np.random.uniform(-1, 1, [input_size, hidden_size])
        self.b = np.zeros(hidden_size)
        self.x = None
        self.dW = None
        self.db = None
            
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        self.out = sigmoid(out)
        return self.out
    
    def backward(self, dout):
        dout = dout * derivate_sigmoid(self.out)
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        return dx

class NeuralNetwork():
    def __init__(self, input_size, hidden_size, num_step=2000, learning_rate=0.5, print_interval=100):
        self.num_step = num_step
        self.learning_rate = learning_rate
        self.print_interval = print_interval
        
        # initiate layers
        self.layers = []
        self.layers.append(HiddenLayer(input_size, hidden_size))
        self.layers.append(HiddenLayer(hidden_size, hidden_size))
        self.lastLayer = HiddenLayer(hidden_size, 1)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return self.lastLayer.forward(x)
        
    def backward(self):
        # calculate gradients
        dout = self.lastLayer.backward(self.error)
        layers = list(self.layers)
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
                
        # update weights
        self.lastLayer.W -= self.learning_rate * self.lastLayer.dW
        self.lastLayer.b -= self.learning_rate * self.lastLayer.db
        for layer in self.layers:
            layer.W -= self.learning_rate * layer.dW
            layer.b -= self.learning_rate * layer.db
        
    def training(self, inputs, labels):
        loss_list = []
        acc_list = []
        for epochs in range(self.num_step):
            for idx in range(inputs.shape[0]):
                self.output = self.forward(inputs[idx:idx+1, :])
                self.error = self.output - labels[idx]
                self.backward()
            
            if epochs % self.print_interval == 0:
                loss = self.loss(inputs, labels)
                acc = self.accuracy(inputs, labels)
                loss_list.append(loss)
                acc_list.append(acc)
                print ("Epochs: %4s, loss: %.10f, acc: %.10f"%
                       (epochs, loss, acc))
                
        return loss_list, acc_list

    # calculate the loss
    def loss(self, inputs, labels):
        return np.sum(np.square(self.predict(inputs)-labels)) / labels.shape[0]
    
    # reutrn the predict value
    def predict(self, inputs):
        output = []
        for idx in range(inputs.shape[0]):
            output.append(self.forward(inputs[idx:idx+1, :]))
        output = np.array(output).reshape(inputs.shape[0], 1)
        return output
        
    # return the predict label
    def get_output(self, inputs):
        return  np.around(self.predict(inputs))
    
    # calculate accuracy
    def accuracy(self, inputs, labels):
        return np.sum(self.get_output(inputs) == label) / float(inputs.shape[0])
         
    @staticmethod
    def show_result(x, y, pred_y):
        assert x.shape[0] == y.shape[0]
        assert x.shape[0] == pred_y.shape[0]

        plt.figure()

        plt.subplot(1, 2, 1)
        plt.title('Ground truth', fontsize=18)
        for i in range(x.shape[0]):
            if y[i] == 0:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1], 'bo')

        plt.subplot(1, 2, 2)
        plt.title('Predict result', fontsize=18)
        for i in range(x.shape[0]):
            if pred_y[i] == 0:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1], 'bo')

        plt.show()
        
    @staticmethod
    def show_loss_acc(loss, acc):
        plt.figure()
        
        plt.subplot(1, 2, 1)
        plt.title('Loss', fontsize=18)
        plt.plot(loss)
        plt.subplot(1, 2, 2)
        plt.title('Accuracy', fontsize=18)
        plt.plot(acc)
        plt.show()


if __name__ == '__main__':
    data_type = sys.argv[1]
    data, label = GenData.fetch_data(data_type, 70)

    net = NeuralNetwork(input_size=data.shape[1], hidden_size=100, num_step=2000, learning_rate=0.1, print_interval=100)
    loss, acc = net.training(data, label)

    print(net.predict(data))

    pred_y = net.get_output(data)
    NeuralNetwork.show_result(data, label, pred_y)
    NeuralNetwork.show_loss_acc(loss, acc)