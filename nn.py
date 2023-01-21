import numpy as np
import matplotlib.pyplot as plt

'''
-generate data
-neural network
    -init
    -forward
    -backward
    -activation
    -error
    -update
    -train
    -predict
'''
np.random.seed(0)

def generate_data(n):
    x = np.linspace(0, 1, n)
    x = x.reshape(len(x), 1)
    y = np.sin(2 * np.pi * x)
    return x, y

x, y = generate_data(100)

#plt.plot(x, y)
#plt.show()

class NeuralNetwork:
    def __init__(self, n_input, n_hidden, n_output, n_layers):
        '''
        n_input = dimension of the input
        n_hidden = number of hidden neurons in hidden layer
        n_output = dimension of the output
        n_layers = number of hidden layers
        '''
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers

        self.weights = []
        self.biases = []

        self.grad_weights = []
        self.grad_biases = []
        self.deltas = []

        #Initializing weights
        self.weights.append(np.random.randn(self.n_input, self.n_hidden))
        for _ in range(self.n_layers - 1):
            self.weights.append(np.random.randn(self.n_hidden, self.n_hidden))
        self.weights.append(np.random.randn(self.n_hidden, n_output))
        #Initializing biases
        self.biases.append(np.zeros((1, n_hidden)))
        for _ in range(self.n_layers - 1):
            self.biases.append(np.zeros((1, n_hidden)))
        self.biases.append(np.zeros((1, n_output)))
        
        #self.biases[-1][0][0] = 2 #Increases all outputs by 2
    
    def relu(self, x):
        '''
        ReLU function:
            If x is positive return x else return 0
        '''
        return np.maximum(0, x)

    def relu_derivate(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def forward(self, inputs):
        '''
        Forward path trough the network
        '''
        a = inputs
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.relu(z)
        return a

    def mse(self, x, y):
        '''
        Mean square error:
            Calculates the mean of the squared differences between the predicted and actual values.
        '''
        return np.mean(np.power(self.forward(x) - y, 2))

    def backward(self, x, y): #Revisar
        pred = self.forward(x)
        error = self.mse(x, y)
        self.grad_weights.append(error * self.relu_derivate(pred))

    def update(self):
        pass

    def train(self, x, y, epochs=2000):
        for _ in range(epochs):
            self.forward(x)
            for i in range(len(self.weights)):
                self.weights[i] += 0.00007 
            print(self.mse(x, y))

    def predict(self, x):
        return self.forward(x)


nn = NeuralNetwork(1, 10, 1, 2)

nn.train(x, y)



y_pred = nn.predict(x)

#acuracy = np.mean(y_pred==y)
#print(acuracy)

plt.plot(x, y_pred)
plt.plot(x, y)
plt.show()