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

def generate_sinusoidal_data(n):
    x = np.linspace(0, 1, n)
    x = x.reshape(len(x), 1)
    y = np.sin(2 * np.pi * x)
    return x, y

def generate_linear_data(n):
    x = np.linspace(0, 1, n)
    x = x.reshape(len(x), 1)
    y = x
    return x, y


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
        self.weights.append(np.random.randn(n_input, n_hidden))
        for _ in range(n_layers - 1):
            self.weights.append(np.random.randn(n_hidden, n_hidden))
        self.weights.append(np.random.randn(n_hidden, n_output))
        #Initializing biases
        self.biases.append(np.zeros((1, n_hidden)))
        for _ in range(n_layers - 1):
            self.biases.append(np.zeros((1, n_hidden)))
        self.biases.append(np.zeros((1, n_output)))
        
        #self.biases[-1][0][0] = 2 #Increases all outputs by 2
    
    def tanh(self, x):
        '''
        Activation function
        '''
        #return np.maximum(0, x)
        return np.tanh(x)

    def tanh_derivate(self, x):
        return 1 - np.power(self.tanh(x), 2)

    def forward(self, x):
        '''
        Forward path trough the network
        '''
        self.z = []
        self.a = []
        self.z.append(x)
        self.a.append(x)

        for i in range(self.n_layers):
            self.z.append(np.dot(self.a[i], self.weights[i]) + self.biases[i])
            self.a.append(self.tanh(self.z[i + 1]))

        return self.a[-1]

    def mse(self, x, y):
        '''
        Mean square error:
            Calculates the mean of the squared differences between the predicted and actual values.
        '''
        return np.mean(np.power(self.forward(x) - y, 2))

    def backward(self, x, y):
        '''
        Backwards path trough the network, backpropagate the error
        '''
        self.grad_weights = []
        self.grad_biases = []
        self.deltas = []

        self.deltas.append(self.a[-1] - y) #How wrong is the nn for each pred

        self.grad_weights.append(np.dot(self.a[-2].T, self.deltas[-1]))
        self.grad_biases.append(np.sum(self.deltas[-1], axis=0, keepdims=True))

        for i in range(self.n_layers - 1, 0, -1):
            self.deltas.append(np.dot(self.deltas[-1], self.weights[i].T) * self.tanh_derivate(self.z[i]))
            self.grad_weights.append(np.dot(self.a[i - 1].T, self.deltas[-1]))
            self.grad_biases.append(np.sum(self.deltas[-1], axis=0, keepdims=True))

        self.grad_weights.reverse()
        self.grad_biases.reverse()
        self.deltas.reverse()

    def update(self, learning_rate):
        '''
        Update weights and biases
        '''
        for i in range(self.n_layers):
            self.weights[i] -= learning_rate * self.grad_weights[i]
            self.biases[i] -= learning_rate * self.grad_biases[i]

    def train(self, x, y, learning_rate, epochs=5000):
        '''
        Train the network
        '''
        for i in range(epochs):
            self.forward(x)
            self.backward(x, y)
            self.update(learning_rate)
            if i % 100 == 0:
                print(f'Epoch: {i}   {self.mse(x, y)}')

    def predict(self, x):
        y_pred = self.forward(x)
        return y_pred


x, y = generate_sinusoidal_data(100)
#x, y = generate_linear_data(100)

nn = NeuralNetwork(1, 3, 1, 3)

nn.train(x, y, 0.005, 10000)

y_pred = [np.mean(a) for a in nn.predict(x)]

plt.plot(x, y_pred, c='red')
plt.scatter(x, y, c='grey')
plt.show()