#Implementation of https://www.nature.com/articles/323533a0.pdf 
#Learning representations by back-propagating errors
#This code is a replication of their "XOR Test"
import numpy as np
class Network(object):
    def __init__ (self, input_sizes, hidden_size, output_size, lr, input_data, output_data):
        self.layers = []
        self.input_data = input_data
        self.output_data = output_data
        self.learning = lr
        self.input_sizes = input_sizes
        self.layers.append(self.input_sizes)
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]
        for h in hidden_size:
            self.layers.append(h)
        self.layers.append(output_size)
        self.weights = []
        self.biases = []

        for j in range(len(self.layers) - 1):
            w = np.random.randn(self.layers[j], self.layers[j+1])
            b = np.random.randn(1, self.layers[j+1])
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def sigmoidprime(self, a):
        s = self.sigmoid(a)
        return s * (1 - s)
    def forwardpass(self, X):
        self.activations = [X]
        self.zs = []
        a = X
        for W, b in zip(self.weights, self.biases):
            z = a @ W + b
            a = self.sigmoid(z)
            self.zs.append(z)
            self.activations.append(a)
        return a

    
    def backpropogation(self, y):
        deltas = [self.activations[-1] - y]
        for i in range(len(self.layers)-2, 0, -1):
            delta = (deltas[-1] @ self.weights[i].T) * self.sigmoidprime(self.zs[i-1])
            deltas.append(delta)
        deltas.reverse()
        grads_W = []
        grads_b = []
        for i in range(len(deltas)):
            grads_W.append(self.activations[i].T@deltas[i])
            grads_b.append(np.sum(deltas[i], axis=0, keepdims=True))


        for i in range(len(self.weights)):
            self.weights[i] -= self.learning * grads_W[i]
            self.biases[i] -= self.learning * grads_b[i]

    def train(self, epochs):
        for i in range(epochs):
            y_preds = self.forwardpass(self.input_data)
            self.backpropogation(self.output_data)
            if i % 1000 == 0:
                loss = -np.mean(self.output_data*np.log(y_preds + 1e-8) +
                            (1-self.output_data)*np.log(1-y_preds + 1e-8))
                print("epoch", i, "loss:", loss)
        return y_preds


X = np.array([
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [1, 1, 0, 0]
])
epoch = 500
y = np.array([0, 1, 1, 0]).reshape(-1,1)
net = Network(4, 3, 1, lr=0.1, input_data=X, output_data=y)
ypreds_initial = net.forwardpass(net.input_data)
print("Initial predictions:")
print(ypreds_initial)

ypreds_trained = net.train(50000)
print("Predictions after training:")
print(ypreds_trained)

