#Aarav Lala - my implementation of a Single Layer perceptron
#Inspired from this paper: https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf
import numpy as np
import random
class SinglePerceptron(object):
    def __init__(self):
        self.learning = 0.1
        self.epoch = 200
        self.input = np.array([
            [0, 0, 0, 0],  
            [0, 1, 0, 0],  
            [1, 0, 0, 0],  
            [1, 1, 0, 0]   
                        ])
        
        self.output = np.array([0, 0, 1, 1])
        self.weights = np.zeros(4)
        np.random.seed(42)

        self.bias = np.random.uniform(0.1, 2.99)
    def step(self, z):
        if z<0:
            return(0)
        else:
            return(1)

    def feedforward(self):
        for j in range(self.epoch):
            storage = []
            for i in range(len(self.input)):
                z = np.dot(self.input[i], self.weights) + self.bias  
                a = self.step(z)
                storage.append(a)
                self.weights += self.learning * (self.output[i] - a) * self.input[i]
                self.bias += self.learning * (self.output[i] - a)
            Loss = np.sum((self.output - storage) ** 2)
            accuracy = np.mean(storage == self.output)
            print(f"Epoch {j+1}: Loss={Loss:.3f}, Accuracy={accuracy*100:.1f}%, Weights={self.weights}, Bias={self.bias:.3f}") 
            if Loss == 0:
                print("Training complete!")
                break

P = SinglePerceptron()
P.feedforward()





