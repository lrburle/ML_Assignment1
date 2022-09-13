import numpy as np
import matplotlib.pyplot as plt

class assign1:
        def __init__(self):
                self.x_test = np.load('x_test.npy')
                self.x_train = np.load('x_train.npy')
                self.y_test = np.load('y_test.npy')
                self.y_train = np.load('y_train.npy')

                self.w = np.random.random((1, 1)) #Yields a initialized weights of shape 10 x 1 normalized between 0 and 1
                self.a = 0.5 #Learning rate
                self.mu = 0
                self.phi = 1

        def plotGraph(self, xdata, ydata, title, xlabel, ylabel, fileN):
                plt.figure(figsize=[15, 12])
                plt.plot(xdata, ydata, 'o')
                plt.xlabel(xlabel, fontsize=18)
                plt.ylabel(ylabel, fontsize=18)
                plt.title(title, fontsize=24)
                plt.savefig(fileN)
                plt.show()
        
        def hypothesis(self, xdata):
                return np.dot(self.w, xdata)

        def costFunction(self, xdata, ydata):
                for i in range(len(self.w)):
                        self.costf = 0.5 * (self.h[i]*(xdata)-ydata)**2

        def gradientDescent(self,xdata, ydata):
                for j in range((self.w).size):
                        error = np.abs((ydata - self.hypothesis(xdata))) / np.abs(ydata)
                        self.w[j] = self.w[j] + self.a * (ydata - self.hypothesis(xdata)) * xdata
                        return error

        def sigmoid(self, data):
                return np.exp(((data - self.mu) ** 2) / (2*self.phi**2))

        def polynomial_basis(self, data, order):
                out = data[:][:, 0] ** order
                return np.concatenate(data, out) 
        
        def csv_parse(self, csvFile):
                file = open(csvFile, 'rb')
                file = file.readlines()[1:]
                data = np.loadtxt(file, delimiter=',')

                [x,y] = data.shape

                x_train = data[0:(x//2), 1:(y-1)]
                y_train = data[0:(x//2), (y-1)]
                x_test = data[(x//2):x, 1:(y-1)]
                y_test = data[(x//2):x, (y-1)]
                
                return x_train, y_train, x_test, y_test

if __name__ == '__main__':
        a1 = assign1()

        a1.plotGraph(a1.x_train, a1.y_train, 'Initial Test', 'x', 'y', 'init.png')

        #Training the model for assignment 1
        plt.figure()

        for i in range(len(a1.x_train)):
                predictionError = a1.gradientDescent(a1.x_train[i], a1.y_train[i])
                plt.scatter(i, predictionError)
                plt.pause(0.005)
        
        plt.show()

        [x_train, y_train, x_test, y_test] = a1.csv_parse('Assignment1_Q2_Data.csv')

        print(x_train)



