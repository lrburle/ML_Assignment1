import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

class assign1:
        def __init__(self):
                self.x_test = np.load('x_test.npy')
                self.x_train = np.load('x_train.npy')
                self.y_test = np.load('y_test.npy')
                self.y_train = np.load('y_train.npy')

                self.w = np.random.rand(100,1)
                self.a = 0.3 #Learning rate

        def plotGraph(self, xdata, ydata, title, xlabel, ylabel, fileN):
                plt.figure(figsize=[18, 12])
                plt.plot(xdata, ydata, 'o')
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(title)
                plt.show()
                plt.savefig(fileN)
        
        def hypothesis(self, xdata):
                for i in range(len(self.w)):
                        self.h = self.w[i] * xdata

        def costFunction(self, xdata, ydata):
                for i in range(len(self.w)):
                        self.costf = 0.5 * (self.h[i]*(xdata)-ydata)**2

        def gradientDescent(self,xdata, ydata):
                for j in range(len(self.w)):
                        self.w[j] = self.w[j] + self.a * (ydata[j] - self.h[j]) * xdata[j]

if __name__ == '__main__':
        a1 = assign1()

        a1.plotGraph(a1.x_train, a1.y_train, 'Initial Test', 'x', 'y', 'init.png')

        print(a1.x_test)
