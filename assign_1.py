import numpy as np
import matplotlib.pyplot as plt
class assign1:
        def __init__(self):
                self.x_test = np.load('x_test.npy')
                self.x_train = np.load('x_train.npy')
                self.y_test = np.load('y_test.npy')
                self.y_train = np.load('y_train.npy')

                self.a = 0.5 #Learning rate

                #Basis Function Variables
                self.mu = 0 #Mean Value
                self.phi = 1 #Variance
        
        def init_theta(self, m, n):
                self.theta = np.random.random((m, n))

        def plotGraph(self, xdata, ydata, title, xlabel, ylabel, fileN):
                plt.figure(figsize=[15, 12])
                plt.plot(xdata, ydata, 'o')
                plt.xlabel(xlabel, fontsize=18)
                plt.ylabel(ylabel, fontsize=18)
                plt.title(title, fontsize=24)
                plt.savefig(fileN)
                plt.show()
        
        def hypothesis(self, xdata):
                return np.dot(self.theta, xdata)

        def costFunction(self, xdata, ydata):
                self.costf = 0
                for i in range(len(self.theta)):
                        self.costf +=  (self.hypothesis(xdata)-ydata)**2
                self.costf = 0.5 * self.costf

        def gradientDescent(self,xdata, ydata):
                for j in range((self.theta).size):
                        error = np.abs((ydata - self.hypothesis(xdata))) / np.abs(ydata)
                        self.theta[j] = self.theta[j] + self.a * (ydata - self.hypothesis(xdata)) * xdata
                        return error

        def sigmoid(self, data):
                [x,y] = data.shape
                out = np.zeros((x, 1))
                out[:,0] = np.exp(((data[:, 0] - self.mu) ** 2) / (2*self.phi**2)) 
                return np.concatenate((data, out), axis=1)

        def polynomial_basis(self, data, order):
                [x,y] = data.shape
                out = np.zeros((x, 1))
                out[:,0] = data[:, 0] ** order
                return np.concatenate((data, out), axis=1) 
        
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
        
        def prediction_error(self, actual, predicted):
                return (np.abs(actual - predicted) / np.abs(actual))

        def local_weight(self, query, tao):
                [m, n] = self.x_train.shape
                self.w = np.zeros((m, 1))
                
                for i in range(m):
                        self.w[i] = np.exp(-(self.x_train[i] - query)**2 / 2*tao**2)

        def cost_function_LR(self):
                [m, n] = self.w.shape

                for i in range(m):
                       self.costfLR += self.w[i] * (self.theta.T * self.x_train[i] - self.y_train[i])
                
                self.constfLR = 0.5 * self.constfLR 

if __name__ == '__main__':
        a1 = assign1()

        a1.plotGraph(a1.x_train, a1.y_train, 'Initial Test', 'x', 'y', 'init.png')

        #Training the model for assignment 1
        a1.init_theta(1, 1)

        plt.figure()
        for i in range(len(a1.x_train)):
                predictionError = a1.gradientDescent(a1.x_train[i], a1.y_train[i])
                # plt.scatter(i, predictionError)
                # plt.pause(0.005)
        
        # plt.show()

        a1.x_train = a1.polynomial_basis(a1.x_train, 2) #Input data with the order desired to be concatenated into the original dataset.
        a1.x_train = a1.sigmoid(a1.x_train) #Input data with the order desired to be concatenated into the original dataset.



        #Testing the model for Question 1
        e = []

        for i in range(len(a1.x_test)):
                y = a1.hypothesis(a1.x_test[i])
                e[i] = a1.prediction_error(a1.y_test[i], y)
        
        x = np.linspace(1, len(a1.x_test))

        a1.plotGraph(x, e, 'Prediction Error', 'Point Number', 'Prediction Error', 'prediction_error.png')



        #Question 2
 
 
 
	#Updating the data for Question 2 from the CSV file. 
        [a1.x_train, a1.y_train, a1.x_test, a1.y_test] = a1.csv_parse('Assignment1_Q2_Data.csv')

        #Question 3