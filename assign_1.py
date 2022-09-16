import numpy as np
import matplotlib.pyplot as plt
class assign1:
        def __init__(self):

                self.a = 0.5 #Learning rate

                #Basis Function Variables
                self.mu = 0 #Mean Value
                self.phi = 1 #Variance
        
        def init_theta(self, m, n):
                self.theta = np.random.random((m, n))

        def plotGraph(self, xdata, ydata, title, xlabel, ylabel):
                plt.xlabel(xlabel, fontsize=18)
                plt.ylabel(ylabel, fontsize=18)
                plt.title(title, fontsize=24)
                og = plt.plot(xdata, ydata, 'o')
        
        def hypothesis(self, xdata):
                return np.dot(self.theta, xdata)

        def costFunction(self, xdata, ydata):
                self.costf = 0
                for i in range(len(self.theta)):
                        self.costf += (self.hypothesis(xdata)-ydata)**2
                self.costf = 0.5 * self.costf

        def gradientDescent(self, xdata, ydata, j):
                self.theta[0, j] = self.theta[0, j] + self.a * (ydata - self.hypothesis(xdata)) * xdata[j]

        def sigmoid(self, data):
                [x,y] = data.shape
                out = np.zeros((x, 1))
                out[:,0] = np.exp(((data[:, 0] - self.mu) ** 2) / (2*self.phi**2)) 
                return np.concatenate((data, out), axis=1)

        def polynomialBasis(self, data, order):
                [x,y] = data.shape
                out = np.zeros((x, 1))
                out[:,0] = data[:, 0] ** order
                return np.concatenate((data, out), axis=1) 
        
        def concatOnes(self, data):
                [x,y] = data.shape
                out = np.ones((x, 1))
                return np.concatenate((out, data), axis=1)
        
        def csvParse(self, csvFile):
                file = open(csvFile, 'rb')
                file = file.readlines()[1:]
                data = np.loadtxt(file, delimiter=',')

                [x,y] = data.shape

                x_train = data[0:(x//2), 1:(y-1)]
                y_train = data[0:(x//2), (y-1)]
                x_test = data[(x//2):x, 1:(y-1)]
                y_test = data[(x//2):x, (y-1)]
                
                return x_train, y_train, x_test, y_test
        
        def predictionError(self, actual, predicted):
                return (np.abs(actual - predicted) / np.abs(actual))

        def localWeight(self, query, tao):
                [m, n] = self.x_train.shape
                self.w = np.zeros((m, 1))
                
                for i in range(m):
                        self.w[i] = np.exp(-(self.x_train[i] - query) ** 2 / (2*tao**2))

        def costFunctionLR(self):
                [m, n] = self.w.shape

                for i in range(m):
                       self.costfLR += self.w[i] * (self.theta.T * self.x_train[i] - self.y_train[i])
                
                self.constfLR = 0.5 * self.constfLR 
        
        def featureSelection(self, data, columns):
                [x,y] = data.shape
                data_out = np.ones((x, 1))

                for i in columns:
                        data_out = np.concatenate((data_out, data[:, i].reshape((x, 1))), axis=1)

                return data_out
        
        def hypothesisDataGeneration(self, lowerbounds, upperbounds, order):
                xdata = np.linspace(lowerbounds, upperbounds, 1000)
                data = xdata 

                data = data.reshape((1000, 1))

                for i in range(2, order+1):
                        data = self.polynomialBasis(data, i)

                data = self.concatOnes(data) #Input data with the order desired to be concatenated into the original dataset.

                h = []

                for i in range(1000):
                        h.append(self.hypothesis(data[i]))

                return xdata, h

if __name__ == '__main__':
        a1 = assign1()

        x_test = np.load('x_test.npy')
        x_train = np.load('x_train.npy')
        y_test = np.load('y_test.npy')
        y_train = np.load('y_train.npy')

        plt.ion()
        plt.figure(figsize=[15, 12])
        plt.xlabel('x', fontsize=18)
        plt.ylabel('y', fontsize=18)
        plt.title('Question 1 - Linear Regression', fontsize=24)
        plt.ylim(-50, 70)

        #Training the model for assignment 1
        order = 5 #Changes the polynomial basis functions.

        for i in range(2, order+1):
                x_train = a1.polynomialBasis(x_train, i) #Input data with the order desired to be concatenated into the original dataset.

        x_train = a1.concatOnes(x_train) #Input data with the order desired to be concatenated into the original dataset.

        [n, m] = x_train.shape

	#Initialize the Parameters based upon the dimensionality of the training dataset
        a1.init_theta(1, m)

        #Generate the initial hypothesis function
        x, h = a1.hypothesisDataGeneration(np.min(x_train[:, 1]),np.max(x_train[:,1]), order )

        #Plot the initial training data and hypothesis function.
        og, hypth = plt.plot(x_train[:, 1], y_train, 'o', x, h, 'r-')

        hypth.set_label('Hypothesis')
        og.set_label('Training Data Set')
        plt.legend(loc='lower right')
        plt.grid()
        
        print(f'Initialzed theta array is: {a1.theta}')

        for j in range(m):
                print(f'Feature set selection: {j}')
                for i in range(n):
                        a1.gradientDescent(x_train[i], y_train[i], j)
                        x, h = a1.hypothesisDataGeneration(np.min(x_train[:, 1]),np.max(x_train[:,1]), order)
                        hypth.set_ydata(h)
                        plt.draw()
                        plt.pause(0.1)
        
        print(f'Theta array after training: {a1.theta}')

        #Testing the model for Question 1
        for i in range(2, order+1):
                x_test = a1.polynomialBasis(x_test, order) #Input data with the order desired to be concatenated into the original dataset.
        x_test = a1.concatOnes(x_test) #Input data with the order desired to be concatenated into the original dataset.

	# Collect the prediction error for each point in the test data set.
        e = []

        for i in range(n):
                y = a1.hypothesis(x_test[i])
                e.append(a1.predictionError(y_test[i], y))

        x = np.linspace(1, n+1)

        plt.figure()
        a1.plotGraph(x, e, 'Prediction Error', 'Point Number', 'Prediction Error')
        plt.savefig('TestRun.png')
        plt.show()

        #Question 2
 
	#Updating the data for Question 2 from the CSV file. 
        [x_train, y_train, x_test, y_test] = a1.csvParse('Assignment1_Q2_Data.csv')

        x_train = a1.featureSelection(x_train, [1,2])
        x_test = a1.featureSelection(x_test, [1,2])

        [n, m] = x_train.shape

	#Initialize the Parameters based upon the dimensionality of the training dataset
        a1.init_theta(1, m)

        print(f'Initialzed theta array is: {a1.theta}')

        for j in range(m):
                print(f'Feature set selection: {j}')
                for i in range(n):
                        a1.gradientDescent(x_train[i], y_train[i], j)

        
        print(f'Theta array after training: {a1.theta}')

        #Question 3

