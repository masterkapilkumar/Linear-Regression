import numpy as np
from matplotlib import pyplot

class LinearRegression:
    
    def __init__(self, input_file, output_file):
        self.x = self.ReadLinFile(input_file)
        self.y = self.ReadLinFile(output_file)
        self.num_examples = self.x.shape[0]
        self.theta = np.zeros(2)
        self.learning_rate = 0.001
        self.max_iterations = 100000
        self.threshold = 0.0000001
    
    def ReadLinFile(self, file_name):
        fin = open(file_name, 'r')
        data = []
        for inp in fin:
            data.append(float(inp[:-1]))
        return np.array(data)
    
    def NormalizeData(self):
        mu = np.mean(self.x)
        sigma = np.std(self.x)
        self.x = (self.x-mu/sigma)

    def BatchGradientDescent(self):
        converged = False
        self.x = np.c_[np.ones(self.num_examples),self.x]
        iter = 0
        while((not converged) and iter < self.max_iterations):
            error = np.dot(self.x.T, np.dot(self.x, self.theta) - self.y)/self.num_examples
            self.theta = self.theta - self.learning_rate * error
            
            converged = abs(error[0])<=self.threshold and abs(error[1])<=self.threshold
            iter+=1
        
        return self.theta


if __name__=='__main__':
    
    lr = LinearRegression("datasets/linearX.csv","datasets/linearY.csv")
    lr.NormalizeData()
    #print(lr.x)
    #print(lr.y)
    #print(lr.num_examples)
    m,c = lr.BatchGradientDescent()         #y = mx + c
    print("y = "+str(m)+"x + "+str(c))