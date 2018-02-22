import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

class LinearRegression:
    
    def __init__(self, input_file, output_file, eta=0.001, threshold=0.00001, max_iters = 1000000):
        self.x = self.ReadLinFile(input_file)
        self.y = self.ReadLinFile(output_file)
        self.num_examples = self.x.shape[0]
        self.theta = np.zeros(2)
        self.learning_rate = eta
        self.max_iterations = max_iters
        self.threshold = threshold
        self.animate_losses = []
    
    #function to read file with only 1 feature
    def ReadLinFile(self, file_name):
        fin = open(file_name, 'r')
        data = []
        for inp in fin:
            data.append(float(inp[:-1]))
        return np.array(data)
    
    #function to normalize data
    def NormalizeData(self):
        mu = np.mean(self.x)
        sigma = np.std(self.x)
        self.x = (self.x-mu)/sigma

    def BatchGradientDescent(self):
        converged = False
        x = np.c_[np.ones(self.num_examples),self.x]      #add a column of 1's to input x to handle intercept term
        iter = 0
        self.animate_losses = []
        self.theta = np.zeros(2)
        
        while((not converged) and iter < self.max_iterations):
            error = np.dot(x.T, np.dot(x, self.theta) - self.y)      #del J(theta)
            p = np.dot(x, self.theta) - self.y
            loss = np.dot(p.T, p)/2                    #J(theta)
            self.animate_losses.append([self.theta[0],self.theta[1],loss])     #store data for trajectory animation
            self.theta = self.theta - self.learning_rate * error         #gradient descent step
            
            converged = np.linalg.norm(error)<=self.threshold    #convergence criteria
            iter+=1
        
        self.animate_losses = np.array(self.animate_losses)
        return self.theta

def plot(data, plot_type, title="", options=[]):
    if plot_type == "scatter":                 #scatter plot. 'data' has (x,y) pairs. 'options' is empty
        plt.scatter(data[0], data[1], s=5)
    elif plot_type == "equation":              #equation plot. 'data' has equation string. 'options' has domain of 'x'
        x=np.linspace(options[0],options[1],100)
        plt.plot(x, eval(data))
    elif plot_type == "scatterequation":       #equation and scatter plot in same figure. 'data' has (x,y) pairs and equation string. 'options' has domain of 'x'
        plt.scatter(data[0], data[1], s=5)
        x=np.linspace(options[0],options[1],100)
        plt.plot(x, eval(data[2]))
    plt.title(title)
    plt.show()

#function for plotting mesh and contours
def plot_mesh_contour(type, lr):
    
    a=50    #number of points in mesh in one dimension
    x = np.linspace(0,2,a)
    y = np.linspace(-1,1,a)
    X,Y = np.meshgrid(x,y)
    z = np.empty(a*a)
    meshthetas = np.c_[X.reshape((a*a,1)),Y.reshape((a*a,1))]
    
    #compute 'z' values for mesh
    inputX = np.c_[np.ones(lr.num_examples),lr.x]
    for point in range(a*a):
        p = np.dot(inputX, meshthetas[point]) - lr.y
        loss = np.dot(p.T, p)/2              #J(theta)
        z[point] = loss
    z = z.reshape((a,a))
    
    fig = plt.figure()
    if(type=="mesh"):
        grph = fig.add_subplot(111, projection='3d')    #3d plot for mesh
    elif(type=="contour"):
        grph = fig.add_subplot(111)              #2d plot for contours
    grph.set_xlabel('theta 0')
    grph.set_ylabel('theta 1')
    
    #plot destination point of algorithm
    if(type=="mesh"):
        grph.set_zlabel('J(theta)')
        grph.plot_surface(X,Y,z,alpha=0.5)
        x1,y1,z1 = lr.animate_losses[-1]
        grph.plot([x1],[y1],z1,markerfacecolor='g', markeredgecolor='g', marker='o', markersize=3);

        point, = grph.plot([lr.animate_losses[0][0]], [lr.animate_losses[0][1]], lr.animate_losses[0][2],markerfacecolor='b',markeredgecolor='b', marker='o', markersize=1);
    
    #animate trajectory of the algorithm in mesh
    def animate_mesh(n,data,point):
        point.set_data(data[:n,:2].T)
        point.set_3d_properties(data[:n,2])
        return point
    
    #animate trajectory of the algorithm using contours
    def animate_contour(n):
        grph.clear()
        grph.contour(x,y,z,np.unique(lr.animate_losses[:n+1,2]).tolist(),linewidths=0.5)
        grph.set_xlabel('theta 0')
        grph.set_ylabel('theta 1')
        grph.title.set_text("Contours for Batch Gradient Descent trajactory with eta = " + str(lr.learning_rate))
        return grph
    
    try:
        if(type=="mesh"):
            anim = animation.FuncAnimation(fig, animate_mesh, lr.animate_losses.shape[0], fargs=(lr.animate_losses,point),interval=200)
        elif(type=="contour"):
            anim = animation.FuncAnimation(fig, animate_contour,lr.animate_losses.shape[0],interval=200)
        
        plt.title("Mesh and Batch Gradient Descent trajactory with eta = " + str(lr.learning_rate))
        plt.show()
    except:
        print("Done animation")

if __name__=='__main__':
    
    #create a linear regression object
    lr = LinearRegression("linearX.csv","linearY.csv")
    
    #normalize data to 0 mean and 1 standard deviation
    lr.NormalizeData()
    
    #compute parameters using batch gradient descent
    c,m = lr.BatchGradientDescent()         #y = mx + c
    print("Learning rate: "+str(lr.learning_rate))
    print("Stopping Criteria: Norm of Gradient of J(theta)<" + str(lr.threshold))
    print("\ny = "+str(m)+"x + "+str(c))
    
    #plot scatter points along with regression line
    plot([lr.x,lr.y,str(m)+"*x + "+str(c)],"scatterequation","Linear Regression",options=[-2,5])
    
    #plot meshgrid with trajactory
    plot_mesh_contour("mesh", lr)
    
    plot_mesh_contour("contour", lr)
    
    #plot contours for different learning rates
    etas = [0.001, 0.005, 0.009, 0.013, 0.017, 0.021, 0.025]
    
    for eta in etas:
        lr.learning_rate = eta
        c,m = lr.BatchGradientDescent()
        plot_mesh_contour("contour", lr)
        