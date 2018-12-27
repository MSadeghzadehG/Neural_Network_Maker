import numpy as np
import math


c= math.pow(10,-10)
def d1_cal(x,f): #calculates derivative f by x
    return float(f(x) - f(x-c))/float(c)
def d2_cal(x1,x2,f): #calculates derivative f by x1
    return float(f(x1,x2) - f(x1-c,x2))/float(c)


#activation functions
def relu(input):
    return max(0,input)

#loss fucntions
def mse_loss(output,y):
    if type(output) != list:
        output = [output]
    if type(y) != list:
        y = [y]
    if type(output) != np.array:
        output = np.array(output)
    if type(y) != np.array:
        y = np.array(y)
    return sum([(yp-yt)**2 for yp in output for yt in y])/(2*len(y))

class perceptron():

    def __init__(self,W):
        self.output = 0
        self.d_output = []
        self.W = W
        # self.output_f = f

    def cal_output(self,X):    
        self.output = relu(np.matmul(np.transpose(self.W),X)[0][0])
        def output_i(w,x):
            return w*x
        self.d_output = [ d2_cal(self.W[i],X[i],output_i) for i in range(len(X))]
        # print(self.d_output)
        return self.output

    def update_weights(self,alpha,y):
        return [self.W[i][0] - alpha*d2_cal(self.output,y,mse_loss)*self.d_output[i] for i in range(len(self.W))]

X = np.zeros((3,1))
W = X
for i in range(3):
    X[i]=i+1
    W[i]=i+1
print(X)
print(W)
p = perceptron(W)
print(p.cal_output(X))
print(p.update_weights(1,6))