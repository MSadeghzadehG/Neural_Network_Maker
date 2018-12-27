import numpy as np
import math


c= math.pow(10,-10)
def d1_cal(x,f): #calculates derivative f bY x
    return float(f(x) - f(x-c))/float(c)
def d2_cal(x1,x2,f): #calculates derivative f bY x1
    return float(f(x1,x2) - f(x1-c,x2))/float(c)


#activation functions
def relu(input):
    return max(0,input)

#loss fucntions
def mse_loss(output,Y):
    if type(output) != list:
        output = [output]
    if type(Y) != list:
        Y = [Y]
    if type(output) != np.array:
        output = np.array(output)
    if type(Y) != np.array:
        Y = np.array(Y)
    print(output[0])
    print(Y[0])
    o = sum([(Yp-Yt)**2 for Yp in output for Yt in Y])/(2*len(Y))
    print('loss = '+ str(o))
    return o 

class perceptron():
    
    def __init__(self,W):
        self.output = []
        self.d_output = []
        self.W = W  # Note: W is transposed 
        self.m = 0
        # self.output_f = f

    def cal_output(self,X):   
        self.output.append(relu(np.matmul([X],np.transpose([self.W]))[0][0]))
        def output_i(w,x):
            return w*x
        # print(self.output)
        self.d_output.append([d2_cal(self.W[i],X[i],output_i) for i in range(len(self.W))])
        # print(self.d_output)
        return self.output

    def update_weights(self,alpha,Y):
        def f(x1,x2):
            return (x1-x2)**2
        self.W = np.array([self.W[j] - alpha*sum([self.d_output[i][j]*d2_cal(self.output[i],Y[i],f) for i in range(len(Y))])/(2*len(Y)) for j in range(len(self.W))])
        self.output = []
        self.d_output = []
        return self.W

m = 100 #num of data
n = 3 #num of features
Y = [] #labels
X = np.zeros((m,3)) #inputs
for i in range(m):
    for j in range(n):
        X[i][j] = np.random.normal()
    # X[0][0]=1
    # X[0][1]=2
    # X[0][2]=3
    Y.append(sum(X[i][:]))
print(X[0])
print(Y[0])
input()
W = np.zeros(n)
for i in range(n):
    W[i]=np.random.normal()
print(W)
print(Y)
alpha = 0.1 #learning rate
epoch = 100 #num of epochs
p = perceptron(W)
for j in range(epoch):
    for i in range(m):
        # print(Y[i])
        p.cal_output(X[i])
    print(p.update_weights(alpha,Y))

