import numpy as np
import math


c= math.pow(10,-10)
def d1_cal(x,f): #calculates derivative f bY x
    return float(f(x) - f(x-c))/float(c)

def d2_cal(x1,x2,f): #calculates derivative f bY x1
    return float(f(x1,x2) - f(x1-c,x2))/float(c)

# def mse_loss(output,Y):
#     if type(output) != list:
#         output = [output]
#     if type(Y) != list:
#         Y = [Y]
#     if type(output) != np.array:
#         output = np.array(output)
#     if type(Y) != np.array:
#         Y = np.array(Y)
#     # print(output[0])
#     # print(Y[0])
#     return sum([(Yp-Yt)**2 for Yp in output for Yt in Y])/(2*len(Y))

def mse_loss(y_pred,y):
    return (y_pred-y)**2

#activation functions
def relu(input):
    # print(input)
    return max(0,input)
  

class Perceptron():

    def __init__(self,W,activation):
        self.output = []
        self.d_output = []
        self.W = W  # Note: W is transposed
        self.W_next = W 
        self.m = 0
        self.c = 0
        self.next = []
        self.pre = []
        self.activation = activation

    def cal_output(self,X):
        # print('x:' +str(X))
        # print('w:' +str(np.transpose(self.W)))
        self.m = np.matmul([X],np.transpose([self.W]))[0][0]   
        self.output.append(self.activation(self.m))
        def output_i(w,x):
            return w*x
        # print(self.output)
        self.d_output.append([d2_cal(self.W[i],X[i],output_i) for i in range(len(self.W))])
        print(self.d_output)
        return self.output

    def cal_next_W(self,alpha,Y):
        self.W_next = np.array([self.W[j] - alpha*sum([c*self.d_output[i][j]*d2_cal(self.output[i],Y[i],self.f) for i in range(len(Y))])/(2*len(Y)) for j in range(len(self.W))])
    
    def update_weights(self):
        self.W = self.W_next
        self.output = []
        self.d_output = []
        return self.W


class Network:
    def __init__(self):
        self.nodes = []
        self.types = []
        self.loss = 1

    def add_node(self,p,t,n):
        self.nodes.append(p)
        self.types.append(t)

    def connect_nodes(self,p1,p2,n_of_w):
        p1.next.append((p2,n_of_w))
        p2.pre.append((p1,n_of_w))
    
    def forward_prop(self,X):
        to_update = set([])
        for p in self.nodes:
            if self.types[self.nodes.index(p)] == 'input':
                to_update.add(p)
        # cfc = 0
        while (len(to_update)!=0):
            now = to_update
            for i in range(len(X)):
                for p in now:
                    # print(p.W)
                    if self.types[self.nodes.index(p)] == 'input':
                        print(p.cal_output(X[i]))
                    else :
                        print(p.cal_output([x.output[i] for (x,w) in p.pre]))
                    # print('m:'+str(p.m))
                    # print(cfc)
                    # cfc += 1
            to_update = set([])
            for p in now:
                if self.types[self.nodes.index(p)] != 'output': #can do not check this?
                    for (n,w) in p.next:
                        to_update.add(n)
    
    def backward_prop(self,alpha,Y):
        to_update = set([])
        for p in self.nodes:
            if self.types[self.nodes.index(p)] == 'output':
                to_update.add(p)
        
        self.loss = sum([mse_loss(to_update[0].output[i],Y[i]) for i in range(len(Y))])/(2*len(Y))
        self.d_loss = [d2_cal(self.output[i],Y[i],mse_loss) for i in range(len(Y))]
        

        while (len(to_update)!=0):
            now = to_update
            for i in range(len(X)):
                for p in now:
                    if self.types[self.nodes.index(p)] == 'output':
                        p.c=1
                        print(p.cal_next_W(alpha,Y))
                    else :
                        p.c += 1
                        print(p.cal_output([x.output[i] for (x,w) in p.pre]))
                    cfc += 1
            to_update = set([])
            for p in now:
                if self.types[self.nodes.index(p)] != 'output': #can do not check this?
                    for (n,w) in p.next:
                        to_update.add(n)
                # else:
                #     check = False
            # print(to_update)


            




m = 1 #num of data
n = 3 #num of features
Y = [] #labels
X = np.zeros((m,3)) #inputs
for i in range(m):
    for j in range(n):
        X[i][j] = np.random.normal()
    Y.append(sum(X[i][:]))
# print(X[0])
# print(Y[0])
W = np.zeros(n)
for i in range(n):
    W[i]=np.random.normal()
# print(W)
alpha = 0.1 #learning rate
epoch = 100 #num of epochs


# for j in range(epoch):
#     for i in range(m):
#         # print(Y[i])
#         p1.cal_output(X[i])
#     print(p1.update_weights(alpha,Y))
X = [[1,2,3],[1,2,3]]
W = [3,2,1]
Y = [6]
def f():
    return 0
N = Network()
p1 = Perceptron(W,relu)
p2 = Perceptron(W,relu)
p3 = Perceptron(W[0:2],relu)
# print(p3.W)
N.add_node(p3,'output',2)
N.add_node(p2,'input',3)
N.add_node(p1,'input',3)
N.connect_nodes(p1,p3,1)
N.connect_nodes(p2,p3,2)
N.forward_prop(X)
# N.backward_prop(alpha,Y)