import numpy as np
import math


# constant c for calculate derivatives
c = math.pow(10,-15)


def d1_cal(x,f): #calculates derivative f bY x
    return float(f(x) - f(x-c))/float(c)


def d2_cal(x1,x2,f): #calculates derivative f bY x1
    return float(f(x1,x2) - f(x1-c,x2))/float(c)


# loss functions
def mse_loss(y_pred,y):
    return ((y_pred-y)**2)/2


# activation functions
def relu(input):
    return max(0,input)


def relu_n(input):
    if input>=0:
        return input
    else:
        return 0.01*input


def step(input):
    if input>=0:
        return 1
    else:
        return -1


class Perceptron():
    def __init__(self,W,activation):
        self.output = []
        self.d_output = []
        self.W = W  # Note: W is transposed
        self.W_next = W 
        self.m = []
        self.c = []
        self.next = []
        self.pre = []
        self.activation = activation


    def cal_output(self,X):
        # print('x:' +str(X))
        # print('w:' +str(np.transpose(self.W)))
        self.m.append(np.matmul([X],np.transpose([self.W]))[0][0])
        self.output.append(self.activation(self.m[-1]))
        def output_i(w,x):
            return w*x
        # print(self.output)
        self.d_output.append(X)
        # print(self.d_output)
        return self.output


    def cal_next_W(self,alpha,Y,i):
        # print('W: ' + str(self.W))
        # print('c: ' + str(self.c))
        # print('d_out: ' + str(self.d_output[i]))
        self.W_next = np.array([self.W[j] - alpha*self.c[i]*self.d_output[i][j] for j in range(len(self.W))])
        return self.W_next


    def update_weights(self):
        self.W = self.W_next
        return self.W


class Network:
    def __init__(self,l):
        self.nodes = []
        self.types = []
        self.loss = 1
        self.loss_function = l


    def add_node(self,p,t,n):
        self.nodes.append(p)
        self.types.append(t)

    
    def connect_nodes(self,p1,p2,n_of_w_in):
        p1.next.append((p2,n_of_w_in))
        # print(p1.next)
        p2.pre.append((p1,n_of_w_in))
        # print(p2.pre)


    def cal_loss(self,Y):
        for p in self.nodes:
            if self.types[self.nodes.index(p)] == 'output':
                self.loss = sum([self.loss_function(p.output[i],Y[i]) for i in range(len(Y))])/(len(Y))
                print('====================')
                return self.loss


    def forward_prop(self,X):
        to_update = set([])
        for p in self.nodes:
            if self.types[self.nodes.index(p)] == 'input':
                to_update.add(p)
        while (len(to_update)!=0):
            now = to_update
            for i in range(len(X)):
                for p in now:
                    # print(p.W)
                    if self.types[self.nodes.index(p)] == 'input':
                        p.cal_output(X[i])
                    else :
                        p.cal_output([x.output[i] for (x,w) in p.pre])
            to_update = set([])
            for p in now:
                if self.types[self.nodes.index(p)] != 'output': #can do not check this?
                    for (n,w) in p.next:
                        to_update.add(n)


    def backward_prop(self,alpha,Y):
        to_update = set([])
        print(self.cal_loss(Y))
        for p in self.nodes:
            if self.types[self.nodes.index(p)] == 'output':
                for pre in p.pre:
                    to_update.add(pre)
                self.d_loss = [d2_cal(p.output[i],Y[i],self.loss_function) for i in range(len(Y))]
                for i in range(len(Y)):
                    p.c.append(self.d_loss[i])
                    p.cal_next_W(alpha,Y,i)
                    # print('W_N: '+ str(p.cal_next_W(alpha,Y,i)))
                break
        while (len(to_update)!=0):
            now = to_update
            for (p,n_of_w) in now:
                for i in range(len(Y)):
                    (pn,j) = [t for t in p.next if t[1] == n_of_w][0]
                    if len(p.c) != len(Y):
                        # print('---')
                        # print('c: ' + str(pn.c[i]))
                        # print('d1: '+ str(d1_cal(pn.m[i],pn.activation)))
                        # print('w[j]: '+str(pn.W[j]))
                        # print('---')
                        p.c.append(pn.c[i]*d1_cal(pn.m[i],pn.activation)*pn.W[j])
                    else:
                        p.c[i] += pn.c[i]*d1_cal(pn.m[i],pn.activation)*pn.W[j]
                    p.cal_next_W(alpha,Y,i)
                    # print('W_N: '+ str(p.cal_next_W(alpha,Y,i)))
            to_update = set([])
            for (p,w) in now:
                if self.types[self.nodes.index(p)] != 'input': #can do not check this?
                    for pre in p.pre:
                        to_update.add(pre)
            # print(to_update)
        for p in self.nodes:
            # print(str(self.nodes.index(p))+ ' '+str(p.W) + ' ' + str(p.c))
            p.output = []
            p.d_output = []
            p.c = []
            p.update_weights()
        

# testing...    
m = 100 #num of data
n = 3 #num of input features
alpha = 0.1 #learning rate
epoch = 1000 #num of epochs

# data generation
Y = [] #labels
X = np.zeros((m,3)) #inputs
for i in range(m):
    for j in range(n):
        X[i][j] = np.random.normal()
    Y.append(sum([t**2 for t in X[i]])) 

# initializing weights
W = np.zeros(n) #initial weights
for i in range(n):
    W[i]=np.random.normal()

# print(W)
# print(X[0])
# print(Y[0])
# input()

# building network
N = Network(mse_loss)
p1 = Perceptron(W,relu_n)
p2 = Perceptron(W,relu_n)
p3 = Perceptron(W[0:2],relu_n)
p4 = Perceptron(W[0:2],relu_n)
p5 = Perceptron(W[0:2],relu_n)

N.add_node(p5,'output',2)
N.add_node(p4,'hidden',2)
N.add_node(p3,'hidden',2)
N.add_node(p2,'input',3)
N.add_node(p1,'input',3)

N.connect_nodes(p1,p3,0)
N.connect_nodes(p2,p3,1)
N.connect_nodes(p1,p4,0)
N.connect_nodes(p2,p4,1)
N.connect_nodes(p3,p5,0)
N.connect_nodes(p4,p5,1)

# learning
for i in range(epoch):
    print('--forward start--')
    N.forward_prop(X)
    print('--forward end--')
    print('--backward start--')
    N.backward_prop(alpha,Y)
    print('--backward end--')