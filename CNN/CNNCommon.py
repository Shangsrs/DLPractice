'''
Common function for CNN
'''
import numpy as np
import random
import matplotlib.pyplot as plt


import traceback
def text(v):
    (fn,ln,fn,text) = traceback.extract_stack()[-2]
    begin = text.find('text(')+len('text(')
    end = text.find(')',begin)
    print("\n{0}".format(text[begin:end])," shape:",np.shape(v)," type:",type(v))
#    print(v)
class transFun:
    def __init__(self,name = "sigmod"):
        self.funName = name
    def active(self,z):
        if self.funName == "reLU":
            return self.reLU(z)
        elif self.funName == "softPlus":
            return self.softPlus(z)
        elif self.funName == "tanh":
            return self.tanh(z)
        elif self.funName == "softMax":
            return self.softMax(z)
        else:
            return self.sigmod(z)
    def diff(self,z):
        if self.funName == "reLU":
            return self.diffReLU(z)
        elif self.funName == "softPlus":
            return self.diffSoftPlus(z)
        elif self.funName == "tanh":
            return self.diffTanh(z)
        elif self.funName == "softMax":
            return self.diffSoftMax(z)
        else:
            return self.diffSigmod(z)
    def sigmod(self,z):
        return 1/(1+np.exp(-z))
    #f'(z) = f(z)(1-f(z))
    def diffSigmod(self,z):
        return self.sigmod(z)*(1-self.sigmod(z))
    #ReLU
    #Rectified Linear Unit
    def reLU(self,z):
        red = [e if e>=0 else 0 for e in z]
        return np.array(red).reshape(z.shape[0],1)
    def diffReLU(self,z):
        red = [1 if e>=0 else 0 for e in z]
        return np.array(red).reshape(z.shape[0],1)
    #softPlus
    def softPlus(self,z):
        return np.log(np.exp(z)+1)
    def diffSoftPlus(self,z):
        return np.exp(z)/(np.exp(z)+1)
    #softMax
    def softMax(self,z):
        return np.exp(z)/sum(np.exp(z))
    def diffSoftMax(self,z):
        return z
    #softMax
    def max(self,z):
        m = np.max(z)
        return m*np.ones(z.shape,1)
    def diffMax(self,z):
        return z
    #thah
    def tanh(self,z):
        return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
    def diffTanh(self,z):
        expZ = np.exp(z)
        expPz = np.exp(-z)
        return ((expZ+expPz)**2-(expZ-expPz)**2)/(expZ+expPz)**2
class sample:
    def __init__(self,ks,step,kernel,bias):
        self.kernel = kernel
        self.step = step
        self.ks = ks
        self.bias = bias
    def downMax(self,matrix):
        self.matrix = matrix
        self.ms = self.matrix.shape
        re = np.zeros(((self.ms[0]-self.ks[0])//self.step+1,(self.ms[1]-self.ks[1])//self.step+1))
        for x in range(0,self.ms[0]-self.ks[0]+1,self.step):
            for y in range(0,self.ms[1]-self.ks[1]+1,self.step):
                mat = self.matrix[x:x+self.ks[0],y:y+self.ks[1]]
                mat = mat * self.kernel
                e = np.max(mat) + self.bias
                re[(x)//self.step][(y)//self.step]=e
        return re
    def downAvg(self,matrix):
        self.matrix = matrix
        self.ms = self.matrix.shape
        re = np.zeros(((self.ms[0]-self.ks[0])//self.step+1,(self.ms[1]-self.ks[1])//self.step+1))
        for x in range(0,self.ms[0]-self.ks[0]+1,self.step):
            for y in range(0,self.ms[1]-self.ks[1]+1,self.step):
                mat = self.matrix[x:x+self.ks[0],y:y+self.ks[1]]
                mat = mat * self.kernel
                e = np.average(mat) + self.bias
                re[(x)//self.step][(y)//self.step]=e
        return re
    def upMax(self,matrix):
        self.matrix = matrix
        self.ms = self.matrix.shape
        re = np.zeros(self.ms)
        for x in range(0,self.ms[0]-self.ks[0]+1,self.ks[0]):
            for y in range(0,self.ms[1]-self.ks[1]+1,self.ks[1]):
                mat = self.matrix[x:x+self.ks[0],y:y+self.ks[1]]
                mat = mat * self.kernel
                e = np.max(mat)
                index = np.argmax(mat)
                sub = np.zeros(mat.shape)
                rol = index // self.ks[1]
                col = index % self.ks[1]
                sub[rol][col] = e + self.bias
                re[x:x+self.ks[0],y:y+self.ks[1]] = sub
        return re
    def upAvg(self,matrix):
        self.matrix = matrix
        self.ms = self.matrix.shape
        re = np.zeros(self.ms)
        for x in range(0,self.ms[0]-self.ks[0]+1,self.ks[0]):
            for y in range(0,self.ms[1]-self.ks[1]+1,self.ks[1]):
                mat = self.matrix[x:x+self.ks[0],y:y+self.ks[1]]
                mat = mat * self.kernel
                e = np.average(mat)
                sub = e*np.ones(mat.shape) + self.bias
                re[x:x+self.ks[0],y:y+self.ks[1]] = sub
        return re
class maxPool:
    def __init__(self,ks,step,kernel,bias):
        self.sam = sample(ks,step,kernel,bias)
    def forward(self,matrix):
        return self.sam.downMax(matrix)
    def backward(self,netTarget):pass
class avgPool:
    def __init__(self,ks,step,kernel,bias):
        self.sam = sample(ks,step,kernel,bias)
    def forward(self,matrix):
        return self.sam.downAvg(matrix)
    def backward(self,netTarget):pass
#convolution
class conv:
    def __init__(self,ks,step,kernel,bias):
        self.step = step
        self.kernel= kernel
        self.ks = ks
        self.bias = bias
    def forward(self,matrix):
        self.matrix = matrix
        self.ms = self.matrix.shape
        re = np.zeros(((self.ms[0]-self.ks[0])//self.step+1,(self.ms[1]-self.ks[1])//self.step+1))
        for x in range(0,self.ms[0]-self.ks[0]+1,self.step):
            for y in range(0,self.ms[1]-self.ks[1]+1,self.step):
                mat = self.matrix[x:x+self.ks[0],y:y+self.ks[1]]
                #mat = mat * self.reverse(kernel)
                mat = mat * self.kernel
                e = np.sum(mat) + self.bias
                re[(x)//self.step][(y)//self.step] = e
        return re
    def backward(self,netTarget):
        error = netTarget - self.matrix
    def reverse(self,kernel):
        ks = kernel.shape
        len = ks[0]*ks[1]
        skernel = kernel.reshape(len,)[-1::-1]
        return skernel.reshape(ks[1],ks[0])
class padding:
    def __init__(self,size,ele = 0):
        self.size = size
        self.ele = ele
    def forward(self,matrix):
        self.matrix = matrix
        self.ms = self.matrix.shape
        re = self.ele * np.ones((self.ms[0]+self.size[0]*2,self.ms[1]+self.size[1]*2))
        re[self.size[0]:self.size[0]+self.ms[0],self.size[1]:self.size[1]+self.ms[1]] = self.matrix
        return re
    def backward(self):pass
def netOutErr(target,result):
    return result - target 
class fullyConnected:
    def __init__(self,netSt,learnRate,b,w):
        self.size = len(netSt)
        self.learnRate = learnRate
        self.bias = b
        self.weight = w
        self.tf = transFun("")
        self.weights=[]
        self.biases=[]
        self.netOut = []
        self.netIn = []
        self.inputShape = (0,0)
    def forward(self,matrix):
#        text(matrix)
        self.inputShape = matrix.shape
        x = matrix.ravel()
        x = x.reshape(len(x),1)
        netOut = []
        netOut.append(x)
        netIn = [x]
        for w,b in zip(self.weight,self.bias):
            z = np.dot(w,netOut[-1])+b
            netIn.append(z)
            netOut.append(self.tf.active(z))
        self.netIn = netIn
        self.netOut = netOut
        return netOut[-1]
    def backward(self,netTarget):
        deltaWeight = [np.zeros(w.shape) for w in self.weight]
        deltaBias = [np.zeros(b.shape) for b in self.bias]
        deltaOut = netOutErr(netTarget,self.netOut[-1])*self.tf.diff(self.netIn[-1])
        deltaBias[-1] = deltaOut
        deltaWeight[-1] = np.dot(deltaOut,self.netOut[-2].T)
        for backLayer in range(1,self.size-1):
            deltaOut = np.dot(self.weight[-backLayer].T,deltaOut)
            deltaHide = self.tf.diff(self.netIn[-backLayer-1])*deltaOut
            deltaBias[-backLayer-1] = deltaHide
            layerOut = self.netOut[-backLayer-2]
            deltaWeight[-backLayer-1] = np.dot(deltaHide,layerOut.T)
            deltaOut = deltaHide
        self.weight = [w-d for w,d in zip(self.weight,deltaWeight)]
        self.bias = [b-d for b,d in zip(self.bias , deltaBias)]
        self.weights.append(self.weight[-1][-1])
        self.biases.append(self.bias[-1][-1])
        deltaOut = np.dot(self.weight[0].T,deltaOut)
        text(deltaOut)
        return deltaOut.reshape(self.inputShape)
class CNN:
    def __init__(self,netStruct):
        self.netStruct = netStruct
    def fit(self,trainData,batchSize):
        random.shuffle(trainData)
        for i in range(0,len(trainData),batchSize):
            batchData = trainData[i:i+batchSize]
            self.trainBatch(batchData)
    def trainBatch(self,batchData):
        for (x , y) in batchData:
            y = y.reshape(y.shape[0],1)
            netIn = x
            netTarget = y
            for net in self.netStruct:
                mat = net.forward(netIn)
                netIn = mat
            for net in self.netStruct[-1:0:-1]:
                mat = net.backward(netTarget)
                netTarget = mat
    def test(self,testData):
        re = []
        for x in testData:
            netIn = x
            for net in self.netStruct:
                mat = net.forward(netIn)
                netIn = mat
            re.append(netIn)
        return re

if __name__ == "__main__":
    mat = np.random.randint(0,9,(6,6))
#    mat = np.random.uniform(0,9,(6,6))
    text(mat)
    ks = (2,2)
#    kernel = np.random.randint(0,8,(2,2))
    step = 1
    sam = sample(ks,step)
    downMax = sam.downMax(mat)
    text(downMax)
    downAvg = sam.downAvg(mat)
    text(downAvg)
    upAvg = sam.upAvg(mat)
    text(upAvg)
    upMax = sam.upMax(mat)
    text(upMax)
    pool = maxPool(ks,step)
    poolMax = pool.forward(mat)
    text(poolMax)
    pool = avgPool(ks,step)
    poolAvg = pool.forward(mat)
    text(poolAvg)
    pad = padding((2,2))
    padMat = pad.forward(mat)
    text(padMat)
    con = conv(ks,step)
    conMat = con.forward(mat)
    text(conMat)
