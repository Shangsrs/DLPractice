import numpy as np
import random
#import matplotlib.pyplot as plt

import traceback
def text(v):
    (fn,ln,fn,text) = traceback.extract_stack()[-2]
    begin = text.find('text(')+len('text(')
    end = text.find(')',begin)
    print("\n{0}".format(text[begin:end])," shape:",np.shape(v)," type:",type(v))
    print(v)


def sigmod(z):
    return 1/(1+np.exp(-z))

#f'(z) = f(z)(1-f(z))
def diffSigmod(z):
    return np.multiply(sigmod(z),(1-sigmod(z)))

def activeFunction(z):
    return sigmod(z)

def diffFunction(f):
    return np.multiply(f,(1-f))

def diffActiveFunction(z):
    return diffSigmod(z)

#f(z) = result 
#f'(z) = f(z)(1-f(z))
def netOutErr(target,result):
    return target - result


class network:
    def __init__(self,netTopology):
        self.size = len(netTopology)
        self.weight = [np.random.randn(y,x) for (x,y) in zip(netTopology[:-1],netTopology[1:])]
        self.bias = [np.random.randn(x,1) for x in netTopology[1:]]
        self.weights=[]
        self.biases=[]
  #      text(self.weight)
  #      text(self.weight)
  #      text(self.bias)
  #      text(self.bias)

    def SGD(self,trainData,cycle,numPerCycle,learnRate,testData):
        self.fit(self,trainData,cycle,numPerCycle,learnRate)
        
    
    def fit(self,trainData,cycle,numPerCycle,learnRate,):
        if cycle*numPerCycle > len(trainData):
            cycle = len(trainData)//numPerCycle
        random.shuffle(trainData)
   #     text(cycle)
    #    text(numPerCycle)
        for currCycle in range(cycle):
            batchData = [trainData[i] for i in range(currCycle*numPerCycle,
            (currCycle+1)*numPerCycle)]
     #       print(len(batchData))
            self.training(batchData,learnRate)
                
    def test(self,testData): 
        y=0
        for dataIn,dataOut in testData:
            netOut = self.FeedForward(dataIn)
#            resultIndex = np.argmax(netOut)
      #      print(dataOut)
      #      print(netOut)
      #      text(dataOut)
      #      text(netOut[-1])
            if(np.argmax(np.array(netOut[-1])) == dataOut):
                y+=1
        return y

        
    def training(self,batchData,learnRate):
        for (netIn , netTarget) in batchData:
   #         text(netIn)
   #         text(netTarget)
            netOut = self.FeedForward(netIn)
            deltaWeight,deltaBias = self.BackProp(netOut,netTarget)
            self.weight = [w-learnRate*nw
                            for w, nw in zip(self.weight, deltaWeight)]
            self.bias = [b-learnRate*nb
                           for b, nb in zip(self.bias, deltaBias)]
            self.weights.append(self.weight[1][0])
            self.biases.append(self.bias[1][0])
            
    
    def FeedForward(self,netIn):
        netOut = []
 #       ni0 =[[n0] for n0 in netIn]
 #       netOut.append(np.array(ni0))   
        netOut.append(netIn)        
        layerIn = netIn
        for w,b in zip(self.weight,self.bias):
     #       text(w)
    #        text(netOut[-1])
            z = np.dot(w,netOut[-1])+b
     #       text(z)
            netOut.append(activeFunction(z))
        return netOut

        
    def BackProp(self,netOut,netTarget):
        deltaWeight = [np.zeros(w.shape) for w in self.weight]
        deltaBias = [np.zeros(b.shape) for b in self.bias]
        deltaB = np.multiply(netOutErr(netTarget,netOut[-1]),diffFunction(netOut[-1]))
        deltaBias[-1] = deltaB
        deltaWeight[-1] = np.dot(deltaB,netOut[-2].T)
        for backLayer in range(1,self.size-1):
            deltaOut = np.dot(self.weight[-backLayer].T,deltaB)
            deltaB = np.multiply(diffFunction(netOut[-backLayer-1]),deltaOut)
            deltaBias[-backLayer-1] = deltaB
            layerOut = netOut[-backLayer-2]
            deltaWeight[-backLayer-1] = np.dot(deltaB,layerOut.T)            
        return deltaWeight,deltaBias
                
                
    

