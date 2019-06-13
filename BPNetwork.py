import numpy as np
import random

#li=[np.random.randn(y,1) for y in sizes[1:]]

import traceback

def text(v):
    (fn,ln,fn,text) = traceback.extract_stack()[-2]
    begin = text.find('text(')+len('text(')
    end = text.find(')',begin)
    print("{0}".format(text[begin:end]),np.shape(v),type(v))
    print(v)

def sigmod(z):
    return 1/(1+np.exp(-z))

def diffSigmod(z):
    return sigmod(z)*(1-sigmod(z))

def activeFunction(z):
    return sigmod(z)

def diffActiveFunction(z):
    return diffSigmod(z)

def netOutErr(target,result):
    return (target - result)*diffSigmod(result)


class network:
    def __init__(self,netTopology):
        self.size = len(netTopology)
        self.weight = [np.random.randn(x,y) for (x,y) in zip(netTopology[:-1],netTopology[1:])]
        self.bias = [np.random.randn(x,1) for x in netTopology[1:]]
        text(self.weight)
        text(self.bias)

    def SGD(self,trainData,cycle,numPerCycle,learnRate,testData):
        self.fit(self,trainData,cycle,numPerCycle,learnRate)
        
    
    def fit(self,trainData,cycle,numPerCycle,learnRate,):
        if cycle*numPerCycle > len(trainData):
            cycle = len(trainData)//numPerCycle
        random.shuffle(trainData)
        for currCycle in range(cycle):
            batchData = [trainData[i] for i in range(currCycle*numPerCycle,
            (currCycle+1)*numPerCycle)]
            self.training(batchData,learnRate)
                
    def test(self,testData): 
        y=[]
        for dataIn,dataOut in testData:
            netOut = self.FeedForward(dataIn)
            y.append(netOut[-1])
        return y

        
    def training(self,batchData,learnRate):
        for (netIn , netTarget) in batchData:
            netOut = self.FeedForward(netIn)
            deltaWeight,deltaBias = self.BackProp(netIn,netOut,netTarget)
            text(self.weight)
            text(deltaWeight)
            for i in range(len(self.weight)):
                self.weight[i] = self.weight[i] - learnRate*deltaWeight[i]
                self.bias[i] = self.bias[i] - learnRate*deltaBias[i]
            """
            text(self.weight)
            print(deltaWeight)
            print(deltaBias)
            """
            
    
    def FeedForward(self,netIn):
        netOut = []
        layerIn = netIn
        for layer in range(self.size-1):
     #       text(layerIn)
      #      text(self.weight[layer])
            zs = np.dot(layerIn.T,self.weight[layer])
            z = []
            for i in range(len(zs)):
                z.append(zs[i]+self.bias[layer][i])
  #          z = zs.T + self.bias[layer]
  #          text(zs)
   #         text(self.bias[layer])
    #        text(z)
            layerOut = activeFunction(np.array(z))
            netOut.append(layerOut)
            layerIn = layerOut
     #   text(netOut)
        return netOut

        
    def BackProp(self,netIn,netOut,netTarget):
        deltaWeight = [np.zeros(w.shape) for w in self.weight]
        deltaBias = [np.zeros(b.shape) for b in self.bias]
        deltaOut = netOutErr(netTarget,netOut[-1])
        deltaBias[-1] = deltaOut
        deltaWeight[-1] = deltaOut*netOut[-1]

        text(deltaOut)

        for backLayer in range(0,self.size-2):
            deltaOut = np.multiply(deltaOut,self.weight[-backLayer])
            text(deltaOut)
            text(netOut[-backLayer])
            deltaHide = np.dot(deltaOut,diffActiveFunction(netOut[-backLayer]))
            text(deltaHide)

            deltaBias[-backLayer-1] = deltaHide
            deltaWeight[-backLayer-1] = np.multiply(deltaHide,netOut[-backLayer])
        print(deltaWeight)
        print(deltaBias)
        return deltaWeight,deltaBias
                
                
    

