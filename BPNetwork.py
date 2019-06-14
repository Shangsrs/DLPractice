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
   #     text(self.weight)
   #     text(self.bias)

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
            dw0 =[[w0] for w0 in deltaWeight[0]]
            deltaWeight[0]=np.array(dw0)
            '''
            text(netOut)
            text(self.weight)
            text(deltaWeight)
            text(self.weight[0])
            text(deltaWeight[0])
            '''
            self.weight = [w-learnRate*nw
                            for w, nw in zip(self.weight, deltaWeight)]
            self.bias = [b-learnRate*nb
                           for b, nb in zip(self.bias, deltaBias)]
            self.weights.append(self.weight[0][0])
            self.biases.append(self.bias[0][0])
            """
            print("after bp\n\n\n")
            text(self.weight)
            text(deltaWeight)
            text(self.weight)
            print(deltaWeight)
            print(deltaBias)
            """
            
    
    def FeedForward(self,netIn):
        netOut = []
        netOut.append(np.array(netIn))
        layerIn = netIn
        for layer in range(self.size-1):
   #         text(layerIn)
   #         text(self.weight[layer])
    #        text(self.bias[layer])
            zs = np.dot(self.weight[layer],layerIn)
            z = np.array([zi-bi for zi,bi in zip(zs,self.bias[layer])])
#            text(z)
            layerOut = activeFunction(z)
            netOut.append(layerOut)
            layerIn = layerOut
   #     text(netOut)
        return netOut

        
    def BackProp(self,netIn,netOut,netTarget):
        deltaWeight = [np.zeros(w.shape) for w in self.weight]
        deltaBias = [np.zeros(b.shape) for b in self.bias]
  #      text(netOut)
  #      text(netTarget)
        deltaOut = np.multiply(netOutErr(netTarget,netOut[-1]),diffFunction(netOut[-1]))
     #   text(deltaOut)
        deltaBias[-1] = deltaOut
        deltaWeight[-1] = np.dot(deltaOut,netOut[-2].T)
     #   text(deltaBias)
      #  text(deltaWeight)

        for backLayer in range(1,self.size-1):
       #     text(self.weight)
       #     text(deltaOut)
            deltaOut = np.dot(self.weight[-backLayer].T,deltaOut)
       #     text(deltaOut)
      #      text(netOut[-backLayer])
            deltaHide = np.multiply(diffFunction(netOut[-backLayer-1]),deltaOut)

            deltaBias[-backLayer-1] = deltaHide
            layerOut = netOut[-backLayer-2]
   #         print("\n\n\n")
    #        text(deltaHide)
    #        text(layerOut)
    #        print("\n\n\n")
            deltaWeight[-backLayer-1] = np.dot(deltaHide,layerOut.T)
            deltaOut = deltaHide
  #      text(deltaWeight)
  #      text(deltaBias)
        return deltaWeight,deltaBias
                
                
    

