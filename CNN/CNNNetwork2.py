import numpy as np
import random
import matplotlib.pyplot as plt
import CNNCommon

#li=[np.random.randn(y,1) for y in sizes[1:]]

import traceback

def text(v):
    (fn,ln,fn,text) = traceback.extract_stack()[-2]
    begin = text.find('text(')+len('text(')
    end = text.find(')',begin)
    print("\n{0}".format(text[begin:end])," shape:",np.shape(v)," type:",type(v))
#    print(v)

#f(z) = result 
#f'(z) = f(z)(1-f(z))
def netOutErr(target,result):
    return result - target 

class network:
    def __init__(self,netTopology,b,w,kernel):
        self.size = len(netTopology)
        self.bias = b
        self.weight = w
        self.kernel = kernel
 #       self.tf = CNNCommon.transFun("reLU")
        self.tf = CNNCommon.transFun()
        self.weights=[]
        self.biases=[]
    
    def fit(self,trainData,cycle,numPerCycle,learnRate):
        if cycle*numPerCycle > len(trainData):
            cycle = len(trainData)//numPerCycle
        random.shuffle(trainData)
        num = 0
        for i in range(0,len(trainData),numPerCycle):
            num +=1
            print(num)
            batchData = trainData[i:i+numPerCycle]
            self.training(batchData,learnRate)
                
    def test(self,testData): 
        y=[]
        for dataIn,dataOut in testData:
            dataIn = dataIn.reshape(dataIn.shape[0],1)
            netOut = self.FeedForward(dataIn)
            y.append(netOut[-1])
        return y

        
    def training(self,batchData,learnRate):
        deltaWeight = [np.zeros(w.shape) for w in self.weight]
        deltaBias = [np.zeros(b.shape) for b in self.bias]
  #      text(batchData)
        for (netIn , netTarget) in batchData:
            netIn = netIn.reshape(netIn.shape[0],1)
            netTarget = netTarget.reshape(netTarget.shape[0],1)
            netOut = self.FeedForward(netIn)
            deltaW,deltaB = self.BackProp(netOut,netTarget)
            deltaWeight = [dw+pw for dw,pw in zip(deltaWeight,deltaW)]
            deltaBias = [db+pb for db,pb in zip(deltaBias,deltaB)]
        self.weight = [w-learnRate*nw/len(batchData)
                        for w, nw in zip(self.weight, deltaWeight)]
        self.bias = [b-learnRate*nb/len(batchData)
                       for b, nb in zip(self.bias, deltaBias)]
        self.weights.append(self.weight[-1][-1])
        self.biases.append(self.bias[-1][-1])
            
    
    def FeedForward(self,netIn):
        netOut = []
        netOut.append(netIn)
        for w,b in zip(self.weight,self.bias):
            z = np.dot(w,netOut[-1])+b
            netOut.append(self.tf.active(z))
  #      text(netOut)
        return netOut

        
    def BackProp(self,netOut,netTarget):
        deltaWeight = [np.zeros(w.shape) for w in self.weight]
        deltaBias = [np.zeros(b.shape) for b in self.bias]
        deltaOut = np.multiply(netOutErr(netTarget,netOut[-1]),self.tf.diffOut(netOut[-1]))
        deltaBias[-1] = deltaOut
        deltaWeight[-1] = np.dot(deltaOut,netOut[-2].T)
        for backLayer in range(1,self.size-1):
            deltaOut = np.dot(self.weight[-backLayer].T,deltaOut)
            deltaHide = np.multiply(self.tf.diffOut(netOut[-backLayer-1]),deltaOut)
            deltaBias[-backLayer-1] = deltaHide
            layerOut = netOut[-backLayer-2]
            deltaWeight[-backLayer-1] = np.dot(deltaHide,layerOut.T)
            deltaOut = deltaHide
        return deltaWeight,deltaBias
                
