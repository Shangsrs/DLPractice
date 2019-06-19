import BPNetwork as bp

import numpy as np
import matplotlib.pyplot as plt

import traceback
def text(v):
    (fn,ln,fn,t) = traceback.extract_stack()[-2]
    begin = t.find('text(')+len('text(')
    end = t.find(')',begin)
    print("\n{0}".format(t[begin:end])," shape:",np.shape(v)," type:",type(v))
    print(v)

# y = x**2 test 
trainLen = 10000
trainX = np.random.randn(trainLen,1)
trainY = np.array([i**2 for i in trainX])
training_data = list(zip(trainX,trainY))


#init
initNet = [1,5,1]
try:
    arr=np.load('QuadBiasWeigt.npz')
except:
    b=[np.random.randn(x,1) for x in initNet[1:]]
    w=[np.random.randn(y,x) for (x,y) in zip(initNet[:-1],initNet[1:])]
else:
    b = arr['bias']
    w = arr['weight']
net = bp.network(initNet,b,w)

#train
needTrain = True
if needTrain:
    cycle = 1
    numPerCycle = 2
    learnRate = 10
    net.fit(training_data,cycle,numPerCycle,learnRate)
    np.savez("QuadBiasWeigt",bias = net.bias,weight = net.weight)


testLen = 100
testX = np.random.randn(testLen,1)
testY = [i**2 for i in testX]
testData = list(zip(testX,testY))
reY = net.test(testData)

plt.scatter(testX,testY,marker='*')
plt.scatter(testX,reY,marker='o')
plt.show()


#Show Figure
if needTrain:
    lenWeight = len(net.weights)
    i = [t for t in range(lenWeight)]
    #plt.plot(i,net.weights)
    plt.plot(i,net.biases)
    plt.show()

