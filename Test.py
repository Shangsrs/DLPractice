import BPNetwork

import numpy as np
import matplotlib.pyplot as plt


trainLen = 1000
trainX = np.random.randn(trainLen,1)
trainY = [i**2 for i in trainX]
training_data = list(zip(trainX,trainY))

#for i in training_data : print(i)
testLen = 100
testX = np.random.randn(testLen,1)
testY = [i**2 for i in testX]
testData = list(zip(testX,testY))

initNet = [1,5,1]
net = BPNetwork.network(initNet)
cycle = 1
numPerCycle = 10
learnRate = 3
net.fit(training_data,cycle,numPerCycle,learnRate)

len = len(net.weights)
i = [t for t in range(len)]


plt.plot(i,net.weights)
plt.plot(i,net.biases)
plt.figure();

reY = net.test(testData)
plt.scatter(trainX,trainY,marker='*')
plt.scatter(testX,testY,marker='<')
plt.scatter(testX,reY,marker='o')
plt.show()

<<<<<<< HEAD
'''

trX,trY = vx,vy for vx,vy in zip(trainX,trainY)
=======

'''
trX,trY = [vx,vy for (vx,vy) in zip(trainX,trainY)]
>>>>>>> d3326913ae2530a89c0e2f7aa972e4d6e29365fc
plt.scatter(trX,trY,marker='o')

teX,teY = [vx,vy for (vx,vy) in zip(testX,testY)]
plt.scatter(teX,teY,marker='o')

plt.scatter(teX,reY,marker='o')

plt.scatter(trainX,trainY,marker='<')
plt.show()
'''
