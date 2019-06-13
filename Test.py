import BPNetwork

import numpy as np
import matplotlib.pyplot as plt


trainLen = 100
trainX = np.random.randn(trainLen,1)
trainY = [i**2 for i in trainX]
training_data = list(zip(trainX,trainY))

#for i in training_data : print(i)
testLen = 100
testX = np.random.randn(testLen,1)
testY = [i**2 for i in testX]
testData = list(zip(testX,testY))

#plt.scatter(trainX,trainY,marker='*')
#plt.show()


initNet = [1,5,1]
net = BPNetwork.network(initNet)
cycle = 1
numPerCycle = 2
learnRate = 3
net.fit(training_data,cycle,numPerCycle,learnRate)

'''
y = net.test(testData)
#print("{x} - {y}".format(testX,y))
#print(testX)
#print(y)
#print(len(testX))
#print(len(y))
plt.scatter(testX,list(y),marker='*')
plt.show()
'''
