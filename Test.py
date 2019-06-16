import BPNetwork3 as bp

import numpy as np
import matplotlib.pyplot as plt

import traceback
def text(v):
    (fn,ln,fn,text) = traceback.extract_stack()[-2]
    begin = text.find('text(')+len('text(')
    end = text.find(')',begin)
    print("\n{0}".format(text[begin:end])," shape:",np.shape(v)," type:",type(v))
#    print(v)

'''
# y = x**2 test 

trainLen = 600
trainX = np.random.randn(trainLen,1)
trainY = [i**2 for i in trainX]
training_data = list(zip(trainX,trainY))

testLen = 600
testX = np.random.randn(testLen,1)
testY = [i**2 for i in testX]
testData = list(zip(testX,testY))

initNet = [1,5,1]
net = bp.network(initNet)
cycle = 1
numPerCycle = 600
learnRate = 6
net.fit(training_data,cycle,numPerCycle,learnRate)

len = len(net.weights)
i = [t for t in range(len)]

plt.plot(i,net.weights)
plt.plot(i,net.biases)
plt.figure();

reY = net.test(testData)
print(reY)
plt.scatter(trainX,trainY,marker='*')
#plt.scatter(testX,testY,marker='<')
plt.scatter(testX,reY,marker='o')
plt.show()
'''

'''
trX,trY = vx,vy for vx,vy in zip(trainX,trainY)

trX,trY = [vx,vy for (vx,vy) in zip(trainX,trainY)]
>>>>>>> d3326913ae2530a89c0e2f7aa972e4d6e29365fc
plt.scatter(trX,trY,marker='o')

teX,teY = [vx,vy for (vx,vy) in zip(testX,testY)]
plt.scatter(teX,teY,marker='o')

plt.scatter(teX,reY,marker='o')

plt.scatter(trainX,trainY,marker='<')
plt.show()
'''

'''
#np.exp(z) test
ex = [i/10 for i in range(-100,100,1)]

ey = 1/(1+np.exp(-1*np.array(ex)))

plt.plot(ex,ey)
plt.show()
'''

import mnist_loader
training_data,validation_data,test_data = mnist_loader.load_data_wrapper()
print("training data")
print(type(training_data))
print("trainging_data:",len(training_data))
print("test_data:",len(test_data))
print(training_data[0][0].shape)
print(training_data[0][1].shape)

trainData = training_data[:]
testData = test_data[0:100]

initNet = [784,30,10]
net = bp.network(initNet)
cycle = 10
numPerCycle = 20
learnRate = 6

net.fit(trainData,cycle,numPerCycle,learnRate)

'''
weightLen = len(net.weights)
i = [t for t in range(weightLen)]
plt.plot(i,net.weights)
plt.plot(i,net.biases)
#plt.show()
'''
testTarget = [y for (x,y) in testData]
testResult = net.test(testData)

print("{0} / {1}".format(testResult,len(testData)))


