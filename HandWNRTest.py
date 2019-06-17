import BPNetwork as bp
import numpy as np
import matplotlib.pyplot as plt

import traceback
def text(v):
    (fn,ln,fn,text) = traceback.extract_stack()[-2]
    begin = text.find('text(')+len('text(')
    end = text.find(')',begin)
    print("\n{0}".format(text[begin:end])," shape:",np.shape(v)," type:",type(v))
#    print(v)

import mnist_loader
training_data,validation_data,test_data = mnist_loader.load_data_wrapper()
print("training data")
print(type(training_data))
print("trainging_data:",len(training_data))
print("test_data:",len(test_data))
print(training_data[0][0].shape)
print(training_data[0][1].shape)

#train
trainData = training_data[:]
initNet = [784,30,10]
net = bp.network(initNet)
cycle = 1000
numPerCycle = 50
learnRate = 5
text(trainData)
net.fit(trainData,cycle,numPerCycle,learnRate)

#test
testData = test_data[0:10000]
text(testData)
testResult = net.test(testData)
text(testResult)

#result 
testR = 0
testTarget = [y for (x,y) in testData]
for (x,y) in zip(testResult,testTarget):
    targetIndex = np.argmax(np.array(x))
    if targetIndex == y:
        testR +=1
print("\nTest Result: {0} / {1}".format(testR,len(testData)))

'''
#Show Figure
lenWeight = len(net.weights)
i = [t for t in range(lenWeight)]
text(net.weights)
text(net.biases)
#plt.plot(i,net.weights)
plt.plot(i,net.biases)
plt.show()
'''

