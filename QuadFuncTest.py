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

#text(trainX)
#text(trainY)
initNet = [1,5,1]
net = bp.network(initNet)
cycle = 1
numPerCycle = 2
learnRate = 10
net.fit(training_data,cycle,numPerCycle,learnRate)

'''
lenWeight = len(net.weights)
i = [t for t in range(lenWeight)]
plt.plot(i,net.weights)
plt.plot(i,net.biases)
plt.figure();
'''

testLen = 100
testX = np.random.randn(testLen,1)
testY = [i**2 for i in testX]
testData = list(zip(testX,testY))
reY = net.test(testData)

#resultY = [z for x,y,z in reY]

#plt.scatter(trainX,trainY,marker='<')
plt.scatter(testX,testY,marker='*')
plt.scatter(testX,reY,marker='o')
plt.show()




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



