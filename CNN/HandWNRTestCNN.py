import CNNCommon as cc
import numpy as np

#draw numerical picture
import matplotlib.pyplot as plt
def drawData(failTestIndex,failTestData):
    subfx=4
    subfy=10
    subIndex = 1
    for index,trainData in zip(failTestIndex,failTestData):
        failIndex = np.argmax(np.array(index))
 #       text(index)
        plt.subplot(subfx,subfy,subIndex)
        subIndex +=1
        for i in range(28):
            for j in range(28):
                if trainData[0][i*28 + j] != 0:
                    x = j
                    y = -i
                    plt.scatter(x,y,c='k',marker="*")
                    plt.xticks([])
                    plt.yticks([])
                    plt.title(str(trainData[1])+"_"+str(failIndex))
    plt.show()



#result show
def testShow(testData,testResult):
    testR = 0
    testTarget = [y for (x,y) in testData]
    failTestData =[]
    failTestIndex = []
    for (x,y) in zip(testResult,testData):
        targetIndex = np.argmax(np.array(x))
        if targetIndex == y[1]:
            testR +=1
        else:   
            failTestData.append(y)
            failTestIndex.append(x)
    print("\nTest Result: {0} / {1}".format(testR,len(testData)))
    failLen = 30
#    drawData(failTestIndex[:failLen],failTestData[:failLen])

def netTest(testData,net):
    testResult = net.test(testData)
    testShow(testData,testResult)

#show value
import traceback

def text(v):
    (fn,ln,fn,text) = traceback.extract_stack()[-2]
    begin = text.find('text(')+len('text(')
    end = text.find(')',begin)
    print("\n{0}".format(text[begin:end])," shape:",np.shape(v)," type:",type(v))
#    print(v)

#loader handwritten numerical data
import mnist_loader
training_data,validation_data,test_data = mnist_loader.load_data_wrapper()
print("training data")
print(type(training_data))
print("trainging_data:",len(training_data))
print("test_data:",len(test_data))
print(training_data[0][0].shape)
print(training_data[0][1].shape)


'''
try:
    arr=np.load('HandBiasWeigt.npz')
except:
    b=[np.random.randn(x,1) for x in initNet[1:]]
    w=[np.random.randn(y,x) for (x,y) in zip(initNet[:-1],initNet[1:])]
else:
    b = arr['bias']
    w = arr['weight']
'''


#init
oldTrainData = training_data[:]
trainData = []
for pic,target in oldTrainData:
    p = np.array(pic).reshape(28,28)
    li = (p,target)
    trainData.append(li)

trainData = training_data[:]


kernelShape = (3,3)
kernel = np.ones(kernelShape)

initNet = [784,30,10]

b=[np.random.randn(x,1) for x in initNet[1:]]
w=[np.random.randn(y,x) for (x,y) in zip(initNet[:-1],initNet[1:])]
net = cc.fullyConnected(initNet,b,w,kernel)

#train
needTrain = True
#test
needTest = True
if needTrain:
    batchSize = 10
    learnRate = 5
    for i in range(30):
        net.fit(trainData,batchSize,learnRate)
        if needTest:
            testData = test_data[:]
            netTest(testData,net)
    
#    np.savez("HandBiasWeigt",bias = net.bias,weight = net.weight)




#Show Figure
if needTrain:
    lenWeight = len(net.weights)
    i = [t for t in range(lenWeight)]
    #plt.plot(i,net.weights)
    plt.plot(i,net.biases)
    plt.show()


