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
print("validation_data:",len(validation_data))
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
def dataTrans(training_data):
    oldTrainData = training_data[:]
    input = []
    target = []
    for x,y in oldTrainData:
        p = np.array(x).reshape(28,28)
        input.append(p)
        target.append(y)
    return input,target


input,target = dataTrans(training_data[:500])
trainData = zip(input,target)
trainData = list(trainData)

kernelShape = (3,3)
kernel = np.ones(kernelShape)
initNet = []

p0_ks =(2,2)
c1_ks = (5,5)
s2_ks = (2,2)
c3_ks = (5,5)
s4_ks = (2,2)
f5_netSt = (25,10,10)
f5_learnRate = 5

try:
    arr=np.load('CNNKernel.npz')
except:
    pre = 0
    f5_b = [np.random.randn(y,1) for y in f5_netSt[1:]]
    f5_w = [np.random.randn(y,x) for x,y in zip(f5_netSt[:-1],f5_netSt[1:])]
else:
    c1_k = arr['c1k']
    s2_k = arr['s2k']
    c3_k = arr['c3k']
    s4_k = arr['s4k']
    c1_b = arr['c1b']
    s2_b = arr['s2b']
    c3_b = arr['c3b']
    s4_b = arr['s4b']
    pre = arr['precision']
    f5_b = arr['f5b']
    f5_w = [arr['f5w0'],arr['f5w1']]

p0 = cc.padding(p0_ks,0)
c1 = cc.conv(c1_ks,1,c1_k,c1_b)
s2 = cc.maxPool(s2_ks,2,s2_k,s2_b)
c3 = cc.conv(c3_ks,1,c3_k,c3_b)
s4 = cc.maxPool(s4_ks,2,s4_k,s4_b)

f5 = cc.fullyConnected(f5_netSt,f5_learnRate,f5_b,f5_w)
netStruct = [p0,c1,s2,c3,s4,f5]

net = cc.CNN(netStruct)
batchSize = 10

input,target = dataTrans(test_data[1000:2000])
testData = input

print("\nTest\n")
re = net.test(testData)

testR = 0
testTarget = target
failTestData =[]
failTestIndex = []
for (x,y) in zip(re,testTarget):
    targetIndex = np.argmax(np.array(x))
    if targetIndex == y:
        testR +=1
    else:   
        failTestData.append(y)
        failTestIndex.append(x)
print("\nTest Result: {0} / {1}".format(testR,len(testData)))
failLen = 30

newPre = testR/len(testData)

if newPre > pre:
    pre = newPre
    np.savez("CNNKernel",c1k=c1_k,s2k=s2_k,c3k=c3_k,s4k=s4_k,
    c1b=c1_b,s2b=s2_b,c3b=c3_b,s4b=s4_b,precision=pre,
    f5b=net.netStruct[-1].bias, f5w0=net.netStruct[-1].weight[0],
    f5w1=net.netStruct[-1].weight[1])


#Show Figure
needTrain = False
if needTrain:
    lenWeight = len(net.netStruct[-1].weights)
    i = [t for t in range(lenWeight)]
#    plt.plot(i,net.netStruct[-1].weights)
#    plt.show()
    plt.plot(i,net.netStruct[-1].biases)
    plt.show()

