import BPNetwork as bp
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

#train
trainData = training_data[:]
initNet = [784,30,10]
net = bp.network(initNet)
cycle = 10000
numPerCycle = 2
learnRate = 5
text(trainData)
net.fit(trainData,cycle,numPerCycle,learnRate)

#test
testData = test_data[:]
text(testData)
testResult = net.test(testData)
text(testResult)

#result 
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
drawData(failTestIndex[:failLen],failTestData[:failLen])

#Show Figure
lenWeight = len(net.weights)
i = [t for t in range(lenWeight)]
text(net.weights)
text(net.biases)
#plt.plot(i,net.weights)
plt.plot(i,net.biases)
plt.show()


