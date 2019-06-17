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
trainData = training_data[0:20]

text(trainData)


subfx=3
subfy=10
subIndex = 1
for pix,num in trainData:
    correctNum = np.argmax(np.array(num))
    plt.subplot(subfx,subfy,subIndex)
    subIndex +=1
    for i in range(28):
        for j in range(28):
            if pix[i*28 + j] != 0:
                x = j
                y = -i
                plt.scatter(x,y,c='k',marker="*")
                plt.xticks([])
                plt.yticks([])
                plt.title(correctNum)
#                plt.scatter(x,y,c=pix[i*28 + j],marker="o")
plt.show()


