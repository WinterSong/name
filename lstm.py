# coding: utf-8

# this script is another way to organize the method to feed lstm

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, TimeDistributed
from keras.optimizers import RMSprop
import numpy as np
import random
import h5py
inputFile = open('/disk/dzy/mpii/labels/detectionGroundtruth-1-0.csv','r')
# read the feature
feature = h5py.File('/disk/dzy/mpii/features/hog/IntegralHist/s08-d02-hard-0.mat','r')
X = [list(i) for i in list(feature.get('integralFrameHist'))]
Y = [0] * len(X)
# read the label for each picture
for line in inputFile.readlines():
    line = line.strip()
    arr = line.split(',')
    label = int(arr[4])-1
    if label == 0:
        continue
    videoName = arr[1]
    if videoName != 's08-d02-cam-002':
        break
    startFrame = int(arr[2])
    endFrame = int(arr[3])
    Y[startFrame:endFrame+1] = [label]*(endFrame-startFrame+1)

maxlen = 100
step = 3
x_train = []
y_train = []
x_test = []
y_test = []
testSize = 50

for i in range(0,len(X)-maxlen-testSize*step,step):
    x_train.append(X[i:i+maxlen])
    tmp = [0]*65
    flag = 0
    for j in range(i,i+maxlen):
        if (Y[j] == flag or Y[j] == 0):
            continue
        else:
            tmp[Y[j]] = 1
            flag = Y[j]
    y_train.append(tmp)

for i in range(len(X)-maxlen-testSize*step,len(X)-maxlen,step):
    x_test.append(X[i:i+maxlen])
    tmp = [0]*65
    flag = 0
    for j in range(i,i+maxlen):
        if (Y[j] == flag or Y[j] == 0):
            continue
        else:
            tmp[Y[j]] = 1
            flag = Y[j]
    y_test.append(tmp)

model = Sequential()
model.add(LSTM(32, input_shape=(maxlen,len(X[0])), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(65))
model.add(Activation('softmax'))

optimizer = RMSprop(lr = 0.01)
model.compile(loss='categorical_crossentropy',optimizer=optimizer)


for iteration in range(1,100):
    print('Iteration',iteration)
    model.fit(x_train,y_train,batch_size=128,nb_epoch=1)
    if iteration%2 == 0:
        hit = 0
        start = random.randint(0,testSize-11)
        pres = model.predict(x_test[start:start+10])
        labs = y_test[start:start+10]
        for testIteration in range(10):
            hit += 1 if (np.argmax(pres[testIteration]) == np.argmax(labs[testIteration])) else 0
        print hit
        hit /= (10*len(pres[0]))
        print('test accuracy:',hit)
