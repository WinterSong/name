# coding: utf-8

# this script is another way to organize the method to feed lstm

from keras.models import Seqnential
from keras.layers import Dense, Activation, LSTM, Dropout
import numpy as np
import random
import h5py
inputFile = open('~/disk/mpii/labels/detectionGroundtruth-1-0.csv','r')
# read the feature
feature = h5py.File('~/disk/mpii/features/hog/IntegralHist/s08-d02-hard-0.mat','r')
X = np.array(feature.get('integralFrameHist'))
Y = np.zeros(len(X))
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
x_ = []
y_ = []

for i in range(0,len(X)-maxlen,step):
    x_.append(X[i:i+maxlen])
    tmp = [0]*65
    flag = 0
    for j in range(i,i+maxlen):
        if (Y[j] == flag or Y[j] == 0):
            continue
        else:
            tmp[Y[j]] = 1
            flag = Y[j]
    y_.append(tmp)

model = Seqnential()
model.add(LSTM(128, input_shape=(maxlen,len(X[0]))))
model.add(Dense(65))
model.add(Activation('softmax'))

optimizer = RMSprop(lr = 0.01)
model.compile(loss='categorical_crossentropy',optimizer=optimizer)


for iteration in range(1,100):
    print('Iteration',iteration)
    model.fit(x_,y_,batch_size=128,nb_epoch=1)
    if iteration%2 == 0:
        hit = 0
        start = random.randint(0,int((len(X)-maxlen)/step)-10)
        pres = model.predict(x_[start:start+10])
        labs = y_[start:start+10]
        for testIteration in range(10):
            hit += 1 if (labs[testIteration][idx])
            for idx in range(len(pres[testIteration])):
                hit += 1 if (np.argmax(pres[testIteration][idx]) == np.argmax(labs[testIteration][idx])) else 0
        hit /= (10*len(pres[0]))
        print('test accuracy:',hit)