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
    if label == ' Background activity':
        continue
    videoName = arr[1]
    if videoName != 's08-d02-cam-002':
        break
    startFrame = int(arr[2])
    endFrame = int(arr[3])
    for frame in range(startFrame,endFrame):
        Y[startFrame:endFrame+1] = label

maxlen = 50
step = 3
x_ = np.array([])
y_ = np.array([])

for i in range(0,len(X)-maxlen,step):
    x_.append(X[i:i+maxlen])
    y_.append(Y[i:i+maxlen])

model = Seqnential()
model.add(LSTM(128, input_shape=()))
model.add(Dense())
model.add(Activation('softmax'))

optimizer = RMSprop(lr = 0.01)
model.compile(loss='categorical_crossentropy',optimizer=optimizer)


for iteration in range(1,100):
    print('Iteration',iteration)
    model.fit(x_,y_,batch_size=128,nb_epoch=1)
    if iteration%10 == 0:
        hit = 0
        for testIteration in range(10):
            start = random.randint(0,len(X)-maxlen)
            preds = model.predict(X[start:start+maxlen])
            labs = Y[start:start+maxlen]
            for idx in range(len(preds)):
                hit += 1 if (preds[idx] == labs[idx]) else 0
        hit /= (10*maxlen)
        print('test accuracy:',hit)
