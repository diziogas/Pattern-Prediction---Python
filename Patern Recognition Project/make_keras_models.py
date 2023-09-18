#to neural network ginetai edw
from __future__ import print_function
# from Paper import *
# import pickle
# from timeit import default_timer as timer
import numpy as np
# import tensorflow as tf
import sys
import matplotlib.pyplot as plt
# import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
import random
from timeit import default_timer as timer

def toList(text):
	intlist = [int(i) for i in text.split(',')]
	return intlist

timeline_years = []
timeline_refs = []

dataset = None

if len(sys.argv) < 2:
	print("Give timeline length as sys arguement")
	sys.exit()

SAMPLE_LENGTH = int(sys.argv[1])
print(SAMPLE_LENGTH)
# sys.exit()
TRAIN_SIZE = SAMPLE_LENGTH-1

with open('timelines/timeline'+str(SAMPLE_LENGTH)+'_years_refs.txt','r') as handle:
	i = 0
	for line in handle:
		line = line.rstrip().rstrip(',')
		newlist = toList(line)
		if i % 2 == 0:#even = years
			timeline_years.append(newlist)
		else:#odd = references
			timeline_refs.append(newlist)
		i+=1

new_timeline_years = []
refs = []

count0 = 0

for i,ref in enumerate(timeline_refs):
	# for r in ref:
	# 	if r == 0:
	# 		count0 += 1
	if len(ref) == SAMPLE_LENGTH:#cut reference list with length > 20
	# if count0 < SAMPLE_LENGTH/2:
		refs.append(ref)
		new_timeline_years.append(timeline_years[i])
# new_timeline_years = timeline_years
print(len(refs))
dataset = np.array(refs)
yearset = np.array(new_timeline_years)

start = timer()

# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)

test_size=SAMPLE_LENGTH - TRAIN_SIZE
train,test=dataset[:,0:TRAIN_SIZE],dataset[:,TRAIN_SIZE]

look_back = 1

train = np.reshape(train,(train.shape[0],1,train.shape[1]))
test = np.reshape(test,(test.shape[0],1))

model = Sequential()
model.add(LSTM(TRAIN_SIZE, input_shape=(1, TRAIN_SIZE)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(train, test, epochs=100, batch_size=1, verbose=2)
model.evaluate(train,test)

end = timer()
diff= end-start
print('Time for the system to train the model'+str(SAMPLE_LENGTH)+' in ' + str(diff) +' seconds.')

#Save the model
# serialize model to JSON
model_json = model.to_json()
with open("models/model"+str(SAMPLE_LENGTH)+".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/modelweights"+str(SAMPLE_LENGTH)+".h5")
print("Saved model"+str(SAMPLE_LENGTH)+" to disk")