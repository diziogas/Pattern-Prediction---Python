#to neural network ginetai edw
from __future__ import print_function
# from Paper import *
# import pickle
# from timeit import default_timer as timer
import numpy as np
#import tensorflow as tf
import sys
import matplotlib.pyplot as plt
# import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import model_from_json
from keras import utils
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
import random
from timeit import default_timer as timer

def toList(text):
	intlist = [int(i) for i in text.split(',')]
	return intlist

def nameIteratorBits(iter,bits,type=''):
	length = len(str(iter))
	filename = ""
	rightMostBits = bits - length
	if rightMostBits > 0:
		filename = '0'*rightMostBits + str(iter)
	else:
		filename = str(iter)
	return filename+type

# if len(sys.argv) < 2:
# 	print("Give timeline length as sys arguement")
# 	sys.exit()

start = timer()




EXAMS=True
filepath = ''
filepaths = []
if EXAMS == False:
	filepath = 'timelines/timeline'+str(20)+'_years_refs.txt'
	filepaths.append(filepath)
else:
	for i in range(1,21):
		filepath = 'timeseries/timeseries'+nameIteratorBits(i,2,'.txt')
		filepaths.append(filepath)

print(filepaths)
for filepath in filepaths:
	timeline_years = []
	timeline_refs = []

	dataset = None
	irefs = 0
	iyears = 0
	with open(filepath,'r') as handle:
		i = 0
		for line in handle:
			line = line.rstrip().rstrip(',')
			newlist = toList(line)
			if EXAMS == False:
				if i % 2 == 0:#even = years
					timeline_years.append(newlist)
					iyears += 1
				else:#odd = references
					timeline_refs.append(newlist)
					irefs += 1
			else:#references only
				timeline_refs.append(newlist)
				irefs += 1
			i+=1

	# sys.exit()
	TRAIN_SIZE = len(timeline_refs[0])
	# TRAIN_SIZE = 19
	SAMPLE_LENGTH = TRAIN_SIZE+1
	print(SAMPLE_LENGTH)

	new_timeline_years = []
	# refs = []

	# for i,ref in enumerate(timeline_refs):
	# 	if len(ref) == SAMPLE_LENGTH:#cut reference list with length > 20
	# 		refs.append(ref)
	# 		new_timeline_years.append(timeline_years[i])

	# dataset = np.array(refs)
	# yearset = np.array(new_timeline_years)

	# train,test=dataset[:,0:TRAIN_SIZE],dataset[:,TRAIN_SIZE]

	# look_back = 1

	# train = np.reshape(train,(train.shape[0],1,train.shape[1]))
	# test = np.reshape(test,(test.shape[0],1))


	# load json and create model
	json_file = open('models/model'+str(SAMPLE_LENGTH)+'.json', 'r')
	model = json_file.read()
	json_file.close()
	model = model_from_json(model)
	# load weights into new model
	weights = model.load_weights("models/modelweights"+str(SAMPLE_LENGTH)+".h5")
	print("Loaded model"+str(SAMPLE_LENGTH)+" from disk")

	# evaluate loaded model on test data
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	# print('loss,accuracy:')
	#print(model.evaluate(train, test, verbose=0))
	# print("%s: %.2f%%" + str(model.metrics_names)))

	# input_sample = [3,1,4,6,1,6,0,2,0,0]
	# input_sample = np.reshape(input_sample,(1,1,TRAIN_SIZE))
	# input_sample_pred=model.predict(input_sample)
	# input_sample = np.reshape(input_sample,(TRAIN_SIZE,1))
	# print(input_sample)
	# print('Prediction for 5 : '+str(input_sample_pred[0][0]))
	# print('Correct for 5 : ' + str(1))
	# load json and create model
	model.summary(line_length=None, positions=None, print_fn=None)


	json_file5 = open('models5/model'+str(SAMPLE_LENGTH-1)+'.json', 'r')
	model5 = json_file5.read()
	json_file5.close()
	model5 = model_from_json(model5)
	# load weights into new model
	model5.load_weights("models5/modelweights"+str(SAMPLE_LENGTH-1)+".h5")
	print("Loaded model"+str(SAMPLE_LENGTH-1)+" from disk")
	# evaluate loaded model on test data
	model5.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

	# test5=[1 ,6 ,4 ,7 ,1 ,4 ,3 ,4 ,4 ,0 ,1 ,0 ,3 ,2 ,2 ,0 ,0 ,1 ,1 ]
	# random =[4,0,1,1,2,2]
	# test5r=np.reshape(test5,(1,1,19))
	# test5pre = model5.predict(test5r)
	# print('Prediction is : '+str(test5pre[0][0]))
	# print('Correct is :' +str(random[-1]))
	# plt.plot(test5+random)
	# plt.plot(test5+random[:len(random)-1]+[test5pre[0][0]])
	# plt.show()
	
	# sys.exit()

	for timeline in timeline_refs:
		tr = np.reshape(timeline,(1,1,TRAIN_SIZE))
		prediction1 = model.predict(tr)
		print('Prediction for the next year is : '+str(prediction1[0][0]))
		prediction5=model5.predict(tr)
		print('Prediction for the next five years is : '+str(prediction5[0][0]))
		end = timer()
		diff= end-start
		print('Time for the system to answer is '+str(diff)+' seconds.')
	# filepath_goal = filepath[:len(filepath)-4]
	# filepath_goal += '_goal.txt'
	
	# with open(filepath_goal) as handle:
	# 	print('Opening '+filepath_goal+' file')
	# 	count = 0
	# 	for line in handle:
	# 		correct = int(line)
			
	# 		tr = np.reshape(timeline_refs[count],(1,1,TRAIN_SIZE))
	# 		prediction1 = model.predict(tr)
	# 		# print(dataset[0])
	# 		print('Prediction for the next year is : '+str(prediction1[0][0]))
	# 		print('Correct for the next year is : '+str(correct))
	# 		# sys.exit()

	# 		prediction5=model5.predict(tr)
	# 		print('Prediction for the next five years is : '+str(prediction5[0][0]))
	# 		print('Correct for the next five years is : ' + str(4))

	# 		end = timer()
	# 		diff= end-start
	# 		print('Time for the system to answer is '+str(diff)+' seconds.')
	# 		count += 1

	if EXAMS == False:
		predictions = []
		for index in range(0,len(dataset)):
			#index = int(input('Give me a sample index, [0-'+str(len(train))+']'))
			
			# t = np.reshape([1]*15,(1,1,15))
			# prediction = model.predict(t)
			t = np.reshape(train[index,0], (1,1,TRAIN_SIZE))
			prediction = model.predict(t)
			# print('Correct is = ' + str(dataset[index][TRAIN_SIZE]))
			# print('Prediction is = ' + str(prediction))
			predictions.append(prediction[0][0])

		for i in range(10):
			index = random.randrange(0,len(dataset))
			plt.plot(dataset[index,0:TRAIN_SIZE+1])
			pred_list = [0]*(TRAIN_SIZE+1)
			for j in range(0,TRAIN_SIZE):
				pred_list[j] = dataset[index,j]
			# print(predictions)
			pred_list[TRAIN_SIZE] = predictions[index]
			print(dataset[index,0:TRAIN_SIZE+1])
			print(pred_list)
			plt.plot(pred_list)
			plt.show()


	
	# pp = [1,1,0,2,0,3,4,3,1]#,0,0,3,1,2,]
	# # pp = [2,6,1,5,2,10,4,5,2]#,2,2,2,5,3]
	# pp=[1,0,0,1,0,1,0,0,0,0,0,0,0,2,0,1,2,1,2,1,1,3,0,0,0,3,2,2,0,2,1,1,0,1,0,1,1,1]#,0,3,0,3,4,]
	# pp = np.reshape(pp,(1,1,TRAIN_SIZE))

	
