import os
import sys
import csv
import copy
import math
import random
import pandas as pd
import numpy as np
import keras.metrics
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping
#from keras.utils import plot_model
from sklearn.metrics import roc_curve,auc,average_precision_score
from transfer_fasta_1 import data_process_1

from getData import get_data
from getData import get_data_test
from DProcess import convertRawToXY
from submodel import OnehotNetwork,OtherNetwork,PhysicochemicalNetwork,HydrophobicityNetwork,CompositionNetwork,BetapropensityNetwork,AlphaturnpropensityNetwork
import json
import keras.utils.np_utils as kutils
import tensorflow as tf
from random import choice


def normalization(array):

	normal_array = []
	de = array.sum()
	for i in array:
		normal_array.append(float(i)/de)

	return normal_array

'''
def normalization2(array):

	normal_array = []
	de = 1
	for i in array:
		de *= i
	print(de)
	for i in array:
		nu = i
		print(nu)
		normal_array.append(math.log(nu,de))

	return normal_array
'''
def normalization_softmax(array):

	normal_array = []
	de = 0
	for i in array:
		de += math.exp(i)
	for i in array:
		normal_array.append(math.exp(i)/de)

	return normal_array

def predict_stacked_model(model,test_oneofkeyX,test_physicalXo,test_physicalXp,test_physicalXh,test_physicalXc,test_physicalXb,test_physicalXa):

	testX = [test_oneofkeyX,test_physicalXo,test_physicalXp,test_physicalXh,test_physicalXc,test_physicalXb,test_physicalXa]

	return model.predict(testX,verbose=1)

def fit_stacked_model(model,val_oneofkeyX,val_physicalXo,val_physicalXp,val_physicalXh,val_physicalXc,val_physicalXb,val_physicalXa,valY):

	valX = [val_oneofkeyX,val_physicalXo,val_physicalXp,val_physicalXh,val_physicalXc,val_physicalXb,val_physicalXa]
	early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=5)

	model.fit(valX, valY, epochs=15,callbacks=[early_stopping],class_weight='auto',batch_size=4096,verbose=1,validation_split=0.2)

def define_stacked_model(members):
	# update all layers in all models to not be trainable
	for i in range(len(members)):
		model = members[i]
		for layer in model.layers:
			# make not trainable
			layer.trainable = False
			# rename to avoid 'unique layer name' issue
			layer.name = 'ensemble_' + str(i+1) + '_' + layer.name
	# define multi-headed input
	ensemble_visible = [model.input for model in members]
	# concatenate merge output from each model
	ensemble_outputs = [model.output for model in members]  ##### ensemble_outputs (len7) <class 'list'> model.output <class 'tensorflow.python.framework.ops.Tensor'>
	merge = concatenate(ensemble_outputs) ##### <class 'tensorflow.python.framework.ops.Tensor'>
	hidden = Dense(7, activation='relu')(merge)
	output = Dense(3, activation='softmax')(hidden)
	model = Model(inputs=ensemble_visible, outputs=output)
	#plot graph of ensemble
	#plot_model(model, show_shapes=True, to_file='model_graph.png')
	# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=[keras.metrics.categorical_accuracy])

	return model

def load_all_models(n_models,iteration_times):

	all_models = list()
	for i in range(n_models):
		if(i==0):
			filename = 'model/'+str(iteration_times)+'model/OnehotNetwork.h5'
		elif(i==1):
			filename = 'model/'+str(iteration_times)+'model/OtherNetwork.h5'
		elif(i==2):
			filename = 'model/'+str(iteration_times)+'model/PhysicochemicalNetwork.h5'
		elif(i==3):
			filename = 'model/'+str(iteration_times)+'model/HydrophobicityNetwork.h5'
		elif(i==4):
			filename = 'model/'+str(iteration_times)+'model/CompositionNetwork.h5'
		elif(i==5):
			filename = 'model/'+str(iteration_times)+'model/BetapropensityNetwork.h5'
		elif(i==6):
			filename = 'model/'+str(iteration_times)+'model/AlphaturnpropensityNetwork.h5'
		# load model from file
		model = load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)

	return all_models

def NewshufflePosNeg(data2):

	data2_over=[]
	index = [i for i in range(len(data2))]
	random.shuffle(index)

	data2_over = data2.iloc[:,:].values[index]
	data2_over = pd.DataFrame(data2_over)

	return data2_over

def Newshufflewrr(data1_0,data1_1,data1_2,data1_3):
	##### Create an index with nummber of posnum #####
	index = [i for i in range(len(data1_0))]
	random.shuffle(index)
	data1_0 = pd.DataFrame(data1_0)
	data1_0 = data1_0.iloc[:,:].values[index]
	data1_0_ss = pd.DataFrame(data1_0)

	index = [i for i in range(len(data1_1))]
	random.shuffle(index)
	data1_1 = pd.DataFrame(data1_1)
	data1_1 = data1_1.iloc[:,:].values[index]
	data1_1_ss = pd.DataFrame(data1_1)

	index = [i for i in range(len(data1_2))]
	random.shuffle(index)
	data1_2 = pd.DataFrame(data1_2)
	data1_2 = data1_2.iloc[:,:].values[index]
	data1_2_ss = pd.DataFrame(data1_2)

	index = [i for i in range(len(data1_3))]
	random.shuffle(index)
	data1_3 = pd.DataFrame(data1_3)
	data1_3 = data1_3.iloc[:,:].values[index]
	data1_3_ss = pd.DataFrame(data1_3)
	return data1_0_ss, data1_1_ss, data1_2_ss, data1_3_ss

def get_matrix(windows_0,windows_1,windows_2,windows_3):
	windows_0 = pd.DataFrame(windows_0)
	windows_1 = pd.DataFrame(windows_1)
	windows_2 = pd.DataFrame(windows_2)
	windows_3 = pd.DataFrame(windows_3)
	windows_all = pd.concat([windows_0,windows_1,windows_2,windows_3])
	windows_all = windows_all.iloc[:,:].values
	del windows_0,windows_1,windows_2,windows_3
	return windows_all



val_windows_0, val_windows_1,val_windows_2,val_windows_3 = get_data(r'd/val.txt',r'data/pssmpickle2/', label=True)
all_train_windows_0, all_train_windows_1,all_train_windows_2,all_train_windows_3 = get_data(r'd/train.txt',r'data/pssmpickle2/', label=True)
test_windows_0, test_windows_1,test_windows_2,test_windows_3,name_list,id_list = get_data_test(r'd/test.txt',r'data/pssmpickle2/',label=True)

ff = int(len(test_windows_1)+len(test_windows_2)+len(test_windows_3))  #Fractional factor
test_windows_0 = test_windows_0[0:ff]
ff = int(len(test_windows_1))
test_windows_1 = test_windows_1[0:ff]
ff = int(len(test_windows_2))
test_windows_2 = test_windows_2[0:ff]
ff = int(len(test_windows_3))
test_windows_3 = test_windows_3[0:ff]
test_windows_all = get_matrix(test_windows_0, test_windows_1, test_windows_2, test_windows_3)
# print(test_windows_all)

test_oneofkeyX, testY = convertRawToXY(test_windows_all,codingMode=0)
test_oneofkeyX.shape = (test_oneofkeyX.shape[0],test_oneofkeyX.shape[2],test_oneofkeyX.shape[3])
test_physicalXo, _ = convertRawToXY(test_windows_all,codingMode=9)
test_physicalXo.shape = (test_physicalXo.shape[0],test_physicalXo.shape[2],test_physicalXo.shape[3])
test_physicalXp, _ = convertRawToXY(test_windows_all,codingMode=10)
test_physicalXp.shape = (test_physicalXp.shape[0],test_physicalXp.shape[2],test_physicalXp.shape[3])
test_physicalXh, _ = convertRawToXY(test_windows_all,codingMode=11)
test_physicalXh.shape = (test_physicalXh.shape[0],test_physicalXh.shape[2],test_physicalXh.shape[3])
test_physicalXc, _ = convertRawToXY(test_windows_all,codingMode=12)
test_physicalXc.shape = (test_physicalXc.shape[0],test_physicalXc.shape[2],test_physicalXc.shape[3])
test_physicalXb, _ = convertRawToXY(test_windows_all,codingMode=13)
test_physicalXb.shape = (test_physicalXb.shape[0],test_physicalXb.shape[2],test_physicalXb.shape[3])
test_physicalXa, _ = convertRawToXY(test_windows_all,codingMode=14)
test_physicalXa.shape = (test_physicalXa.shape[0],test_physicalXa.shape[2],test_physicalXa.shape[3])

print("Test data coding finished!")

ff = int(len(val_windows_1)+len(val_windows_2)+len(val_windows_3))  #Fractional factor
val_windows_0 = val_windows_0[0:ff]
ff = int(len(val_windows_1))
val_windows_1 = val_windows_1[0:ff]
ff = int(len(val_windows_2))
val_windows_2 = val_windows_2[0:ff]
ff = int(len(val_windows_3))
val_windows_3 = val_windows_3[0:ff]
val_windows_all = get_matrix(val_windows_0, val_windows_1,val_windows_2,val_windows_3)

val_oneofkeyX,valY = convertRawToXY(val_windows_all,codingMode=0)
val_oneofkeyX.shape = (val_oneofkeyX.shape[0],val_oneofkeyX.shape[2],val_oneofkeyX.shape[3])
val_physicalXo,_ = convertRawToXY(val_windows_all,codingMode=9)
val_physicalXo.shape = (val_physicalXo.shape[0],val_physicalXo.shape[2],val_physicalXo.shape[3])
val_physicalXp,_ = convertRawToXY(val_windows_all,codingMode=10)
val_physicalXp.shape = (val_physicalXp.shape[0],val_physicalXp.shape[2],val_physicalXp.shape[3])
val_physicalXh,_ = convertRawToXY(val_windows_all,codingMode=11)
val_physicalXh.shape = (val_physicalXh.shape[0],val_physicalXh.shape[2],val_physicalXh.shape[3])
val_physicalXc,_ = convertRawToXY(val_windows_all,codingMode=12)
val_physicalXc.shape = (val_physicalXc.shape[0],val_physicalXc.shape[2],val_physicalXc.shape[3])
val_physicalXb,_ = convertRawToXY(val_windows_all,codingMode=13)
val_physicalXb.shape = (val_physicalXb.shape[0],val_physicalXb.shape[2],val_physicalXb.shape[3])
val_physicalXa,_ = convertRawToXY(val_windows_all,codingMode=14)
val_physicalXa.shape = (val_physicalXa.shape[0],val_physicalXa.shape[2],val_physicalXa.shape[3])
del val_windows_0, val_windows_1,val_windows_2,val_windows_3,val_windows_all

print("Val data coding finished!")

iteration_times = 13
for t in range(0,iteration_times):
	print("iteration_times: %d"%t)
	train_windows_0, train_windows_1,train_windows_2,train_windows_3 = Newshufflewrr(all_train_windows_0, all_train_windows_1,all_train_windows_2,all_train_windows_3)

	ff = int(len(train_windows_1)+len(train_windows_2)+len(train_windows_3))  #Fractional factor
	train_windows_0 = train_windows_0[0:ff]
	ff = int(len(train_windows_1))
	train_windows_1 = train_windows_1[0:ff]
	ff = int(len(train_windows_2))
	train_windows_2 = train_windows_2[0:ff]#ff
	ff = int(len(train_windows_3))
	train_windows_3 = train_windows_3[0:ff]#ff

	print("train_0_num: %d"%len(train_windows_0))
	print("train_1_num: %d"%len(train_windows_1))
	print("train_2_num: %d"%len(train_windows_2))
	print("train_3_num: %d"%len(train_windows_3))

	train_windows_all = pd.concat([train_windows_0, train_windows_1,train_windows_2,train_windows_3])
	train_windows_all = NewshufflePosNeg(train_windows_all)
	train_windows_all = pd.DataFrame(train_windows_all)
	matrix_train_windows_all = train_windows_all.iloc[:,:].values

	train_oneofkeyX,trainY = convertRawToXY(matrix_train_windows_all,codingMode=0)
	train_oneofkeyX.shape = (train_oneofkeyX.shape[0],train_oneofkeyX.shape[2],train_oneofkeyX.shape[3])
	train_physicalXo,_ = convertRawToXY(matrix_train_windows_all,codingMode=9)
	train_physicalXo.shape = (train_physicalXo.shape[0],train_physicalXo.shape[2],train_physicalXo.shape[3])
	train_physicalXp,_ = convertRawToXY(matrix_train_windows_all,codingMode=10)
	train_physicalXp.shape = (train_physicalXp.shape[0],train_physicalXp.shape[2],train_physicalXp.shape[3])
	train_physicalXh,_ = convertRawToXY(matrix_train_windows_all,codingMode=11)
	train_physicalXh.shape = (train_physicalXh.shape[0],train_physicalXh.shape[2],train_physicalXh.shape[3])
	train_physicalXc,_ = convertRawToXY(matrix_train_windows_all,codingMode=12)
	train_physicalXc.shape = (train_physicalXc.shape[0],train_physicalXc.shape[2],train_physicalXc.shape[3])
	train_physicalXb,_ = convertRawToXY(matrix_train_windows_all,codingMode=13)
	train_physicalXb.shape = (train_physicalXb.shape[0],train_physicalXb.shape[2],train_physicalXb.shape[3])
	train_physicalXa,_ = convertRawToXY(matrix_train_windows_all,codingMode=14)
	train_physicalXa.shape = (train_physicalXa.shape[0],train_physicalXa.shape[2],train_physicalXa.shape[3])
	print("itreation %d times Train data coding finished!" %t)

	if(t==0):
		struct_Onehot_model = OnehotNetwork(train_oneofkeyX,trainY,val_oneofkeyX,valY,train_time=t)
		physical_O_model = OtherNetwork(train_physicalXo,trainY,val_physicalXo,valY,train_time=t)
		physical_P_model = PhysicochemicalNetwork(train_physicalXp,trainY,val_physicalXp,valY,train_time=t)
		physical_H_model = HydrophobicityNetwork(train_physicalXh,trainY,val_physicalXh,valY,train_time=t)
		physical_C_model = CompositionNetwork(train_physicalXc,trainY,val_physicalXc,valY,train_time=t)
		physical_B_model = BetapropensityNetwork(train_physicalXb,trainY,val_physicalXb,valY,train_time=t)
		physical_A_model = AlphaturnpropensityNetwork(train_physicalXa,trainY,val_physicalXa,valY,train_time=t)
		print("itreation %d times training finished!" %t)
	else:
		struct_Onehot_model = OnehotNetwork(train_oneofkeyX,trainY,val_oneofkeyX,valY,train_time=t,compilemodels=struct_Onehot_model)

		physical_O_model = OtherNetwork(train_physicalXo,trainY,val_physicalXo,valY,train_time=t,compilemodels=physical_O_model)
		physical_P_model = PhysicochemicalNetwork(train_physicalXp,trainY,val_physicalXp,valY,train_time=t,compilemodels=physical_P_model)
		physical_H_model = HydrophobicityNetwork(train_physicalXh,trainY,val_physicalXh,valY,train_time=t,compilemodels=physical_H_model)
		physical_C_model = CompositionNetwork(train_physicalXc,trainY,val_physicalXc,valY,train_time=t,compilemodels=physical_C_model)
		physical_B_model = BetapropensityNetwork(train_physicalXb,trainY,val_physicalXb,valY,train_time=t,compilemodels=physical_B_model)
		physical_A_model = AlphaturnpropensityNetwork(train_physicalXa,trainY,val_physicalXa,valY,train_time=t,compilemodels=physical_A_model)
		print("itreation %d times training finished!" %t)


	monitor = 'val_loss'
	weights = []
	with open ('model/loss/'+str(t)+'onehotloss.json', 'r') as checkpoint_fp:
		weights.append(1/float(json.load(checkpoint_fp)[monitor]))
	with open ('model/loss/'+str(t)+'Onetloss.json', 'r') as checkpoint_fp:
		weights.append(1/float(json.load(checkpoint_fp)[monitor]))
	with open ('model/loss/'+str(t)+'Pnetloss.json', 'r') as checkpoint_fp:
		weights.append(1/float(json.load(checkpoint_fp)[monitor]))
	with open ('model/loss/'+str(t)+'Hnetloss.json', 'r') as checkpoint_fp:
		weights.append(1/float(json.load(checkpoint_fp)[monitor]))
	with open ('model/loss/'+str(t)+'Cnetloss.json', 'r') as checkpoint_fp:
		weights.append(1/float(json.load(checkpoint_fp)[monitor]))
	with open ('model/loss/'+str(t)+'Bnetloss.json', 'r') as checkpoint_fp:
		weights.append(1/float(json.load(checkpoint_fp)[monitor]))
	with open ('model/loss/'+str(t)+'Anetloss.json', 'r') as checkpoint_fp:
		weights.append(1/float(json.load(checkpoint_fp)[monitor]))

	weight_array = np.array(weights, dtype= np.float)
	del weights

	#normalize checkpoit data as weights
	weight_array = normalization(weight_array)

	# data = pd.DataFrame(weight_array)
	# writer = pd.ExcelWriter('result/weight_array.xlsx')		# 写入Excel文件
	# data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
	# writer.save()
	# writer.close()
	# print('write weight*****************************')

	#weight_array = normalization_softmax(weight_array)

	predict_weighted_merge = 0
	#load model weights and checkpoint file
	predict_temp = weight_array[0] * struct_Onehot_model.predict(test_oneofkeyX)
	predict_weighted_merge += predict_temp

	data = pd.DataFrame(struct_Onehot_model.predict(test_oneofkeyX))
	writer = pd.ExcelWriter('result/'+str(t)+'_subnet_1.xlsx')		# 写入Excel文件
	data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
	writer.save()
	writer.close()
	print('write input*****************************')
	result = np.column_stack((testY[:,0],struct_Onehot_model.predict(test_oneofkeyX)[:,0])) # 1.0  1.0  three col score
	result = pd.DataFrame(result)
	result.to_csv(path_or_buf='result/'+str(t)+'_ubi1'+'.txt',index=False,header=None,sep='\t',quoting=csv.QUOTE_NONE)

	result = np.column_stack((testY[:,1],struct_Onehot_model.predict(test_oneofkeyX)[:,1])) # 1.0  1.0  three col score
	result = pd.DataFrame(result)
	result.to_csv(path_or_buf='result/'+str(t)+'_sumo1'+'.txt',index=False,header=None,sep='\t',quoting=csv.QUOTE_NONE)

	
	predict_temp = weight_array[1] * physical_O_model.predict(test_physicalXo)
	data = pd.DataFrame(physical_O_model.predict(test_physicalXo))
	writer = pd.ExcelWriter('result/'+str(t)+'_subnet_2.xlsx')		# 写入Excel文件
	data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
	writer.save()
	writer.close()
	print('write input*****************************')
	result = np.column_stack((testY[:,0],physical_O_model.predict(test_physicalXo)[:,0])) # 1.0  1.0  three col score
	result = pd.DataFrame(result)
	result.to_csv(path_or_buf='result/'+str(t)+'_ubi2'+'.txt',index=False,header=None,sep='\t',quoting=csv.QUOTE_NONE)

	result = np.column_stack((testY[:,1],physical_O_model.predict(test_physicalXo)[:,1])) # 1.0  1.0  three col score
	result = pd.DataFrame(result)
	result.to_csv(path_or_buf='result/'+str(t)+'_sumo2'+'.txt',index=False,header=None,sep='\t',quoting=csv.QUOTE_NONE)

	


	predict_weighted_merge += predict_temp

	predict_temp = weight_array[2] * physical_P_model.predict(test_physicalXp)
	data = pd.DataFrame(physical_P_model.predict(test_physicalXp))
	writer = pd.ExcelWriter('result/'+str(t)+'_subnet_3.xlsx')		# 写入Excel文件
	data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
	writer.save()
	writer.close()
	print('write input*****************************')
	result = np.column_stack((testY[:,0],physical_P_model.predict(test_physicalXp)[:,0]))# 1.0  1.0  three col score
	result = pd.DataFrame(result)
	result.to_csv(path_or_buf='result/'+str(t)+'_ubi3'+'.txt',index=False,header=None,sep='\t',quoting=csv.QUOTE_NONE)

	result = np.column_stack((testY[:,1],physical_P_model.predict(test_physicalXp)[:,1])) # 1.0  1.0  three col score
	result = pd.DataFrame(result)
	result.to_csv(path_or_buf='result/'+str(t)+'_sumo3'+'.txt',index=False,header=None,sep='\t',quoting=csv.QUOTE_NONE)

	predict_weighted_merge += predict_temp

	predict_temp = weight_array[3] * physical_H_model.predict(test_physicalXh)
	data = pd.DataFrame(physical_H_model.predict(test_physicalXh))
	writer = pd.ExcelWriter('result/'+str(t)+'_subnet_4.xlsx')		# 写入Excel文件
	data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
	writer.save()
	writer.close()
	print('write input*****************************')
	result = np.column_stack((testY[:,0],physical_H_model.predict(test_physicalXh)[:,0]))# 1.0  1.0  three col score
	result = pd.DataFrame(result)
	result.to_csv(path_or_buf='result/'+str(t)+'_ubi4'+'.txt',index=False,header=None,sep='\t',quoting=csv.QUOTE_NONE)

	result = np.column_stack((testY[:,1],physical_H_model.predict(test_physicalXh)[:,1])) # 1.0  1.0  three col score
	result = pd.DataFrame(result)
	result.to_csv(path_or_buf='result/'+str(t)+'_sumo4'+'.txt',index=False,header=None,sep='\t',quoting=csv.QUOTE_NONE)

	predict_weighted_merge += predict_temp

	predict_temp = weight_array[4] * physical_C_model.predict(test_physicalXc)
	data = pd.DataFrame(physical_C_model.predict(test_physicalXc))
	writer = pd.ExcelWriter('result/'+str(t)+'_subnet_5.xlsx')		# 写入Excel文件
	data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
	writer.save()
	writer.close()
	print('write input*****************************')
	result = np.column_stack((testY[:,0],physical_C_model.predict(test_physicalXc)[:,0]))# 1.0  1.0  three col score
	result = pd.DataFrame(result)
	result.to_csv(path_or_buf='result/'+str(t)+'_ubi5'+'.txt',index=False,header=None,sep='\t',quoting=csv.QUOTE_NONE)

	result = np.column_stack((testY[:,1],physical_C_model.predict(test_physicalXc)[:,1])) # 1.0  1.0  three col score
	result = pd.DataFrame(result)
	result.to_csv(path_or_buf='result/'+str(t)+'_sumo5'+'.txt',index=False,header=None,sep='\t',quoting=csv.QUOTE_NONE)

	predict_weighted_merge += predict_temp

	predict_temp = weight_array[5] * physical_B_model.predict(test_physicalXb)
	data = pd.DataFrame(physical_B_model.predict(test_physicalXb))
	writer = pd.ExcelWriter('result/'+str(t)+'_subnet_6.xlsx')		# 写入Excel文件
	data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
	writer.save()
	writer.close()
	print('write input*****************************')
	result = np.column_stack((testY[:,0],physical_B_model.predict(test_physicalXb)[:,0]))# 1.0  1.0  three col score
	result = pd.DataFrame(result)
	result.to_csv(path_or_buf='result/'+str(t)+'_ubi6'+'.txt',index=False,header=None,sep='\t',quoting=csv.QUOTE_NONE)

	result = np.column_stack((testY[:,1],physical_B_model.predict(test_physicalXb)[:,1])) # 1.0  1.0  three col score
	result = pd.DataFrame(result)
	result.to_csv(path_or_buf='result/'+str(t)+'_sumo6'+'.txt',index=False,header=None,sep='\t',quoting=csv.QUOTE_NONE)

	predict_weighted_merge += predict_temp

	predict_temp = weight_array[6] * physical_A_model.predict(test_physicalXa)
	data = pd.DataFrame(physical_A_model.predict(test_physicalXa))
	writer = pd.ExcelWriter('result'+str(t)+'_subnet_7.xlsx')		# 写入Excel文件
	data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
	writer.save()
	writer.close()
	print('write input*****************************')
	result = np.column_stack((testY[:,0],physical_A_model.predict(test_physicalXa)[:,0]))# 1.0  1.0  three col score
	result = pd.DataFrame(result)
	result.to_csv(path_or_buf='result/'+str(t)+'_ubi7'+'.txt',index=False,header=None,sep='\t',quoting=csv.QUOTE_NONE)

	result = np.column_stack((testY[:,1],physical_A_model.predict(test_physicalXa)[:,1])) # 1.0  1.0  three col score
	result = pd.DataFrame(result)
	result.to_csv(path_or_buf='result/'+str(t)+'_sumo7'+'.txt',index=False,header=None,sep='\t',quoting=csv.QUOTE_NONE)

	predict_weighted_merge += predict_temp


	print('predict_weighted_merge',predict_weighted_merge)
	score = predict_weighted_merge
	data = pd.DataFrame(score)
	writer = pd.ExcelWriter('result/'+str(t)+'_ensemble.xlsx')		# 写入Excel文件
	data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
	writer.save()
	writer.close()
	print('write output*****************************')
	result = np.column_stack((testY[:,0],predict_weighted_merge[:,0]))# 1.0  1.0  three[] col score
	result = pd.DataFrame(result)
	result.to_csv(path_or_buf='result/'+str(t)+'_ubi0'+'.txt',index=False,header=None,sep='\t',quoting=csv.QUOTE_NONE)

	result = np.column_stack((testY[:,1],predict_weighted_merge[:,1])) # 1.0  1.0  three col score
	result = pd.DataFrame(result)
	result.to_csv(path_or_buf='result/'+str(t)+'_sumo0'+'.txt',index=False,header=None,sep='\t',quoting=csv.QUOTE_NONE)

