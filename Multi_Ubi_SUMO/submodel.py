import numpy as np
import pandas as pd
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.utils.np_utils as kutils
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint,Callback,History
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from keras import backend as K
import keras.metrics
import math
import matplotlib.pyplot as plt
from LossCheckPoint import LossModelCheckpoint
from keras.models import load_model
from keras.models import Model
from keras import backend as K
import tensorflow as tf  
#1
def OnehotNetwork(train_oneofkeyX,trainY,val_oneofkeyX,valY,
				#pre_train_total_path = 'model/pretrain.h5',
				train_time=None,compilemodels=None):
	# Oneofkey_inputs = tf.placeholder(tf.float32,[49,21])
	Oneofkey_inputs = Input(shape= (train_oneofkeyX.shape[1],train_oneofkeyX.shape[2]))  #49*21=1029
	early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=7)
	
	if (train_time==0):
		x = conv.Conv1D(201,2,init='glorot_normal',W_regularizer= l1(0),border_mode="same")(Oneofkey_inputs)
		x = Dropout(0.2)(x)
		x = Activation('relu')(x)

		x = conv.Conv1D(151,3,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(x)
		x = Dropout(0.2)(x)
		x = Activation('relu')(x)

		x = conv.Conv1D(101,5,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(x)
		x = Dropout(0.2)(x)
		x = Activation('relu')(x)

		x = core.Flatten()(x)
		x = BatchNormalization()(x)

		x = Dense(256,init='glorot_normal',activation='relu')(x)
		x = Dropout(0.2)(x)

		x = Dense(128,init='glorot_normal',activation="relu")(x)
		# print('sigmoid')
		# print(x)
		Oneofkey_outputs = Dense(2, init='glorot_normal', activation='sigmoid', W_regularizer = l2(0.001))(x)
		# print(Oneofkey_outputs)

		OnehotNetwork = Model(Oneofkey_inputs,Oneofkey_outputs)

		#optimization = SGD(lr=0.01, momentum=0.9, nesterov= True)
		optimization='Nadam'
		OnehotNetwork.compile(loss='binary_crossentropy',optimizer=optimization,metrics=['accuracy'])
	else:
		OnehotNetwork = compilemodels
		#OnehotNetwork = load_model("model/"+str(train_time-1)+'model/OnehotNetwork.h5')

	if(trainY is not None):
		#checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'model/OnehotNetwork.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		weight_checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'modelweight/OnehotNetworkweight.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min',save_weights_only=True)
		loss_checkpointer = LossModelCheckpoint(model_file_path="model/"+str(train_time)+'model/OnehotNetwork.h5',monitor_file_path="model/loss/"+str(train_time)+"onehotloss.json",verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		#onehotfitHistory = OnehotNetwork.fit(train_oneofkeyX,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,checkpointer,weight_checkpointer],class_weight={0:0.1,1:1},validation_data=(val_oneofkeyX,valY))
	

		onehotfitHistory = OnehotNetwork.fit(train_oneofkeyX,trainY,batch_size=4096,epochs=50,shuffle=True,callbacks=[early_stopping,loss_checkpointer,weight_checkpointer],class_weight='auto',validation_data=(val_oneofkeyX,valY))

		OnehotNetwork = load_model("model/"+str(train_time)+'model/OnehotNetwork.h5')


	return OnehotNetwork
#2
def OtherNetwork(train_physicalXo,trainY,val_physicalXo,valY,
			#pre_train_total_path = 'model/pretrain.h5',
			train_time=None,compilemodels=None):

	physical_O_inputs = Input(shape=(train_physicalXo.shape[1],train_physicalXo.shape[2]))  #49*28=1372

	early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=5)

	if (train_time==0):
		x = core.Flatten()(physical_O_inputs)
		x = BatchNormalization()(x)
		print(x)
		with open('result/x1_inputs.txt', mode='a') as resFile:
			resFile.write(str(x)+'\r\n')
		x = Dense(256,init='glorot_normal',activation='relu')(x)
		x = BatchNormalization()(x)
		# x = Dropout(0.2)(x)

		x = Dense(128,init='glorot_normal',activation='relu')(x)
		x = BatchNormalization()(x)
		# x = Dropout(0.2)(x)
		# tf.print(x)
		# with open('result/physical_O_outputs.txt', mode='a') as resFile:
		# 	resFile.write(tf.print(x))
		physical_O_outputs = Dense(2,init='glorot_normal',activation='sigmoid',W_regularizer= l2(0.001))(x)

		early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=5)

		OtherNetwork = Model(physical_O_inputs,physical_O_outputs)

		#optimization = SGD(lr=0.01, momentum=0.9, nesterov= True)
		optimization='Nadam'
		OtherNetwork.compile(loss='binary_crossentropy',optimizer=optimization,metrics=['accuracy'])
	else:
		OtherNetwork = compilemodels
		#OtherNetwork = load_model("model/"+str(train_time-1)+'model/OtherNetwork.h5')

	if(trainY is not None):
		#checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'model/OtherNetwork.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		weight_checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'modelweight/OtherNetworkweight.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min',save_weights_only=True)
		loss_checkpointer = LossModelCheckpoint(model_file_path="model/"+str(train_time)+'model/OtherNetwork.h5',monitor_file_path="model/loss/"+str(train_time)+"Onetloss.json",verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		#OfitHistory = OtherNetwork.fit(train_physicalXo,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,checkpointer,weight_checkpointer], class_weight={0:0.1,1:1},validation_data=(val_physicalXo,valY))
		OfitHistory = OtherNetwork.fit(train_physicalXo,trainY,batch_size=4096,epochs=50,shuffle=True,callbacks=[early_stopping,loss_checkpointer,weight_checkpointer], class_weight='auto',validation_data=(val_physicalXo,valY))
		OtherNetwork = load_model("model/"+str(train_time)+'model/OtherNetwork.h5')

	return OtherNetwork
#3
def PhysicochemicalNetwork(train_physicalXp,trainY,val_physicalXp,valY,
			#pre_train_total_path = 'model/pretrain.h5',
			train_time=None,compilemodels=None):

	physical_P_inputs = Input(shape=(train_physicalXp.shape[1],train_physicalXp.shape[2]))  #49*46=2254
	early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=5)

	if (train_time==0):
		x = core.Flatten()(physical_P_inputs)
		x = BatchNormalization()(x)

		x = Dense(512,init='glorot_normal',activation='relu')(x)
		x = BatchNormalization()(x)
		# x = Dropout(0.2)(x)

		x = Dense(256,init='glorot_normal',activation='relu')(x)
		x = BatchNormalization()(x)
		# x = Dropout(0.2)(x)

		x = Dense(128,init='glorot_normal',activation='relu')(x)
		x = BatchNormalization()(x)
		# x = Dropout(0.1)(x)

		physical_P_outputs = Dense(2,init='glorot_normal',activation='sigmoid',W_regularizer=l2(0.001))(x)

		PhysicochemicalNetwork = Model(physical_P_inputs,physical_P_outputs)

		#optimization = SGD(lr=0.01, momentum=0.9, nesterov= True)
		optimization='Nadam'
		PhysicochemicalNetwork.compile(loss='binary_crossentropy',optimizer=optimization,metrics=['accuracy'])
	else:
		PhysicochemicalNetwork = compilemodels
		#PhysicochemicalNetwork = load_model("model/"+str(train_time-1)+'model/PhysicochemicalNetwork.h5')

	if(trainY is not None):
		#checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'model/PhysicochemicalNetwork.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		weight_checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'modelweight/PhysicochemicalNetworkweight.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min',save_weights_only=True)
		loss_checkpointer = LossModelCheckpoint(model_file_path="model/"+str(train_time)+'model/PhysicochemicalNetwork.h5',monitor_file_path="model/loss/"+str(train_time)+"Pnetloss.json",verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		#PfitHistory = PhysicochemicalNetwork.fit(train_physicalXp,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,checkpointer,weight_checkpointer],class_weight={0:0.1,1:1},validation_data=(val_physicalXp,valY))
		PfitHistory = PhysicochemicalNetwork.fit(train_physicalXp,trainY,batch_size=4096,epochs=50,shuffle=True,callbacks=[early_stopping,loss_checkpointer,weight_checkpointer],class_weight='auto',validation_data=(val_physicalXp,valY))
		PhysicochemicalNetwork = load_model("model/"+str(train_time)+'model/PhysicochemicalNetwork.h5')

	return PhysicochemicalNetwork
#4
def HydrophobicityNetwork(train_physicalXh,trainY,val_physicalXh,valY,
			#pre_train_total_path = 'model/pretrain.h5',
			train_time=None,compilemodels=None):

	physical_H_inputs = Input(shape=(train_physicalXh.shape[1],train_physicalXh.shape[2]))  #49*149=7301
	early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=5)

	if (train_time==0):
		x = core.Flatten()(physical_H_inputs)
		x = BatchNormalization()(x)

		x = Dense(1024,init='glorot_normal',activation='relu')(x)
		x = BatchNormalization()(x)
		# x = Dropout(0.2)(x)

		x = Dense(512,init='glorot_normal',activation='relu')(x)
		x = BatchNormalization()(x)
		# x = Dropout(0.2)(x)

		x = Dense(256,init='glorot_normal',activation='relu')(x)
		x = BatchNormalization()(x)
		# x = Dropout(0.2)(x)

		x = Dense(128,init='glorot_normal',activation='relu')(x)
		x = BatchNormalization()(x)
		# x = Dropout(0.1)(x)

		physical_H_outputs = Dense(2,init='glorot_normal',activation='sigmoid',W_regularizer=l2(0.001))(x)

		HydrophobicityNetwork = Model(physical_H_inputs,physical_H_outputs)

		#optimization = SGD(lr=0.01, momentum=0.9, nesterov= True)
		optimization='Nadam'
		HydrophobicityNetwork.compile(loss='binary_crossentropy',optimizer=optimization,metrics=['accuracy'])
	else:
		HydrophobicityNetwork = compilemodels
		#HydrophobicityNetwork = load_model("model/"+str(train_time-1)+'model/HydrophobicityNetwork.h5')

	if(trainY is not None):
		#checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'model/HydrophobicityNetwork.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		weight_checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'modelweight/HydrophobicityNetworkweight.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min',save_weights_only=True)
		loss_checkpointer = LossModelCheckpoint(model_file_path="model/"+str(train_time)+'model/HydrophobicityNetwork.h5',monitor_file_path="model/loss/"+str(train_time)+"Hnetloss.json",verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		#HfitHistory = HydrophobicityNetwork.fit(train_physicalXh,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,checkpointer,weight_checkpointer],class_weight={0:0.1,1:1},validation_data=(val_physicalXh,valY))
		HfitHistory = HydrophobicityNetwork.fit(train_physicalXh,trainY,batch_size=4096,epochs=50,shuffle=True,callbacks=[early_stopping,loss_checkpointer,weight_checkpointer],class_weight='auto',validation_data=(val_physicalXh,valY))
		HydrophobicityNetwork = load_model("model/"+str(train_time)+'model/HydrophobicityNetwork.h5')

	return HydrophobicityNetwork
#5
def CompositionNetwork(train_physicalXc,trainY,val_physicalXc,valY,
			#pre_train_total_path = 'model/pretrain.h5',
			train_time=None,compilemodels=None):

	physical_C_inputs = Input(shape=(train_physicalXc.shape[1],train_physicalXc.shape[2]))  #49*24=1176
	early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=10)

	if (train_time==0):
		x = conv.Convolution1D(201,2,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(physical_C_inputs)
		x = Dropout(0.2)(x)
		x = Activation('relu')(x)

		x = conv.Convolution1D(151,3,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(x)
		x = Dropout(0.1)(x)
		x = Activation('relu')(x)

		x = core.Flatten()(x)
		x = BatchNormalization()(x)
		physical_C_outputs = Dense(2,init='glorot_normal',activation='sigmoid',W_regularizer=l2(0.001))(x)

		CompositionNetwork = Model(physical_C_inputs,physical_C_outputs)

		# optimization = SGD(lr=0.01, momentum=0.9, nesterov= True)
		optimization='Nadam'
		CompositionNetwork.compile(loss='binary_crossentropy',optimizer=optimization,metrics=['accuracy'])
	else:
		CompositionNetwork = compilemodels
		#CompositionNetwork = load_model("model/"+str(train_time-1)+'model/CompositionNetwork.h5')

	if(trainY is not None):
		#checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'model/CompositionNetwork.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		weight_checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'modelweight/CompositionNetworkweight.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min',save_weights_only=True)
		loss_checkpointer = LossModelCheckpoint(model_file_path="model/"+str(train_time)+'model/CompositionNetwork.h5',monitor_file_path="model/loss/"+str(train_time)+"Cnetloss.json",verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		#CfitHistory = CompositionNetwork.fit(train_physicalXc,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,checkpointer,weight_checkpointer],class_weight={0:0.1,1:1},validation_data=(val_physicalXc,valY))
		CfitHistory = CompositionNetwork.fit(train_physicalXc,trainY,batch_size=4096,epochs=50,shuffle=True,callbacks=[early_stopping,loss_checkpointer,weight_checkpointer],class_weight='auto',validation_data=(val_physicalXc,valY))
		CompositionNetwork = load_model("model/"+str(train_time)+'model/CompositionNetwork.h5')

	return CompositionNetwork
#6
def BetapropensityNetwork(train_physicalXb,trainY,val_physicalXb,valY,
			#pre_train_total_path = 'model/pretrain.h5',
			train_time=None,compilemodels=None):

	physical_B_inputs = Input(shape=(train_physicalXb.shape[1],train_physicalXb.shape[2]))  #49*37=1813
	early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=10)

	if (train_time==0):
		x = conv.Conv1D(201,2,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(physical_B_inputs)
		x = Dropout(0.2)(x)
		x = Activation('relu')(x)

		x = conv.Conv1D(151,3,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(x)
		x = Dropout(0.2)(x)
		x = Activation('relu')(x)

		x = conv.Conv1D(101,5,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(x)
		x = Dropout(0.1)(x)
		x = Activation('relu')(x)

		x = core.Flatten()(x)
		x = BatchNormalization()(x)
		physical_B_outputs = Dense(2,init='glorot_normal',activation='sigmoid',W_regularizer=l2(0.001))(x)

		BetapropensityNetwork = Model(physical_B_inputs,physical_B_outputs)

		# optimization = SGD(lr=0.01, momentum=0.9, nesterov= True)
		optimization='Nadam'
		BetapropensityNetwork.compile(loss='binary_crossentropy',optimizer=optimization,metrics=['accuracy'])
	else:
		BetapropensityNetwork = compilemodels
		#BetapropensityNetwork = load_model("model/"+str(train_time-1)+'model/BetapropensityNetwork.h5')

	if(trainY is not None):
		#checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'model/BetapropensityNetwork.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		weight_checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'modelweight/BetapropensityNetworkweight.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min',save_weights_only=True)
		loss_checkpointer = LossModelCheckpoint(model_file_path="model/"+str(train_time)+'model/BetapropensityNetwork.h5',monitor_file_path="model/loss/"+str(train_time)+"Bnetloss.json",verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		#BfitHistory = BetapropensityNetwork.fit(train_physicalXb,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,checkpointer,weight_checkpointer],class_weight={0:0.1,1:1},validation_data=(val_physicalXb,valY))
		BfitHistory = BetapropensityNetwork.fit(train_physicalXb,trainY,batch_size=4096,epochs=50,shuffle=True,callbacks=[early_stopping,loss_checkpointer,weight_checkpointer],class_weight='auto',validation_data=(val_physicalXb,valY))
		BetapropensityNetwork = load_model("model/"+str(train_time)+'model/BetapropensityNetwork.h5')

	return BetapropensityNetwork
#7
def AlphaturnpropensityNetwork(train_physicalXa,trainY,val_physicalXa,valY,
			#pre_train_total_path = 'model/pretrain.h5',
			train_time=None,compilemodels=None):

	physical_A_inputs = Input(shape=(train_physicalXa.shape[1],train_physicalXa.shape[2]))  #49*118=5782
	early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=10)

	if (train_time==0):
		x = conv.Conv1D(201,2,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(physical_A_inputs)
		x = Dropout(0.2)(x)
		x = Activation('relu')(x)

		x = conv.Conv1D(151,3,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(x)
		x = Dropout(0.2)(x)
		x = Activation('relu')(x)

		x = conv.Conv1D(101,5,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(x)
		x = Dropout(0.2)(x)
		x = Activation('relu')(x)

		x = conv.Conv1D(51,7,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(x)
		x = Dropout(0.1)(x)
		x = Activation('relu')(x)

		x = core.Flatten()(x)
		x = BatchNormalization()(x)

		physical_A_outputs = Dense(2,init='glorot_normal',activation='sigmoid',W_regularizer=l2(0.001))(x)

		AlphaturnpropensityNetwork = Model(physical_A_inputs,physical_A_outputs)

		#optimization = SGD(lr=0.01, momentum=0.9, nesterov= True)
		optimization='Nadam'
		AlphaturnpropensityNetwork.compile(loss='binary_crossentropy',optimizer=optimization,metrics=['accuracy'])
	else:
		AlphaturnpropensityNetwork = compilemodels
		#AlphaturnpropensityNetwork = load_model("model/"+str(train_time-1)+'model/AlphaturnpropensityNetwork.h5')

	if(trainY is not None):
		#checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'model/AlphaturnpropensityNetwork.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		weight_checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'modelweight/AlphaturnpropensityNetworkweight.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min',save_weights_only=True)
		loss_checkpointer = LossModelCheckpoint(model_file_path="model/"+str(train_time)+'model/AlphaturnpropensityNetwork.h5',monitor_file_path="model/loss/"+str(train_time)+"Anetloss.json",verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		#AfitHistory = AlphaturnpropensityNetwork.fit(train_physicalXa,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,checkpointer,weight_checkpointer],class_weight={0:0.1,1:1},validation_data=(val_physicalXa,valY))
		AfitHistory = AlphaturnpropensityNetwork.fit(train_physicalXa,trainY,batch_size=4096,epochs=50,shuffle=True,callbacks=[early_stopping,loss_checkpointer,weight_checkpointer],class_weight='auto',validation_data=(val_physicalXa,valY))
		AlphaturnpropensityNetwork = load_model("model/"+str(train_time)+'model/AlphaturnpropensityNetwork.h5')
  	
	return AlphaturnpropensityNetwork
