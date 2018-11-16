# -*- encoding:utf-8 -*-
import zipfile
import keras
from keras.layers import *
from keras.models import Model
from keras.preprocessing import  sequence
from keras.preprocessing.text import Tokenizer
from nltk.probability import FreqDist
import os
import glob
import cv2
from keras.layers.normalization import BatchNormalization
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
from keras.regularizers import l1,l2,l1_l2
import random
from functools import reduce
from sklearn.preprocessing import MultiLabelBinarizer
from keras_contrib.losses.jaccard import jaccard_distance
def decompression():
	filepath = '../data/'
	zipfiles = glob.glob(filepath+'*.zip')
	for file in zipfiles:
		savepath = file[:-4]
		if not os.path.exists(savepath):
			f = zipfile.ZipFile(file,'r')
			list(map(lambda x : f.extract(x,savepath),f.namelist()))


def save_key_image(path, save_path, papers=10, img_size=224):
	name = os.path.splitext(os.path.split(path)[-1])[0]
	print(name)
	new_path = save_path + name + '/'
	if not os.path.exists(new_path):
		os.mkdir(new_path)
	cap = cv2.VideoCapture(path)
	frames_num = cap.get(7)
	for i in np.linspace(0, frames_num - 1, num=papers, dtype=int):
		cap.set(cv2.CAP_PROP_POS_FRAMES, i)
		rval, frame = cap.read()
		frame = cv2.resize(frame, (img_size, img_size), fx=0, fy=0, interpolation=cv2.INTER_AREA)
		cv2.imwrite(new_path + str(i) + '.jpg', frame)
	cap.release()


# Saving key images from videos, default 10 frames and image size is 224
def save_key_images(path, save_path, papers=10, img_size=224):
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	return list(map(lambda video: save_key_image(video, save_path, papers, img_size),
					[video for pa in path for video in glob.glob(pa + '*.mp4')]))


# Video model
def attention_3d_block(dim, inputs, name, SINGLE_ATTENTION_VECTOR=True):
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, dim))(a)  # this line is not useful. It's just to know which dimension is what.
    a = Dense(dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction' + name)(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec' + name)(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul' + name)
    return output_attention_mul


def model_attention_applied_after_lstm(inputs, dim, name):
    # vision_model = Dropout(0.5)(inputs)
    vision_model = BatchNormalization(beta_regularizer=L2_regularizer)(inputs)
    # lstm_out = GRU(256, return_sequences=True,kernel_regularizer=L2_regularizer, bias_regularizer=L2_regularizer)(vision_model)
    attention_mul = attention_3d_block(dim, vision_model, name)
    attention_mul = Bidirectional(GRU(512),name='bilstm' + str(name))(attention_mul)
    attention_mul = Dropout(0.35)(attention_mul)
    # attention_mul = GlobalAveragePooling1D()(attention_mul)

    return attention_mul



# Question model
def encoded_video_question_create(video_question_input):
	embedded_question = Embedding(input_dim=2000, output_dim=300, input_length=15)(video_question_input)
	encoded_question = Bidirectional(LSTM(128,kernel_regularizer=L2_regularizer, bias_regularizer=L2_regularizer),name = 'bilstm')(embedded_question)
	encoded_question = Dropout(0.25)(encoded_question)
	question_encoder = Model(inputs=video_question_input, outputs=encoded_question)

	encoded_video_question = question_encoder(video_question_input)
	return encoded_video_question


# Aggregative model
def vqa_model_create(papers, img_size,class_number):
	video_input = Input(shape=(papers,1920),dtype = np.float32)
	video_question_input = Input(shape=(15,), dtype='int32')

	encoded_video =  model_attention_applied_after_lstm(video_input,papers,'1')
	encoded_video_question = encoded_video_question_create(video_question_input)
	merged = keras.layers.concatenate([encoded_video, encoded_video_question])
	merged = BatchNormalization()(merged)
	merged = Dense(512,activation='relu')(merged)
	output = Dropout(0.5)(merged)
	output = Dense(class_number, activation='sigmoid')(output)

	vqa_model = Model(inputs=[video_input,video_question_input], outputs=output)

	# # compile model
	vqa_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	vqa_model.summary()
	return vqa_model
 	 
 	 
 	# build module

# Loading txt into dataframe including questions and answers
def txt_to_df(path):
	if len(path) !=1 :
		df = pd.read_csv(path[0], header=None, dtype=object)\
			.append(pd.read_csv(path[1], header=None, dtype=object))\
			.append(pd.read_csv(path[2], header=None, dtype=object))\
			.append(pd.read_csv(path[3], header=None, dtype=object))
	else:

		df = pd.read_csv(path[0], header=None, dtype=object)
	# df = df[:100]

	df.columns = ['name'] + [str(j) + str(i) for i in ['1', '2', '3', '4', '5'] for j in ['q', 'a1', 'a2', 'a3']]

	file_names = df['name']
	df = pd.wide_to_long(df, stubnames=['q', 'a1', 'a2', 'a3'], i='name', j='qa')
	df['index'] = list(map(lambda x: x[0], df.index))
	df['qa'] = list(map(lambda x: x[1], df.index))
	df['index'] = df['index'].astype('category')
	df['index'].cat.set_categories(file_names, inplace=True)
	df.sort_values(['index', 'qa'], ascending=True, inplace=True)
	df[['a1', 'a2', 'a3']] = df[['a1', 'a2', 'a3']].applymap(lambda x: x.replace('-', ' '))

	return df, file_names

# Transforming answers into input formats
def answer_to_input(df_a):
	count_a = FreqDist(df_a['a1'].append(df_a['a2']).append(df_a['a3']))
	ans_list = list(filter(lambda x :count_a[x]>=4,count_a))
	print(len(ans_list))
	mlb = MultiLabelBinarizer()
	answer_set = list(map(lambda x:set(list(filter(lambda y:y in ans_list,x))),df_a[['a1','a2','a3']].values))
	answer_input = mlb.fit_transform(answer_set)


	return answer_input.astype(np.float32),list(mlb.classes_),len(ans_list)


# Transforming questions into input formats
def question_to_input(df_q1, df_q2):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(df_q1 + df_q2)
	encoded_1 = tokenizer.texts_to_sequences(df_q1)
	encoded_2 = tokenizer.texts_to_sequences(df_q2)
	question_input_train = sequence.pad_sequences(encoded_1, maxlen=15)
	question_input_test = sequence.pad_sequences(encoded_2, maxlen=15)

	return question_input_train, question_input_test


# Transforming images into input formats
def image_to_input(img_path, image_names, papers, img_size, copies):
	x_img = np.zeros((len(image_names) * copies, papers, 1920),np.float32)
	for num, image_name in enumerate(tqdm(image_names,ncols=50)):	
		x_img[copies * num: copies * num+ copies] = np.loadtxt(img_path+image_name + '.txt',delimiter = ',')
	return x_img


def lr_scheduler(epoch):
	if epoch in [15,30,50]:
		K.set_value(vqa_model.optimizer.lr, K.eval(vqa_model.optimizer.lr) * 0.1)
	return K.eval(vqa_model.optimizer.lr)


def balance_train(df_a, train_index, ans_list, aim):
	index_list = [[]] * len(ans_list)
	for ind in train_index:
		index_num = ans_list.index(df_a[ind])
		index_list[index_num].append(ind)
	for i in range(len(index_list)):
		index_length = len(index_list[i])
		if index_length == 0:
			continue
		if index_length < aim:
			index_list[i] = index_list[i] * (aim//index_length) + random.sample(index_list[i], aim % index_length)
		elif index_length > aim:
			index_list[i] = random.sample(index_list[i], aim)
	index_list = reduce(lambda x, y: x + y, index_list)
	#random.shuffle(index_list)
	return index_list


if __name__ == '__main__':
	#set work space
	os.chdir(os.path.split(os.path.realpath(__file__))[0])

	# parameter
	img_size = 224
	papers = 20
	copies = 5
	epochs = 50
	batch_size = 128
	L2_regularizer = l2(0.005)

	#unzip
	decompression()
	
	# data

	# img_path_train = '../data/image_train_key_image_Xception_features/'
	# img_path_test = '../data/image_test_key_image_Xception_features/'

	img_path_train_c3d = '../data/c3d_train/image_train_224_5c3dfeature/'
	img_path_test_c3d = '../data/image_test_c3dNer_feature/'
	img_path_train_xcp = '../data/data_train_key_image_Densenet_features/'
	img_path_test_xcp = '../data/data_last_test_key_image_Densenet_features/'

	txt_path_train = ['../data/VQADatasetA_20180815/VQADatasetA_20180815/train.txt',
	                  '../data/VQADatasetB_test_20180919/train.txt',
	                  '../data/VQA_round2_DatasetA_20180927/semifinal_video_phase1/train.txt',
	                  '../data/VQA_round2_DatasetB_20181025/semifinal_video_phase2/train.txt']
	txt_path_test = ['../data/VQA_round2_DatasetB_20181025/semifinal_video_phase2/test.txt']

	#create questions and answers input
	df_txt_train, file_names_train = txt_to_df(txt_path_train)
	df_txt_test, file_names_test = txt_to_df(txt_path_test)
	df_q_train, df_q_test = question_to_input(list(map(str, df_txt_train['q'])), list(map(str, df_txt_test['q'])))
	df_a_train, ans_list, class_number = answer_to_input(df_txt_train)
	
	# create images input
	X_img_train = image_to_input(img_path_train_xcp, file_names_train, papers, img_size, copies)
	X_img_test = image_to_input(img_path_test_xcp, file_names_test, papers, img_size, copies)


	# #select datas label in in classes
	train_index = list(range(len(df_a_train)))
	#list(filter(lambda x:df_a_train[x].sum() == 1,range(len(df_a_train))))
	random.shuffle(train_index)
	X_img_train_, df_q_train_, df_a_train_ = X_img_train[train_index], \
															df_q_train[train_index], \
															df_a_train[train_index]
	train_verify_split = 3000


	#balance trian data
	# aim = 50
	# train_train_index = balance_train(list(df_txt_train['a1']), train_index[:-train_verify_split], ans_list, aim)

	X_img_train_train, df_q_train_train, df_a_train_train = X_img_train_[:-train_verify_split], \
															df_q_train_[:-train_verify_split], \
															df_a_train_[:-train_verify_split]
	X_img_train_verify, df_q_train_verify, df_a_train_verify = X_img_train_[-train_verify_split:], \
															   df_q_train_[-train_verify_split:], \
															   df_a_train_[-train_verify_split:]


	# model

	vqa_model = vqa_model_create(papers, img_size,class_number)




	
	model_name = '../data/my_model_' + str(img_size) + '_' + str(papers) + '.h5'

	modelCheckpoint = keras.callbacks.ModelCheckpoint(
		model_name,
		monitor='val_loss',
		verbose=1,
		save_best_only=True,
		save_weights_only=False,
		mode='auto',
		period=1)

	learningRateScheduler = keras.callbacks.LearningRateScheduler(lr_scheduler)

	earlyStopping = keras.callbacks.EarlyStopping(
		monitor='val_loss',
		patience=10,
		verbose=1)

	#train
	# vqa_model.fit([X_img_train_train, df_q_train_train], df_a_train_train,
	# 			  epochs=epochs,
	# 			  batch_size = batch_size,
	# 			  #class_weight= 'auto',
	# 			  validation_data = [[X_img_train_verify, df_q_train_verify], df_a_train_verify],
	# 			  callbacks=[modelCheckpoint, learningRateScheduler, earlyStopping]
	# 			  )

	# predict
	# best model load
	vqa_model.load_weights(model_name)

	out = np.array(vqa_model.predict([X_img_train_train, df_q_train_train], batch_size=batch_size, verbose=1))
	from sklearn.metrics import matthews_corrcoef

	acc = []
	accuracies = []
	best_threshold = np.zeros(out.shape[1])
	for i in tqdm(range(out.shape[1]),ncols=50):
		y_prob = np.array(out[:, i])
		threshold = np.arange(y_prob.min(), y_prob.max(), (y_prob.max() - y_prob.min()) / 50)
		for j in threshold:
			#y_pred = [1 if prob >= j else 0 for prob in y_prob]
			y_pred = np.array(y_prob>j) & 1
			acc.append(matthews_corrcoef(df_a_train_train[:, i], y_pred))
		acc = np.array(acc)
		index = np.where(acc == acc.max())
		accuracies.append(acc.max())
		best_threshold[i] = threshold[index[0][0]]
		acc = []
	print(best_threshold)
	# import sys
	# print(list(sorted(df_a_pre[0]))[::-1])
	# sys.exit()
	df_a_pre = np.array(vqa_model.predict([X_img_test, df_q_test], batch_size=batch_size, verbose=1))
	y_pred = np.array(
		[[(df_a_pre[i, j]-best_threshold[j])/best_threshold[j] if df_a_pre[i, j] >= best_threshold[j] else 0 for j in range(df_a_pre.shape[1])] for i in range(len(df_a_pre))])
	df_a_pre = np.array(list(map(np.argmax, df_a_pre))).reshape(-1, 5)

	# result
	result_name = "../submit/submit_"+ datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".txt"

	result = pd.read_csv(txt_path_test[0], header=None)
	for index, num in enumerate([2, 6, 10, 14, 18]):
		result[num] = list(map(lambda x: ans_list[x], df_a_pre[:, index]))
	result.drop([3, 4, 7, 8, 11, 12, 15, 16, 19, 20], axis=1, inplace=True)

	result.to_csv(result_name, header=None, index=None)


	vqa_model.load_weights(model_name)


	df_a_pre = vqa_model.predict([X_img_train, df_q_train], batch_size=batch_size, verbose=1)
	df_a_pre = np.array(list(map(np.argmax, df_a_pre)))

	# result
	result_name = "../data/submit_"+ datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".txt"

	df_txt_train['a_pre'] = list(map(lambda x: ans_list[x],df_a_pre))


	df_txt_train.to_csv(result_name)