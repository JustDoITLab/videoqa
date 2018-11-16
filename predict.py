# -*- encoding:utf-8 -*-
from keras.models import load_model
import numpy as np
import pandas as pd
import datetime
import os


def predict(img_size, papers, copies, epochs, batch_size):
	# load data
	X_img_test = np.loadtxt('../data/X_img_test.txt', delimiter=',').reshape([-1, papers, img_size, img_size, 3])
	df_q_test = np.loadtxt('../data/df_q_test.txt', delimiter=',')
	ans_df = pd.read_table('../data/ans_list.txt', header=None)
	ans_list = list(ans_df[0].T)
	txt_path_test = '../data/VQA_round2_DatasetA_20180927/semifinal_video_phase1/test.txt'

	# model load
	model_name = '../data/my_model_' + str(img_size) + '_' + str(papers) + '.h5'
	vqa_model = load_model(model_name)

	# predict
	df_a_pre = vqa_model.predict([X_img_test, df_q_test], batch_size=batch_size, verbose=1)
	df_a_pre = np.array(list(map(np.argmax, df_a_pre))).reshape(-1, 5)

	# result
	result_name = "../submit/submit_"+ datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".txt"

	result = pd.read_csv(txt_path_test, header=None)
	for index, num in enumerate([2, 6, 10, 14, 18]):
		result[num] = list(map(lambda x: ans_list[x], df_a_pre[:, index]))
	result.drop([3, 4, 7, 8, 11, 12, 15, 16, 19, 20], axis=1, inplace=True)

	result.to_csv(result_name, header=None, index=None)

if __name__ == '__main__':
	os.chdir(os.path.split(os.path.realpath(__file__))[0])

	# parameter
	img_size = 40
	papers = 20
	copies = 5
	epochs = 50
	batch_size = 128

	#decompression()
	#model_run(img_size, papers, copies, epochs, batch_size)
	predict(img_size, papers, copies, epochs, batch_size)