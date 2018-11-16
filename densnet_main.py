#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：albert time:2018/10/22
import os
import pickle
import random
import keras
import pandas as pd
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from nltk.probability import FreqDist
from sklearn.utils import shuffle
from tqdm import tqdm
import  numpy as np

np.random.seed(20181023)


def attention_3d_block(dim, inputs, name, SINGLE_ATTENTION_VECTOR=True):
    """
    attention frame model
    :param dim: dim_length
    :param inputs: model input
    :param name: model name
    :param SINGLE_ATTENTION_VECTOR:
    :return: attention model output
    """
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


def model_attention_applied_after_lstm(inputs, dim,gru_dim, name):
    """
    Create video_model
    :param inputs: video_features input shape
    :param dim: dim length
    :param gru_dim :
    :param name: model name
    :return: video model
    """
    vision_model = BatchNormalization(beta_regularizer=l2(0.005))(inputs)
    # lstm_out = GRU(256, return_sequences=True,kernel_regularizer=L2_regularizer, bias_regularizer=L2_regularizer)(vision_model)
    attention_mul = attention_3d_block(dim, vision_model, name)
    attention_mul = GRU(gru_laterm)(attention_mul)
    attention_mul = Dropout(0.5)(attention_mul)

    return attention_mul



def encoded_video_question_create(video_question_input):
    """
    Create question model
    :param video_question_input: input shape
    :return: question model
    """
    embedded_question = Embedding(input_dim=2000, output_dim=120, input_length=20)(video_question_input)
    encoded_question = Bidirectional(GRU(256,return_sequences=True), name='bilstm1')(embedded_question)
    encoded_question = Bidirectional(GRU(512), name='bilstm2')(encoded_question)
    encoded_question = Dropout(0.25)(encoded_question)
    question_encoder = Model(inputs=video_question_input, outputs=encoded_question)

    encoded_video_question = question_encoder(video_question_input)

    return encoded_video_question


# Aggregative model
def vqa_model_create(class_number):
    """
    Create VQA model
    :param class_number: class number
    :return: VQA model
    """
    video_c3d_input = Input(shape=(20, 4096), dtype=np.float32)
    video_xcep_input = Input(shape=(20, 1920), dtype=np.float32)
    video_question_input = Input(shape=(20,), dtype='int32')

    encoded_video_c3d = model_attention_applied_after_lstm(video_c3d_input, 20,512, '1')
    encoded_video_xcep = model_attention_applied_after_lstm(video_xcep_input, 20,1024 ,'2')

    encoded_video_question = encoded_video_question_create(video_question_input)
    merged = keras.layers.concatenate([
        encoded_video_c3d,
        encoded_video_xcep, encoded_video_question])

    merged = BatchNormalization()(merged)
    merged = Dense(1024)(merged)
    merged = Dropout(0.5)(merged)
    output = Dense(class_number, activation='softmax', kernel_regularizer=L2_regularizer)(merged)

    vqa_model = Model(inputs=[
        video_c3d_input,
        video_xcep_input, video_question_input], outputs=output)

    # compile model
    vqa_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    vqa_model.summary()
    return vqa_model



def txt_to_df(path):
    """
    Loading txt into dataframe including questions and answers
    :param path: question answer path
    :return:
        df: Dataframe of quesiton an answer
        file_name: video's index
    """
    if len(path) != 1:
        df = pd.read_csv(path[0], header=None, dtype=object) \
            .append(pd.read_csv(path[1], header=None, dtype=object)) \
            .append(pd.read_csv(path[2], header=None, dtype=object))\
            .append(pd.read_csv(path[3], header=None, dtype=object))
        df = shuffle(df)
    else:
        df = pd.read_csv(path[0], header=None, dtype=object)
    # df = df[:10]
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

def answer_to_input(df_a):
    """
    Transforming answers into input formats
    :param df_a: answer of  select
    :return:
        answer_input: model's answer input
        ans_list : answer of select
        len(ans_list): class number
    """
    count_a = FreqDist(df_a)
    ans_list = list(filter(lambda x: count_a[x] >= 2, count_a))
    answer_input = pd.get_dummies(df_a)[ans_list].values
    return answer_input, ans_list, len(ans_list)



def question_to_input(df_q,maxlen=20):
    """
    Transforming question into input formats
    :param df_q: quesiton
    :param maxlen: amxlen of input length of quesiton
    :return: question_input
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df_q)
    encoded = tokenizer.texts_to_sequences(df_q)
    question_input = sequence.pad_sequences(encoded, maxlen=maxlen)
    return question_input


def image_to_input(video_features, image_names, papers, copies, size):
    """
    Transforming images into input formats

    :return: video_input
    """
    x_video = np.zeros((len(image_names) * copies, papers, size), np.float32)
    for num, image_name in enumerate(tqdm(image_names, ncols=50)):
        x_img[copies * num: copies * num + copies] = video_features[image_name]
    return x_video


def read_pickle_file(file):
    """
    read file
    """
    return pickle.load(open(file, 'rb'))


if __name__ == '__main__':
    # set work space
    os.chdir(os.path.split(os.path.realpath(__file__))[0])

    # parameter
    epochs = 100
    batch_size = 128
    L2_regularizer = l2(0.005)


    # data
    video_c3d_feature_path = '../data/c3d_features.pkl'
    video_densenet_feature_path = '../data/Densenet_features.pkl'
    video_c3d_feature = read_pickle_file(video_c3d_feature_path)
    video_densenet_feature = read_pickle_file(video_densenet_feature_path)

    txt_path_train = ['../data/VQADatasetA_20180815/VQADatasetA_20180815/train.txt',
                      '../data/VQADatasetB_test_20180919/train.txt',
                      '../data/VQA_round2_DatasetA_20180927/semifinal_video_phase1/train.txt',
                      '../data/VQA_round2_DatasetB_20181025/semifinal_video_phase2/train.txt']
    txt_path_test = ['../data/VQA_round2_DatasetB_20181025/semifinal_video_phase2/test.txt']

    # create questions and answers input
    df_txt_train, file_names_train = txt_to_df(txt_path_train)
    df_txt_test, file_names_test = txt_to_df(txt_path_test)
    df_q_train, df_q_test = question_to_input(list(map(str, df_txt_train['q'])))
    df_a_train, ans_list, class_number = answer_to_input(df_txt_train['a1'])

    X_img_train_c3d = image_to_input(video_c3d_feature, file_names_train, 20, 5, 4096)
    X_img_train_desnsenet = image_to_input(video_densenet_feature, file_names_train, 20, 5, 1920)


    # select datas label in in classes
    # train_index = range(len(df_a_train))
    train_index = list(filter(lambda x: df_a_train[x].sum() == 1, range(len(df_a_train))))
    random.shuffle(train_index)
    X_img_train_c3d,X_img_train_desnsenet, df_q_train, df_a_train = X_img_train_c3d[train_index], \
                                                              X_img_train_desnsenet[train_index], \
                                        df_q_train[train_index], \
                                        df_a_train[train_index]

    # train_verify_split = 3000

    # model
    vqa_model = vqa_model_create( class_number)
    model_name = '../data/my_model.h5'

    from   sklearn.metrics.pairwise import  cosine_similarity

    modelCheckpoint = keras.callbacks.ModelCheckpoint(
        model_name,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1)


    earlyStopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1)

    # train
    vqa_model.fit([
        X_img_train_c3d,
        X_img_train_xcp, df_q_train], df_a_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.06,
        class_weight='auto',
        callbacks=[modelCheckpoint,
                   earlyStopping])

    # predict
    # best model load
    # vqa_model.load_weights(model_name)
    #
    # df_a_pre = vqa_model.predict([
    #    X_img_test_c3d,
    #     X_img_test_xcp, df_q_test], batch_size=batch_size, verbose=1)
    # df_a_pre = np.array(list(map(np.argmax, df_a_pre))).reshape(-1, 5)
    #
    # # result
    # result_name = "../submit/submit_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".txt"
    #
    # result = pd.read_csv(txt_path_test[0], header=None)
    # for index, num in enumerate([2, 6, 10, 14, 18]):
    #     result[num] = list(map(lambda x: ans_list[x], df_a_pre[:, index]))
    # result.drop([3, 4, 7, 8, 11, 12, 15, 16, 19, 20], axis=1, inplace=True)
    #
    # result.to_csv(result_name, header=None, index=None)