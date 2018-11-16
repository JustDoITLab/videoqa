from random import shuffle
import numpy as np
import glob
import tensorflow as tf
import cv2
import sys
import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import image,sequence
from nltk.probability import FreqDist
from skimage import io

def txt_to_df(path):
    df = pd.read_csv(path, header=None)
    df.columns = ['name'] + [str(j)+str(i) for i in ['1','2','3','4','5'] for j in ['q','a1','a2','a3']]
    file_names = df['name']
    df = pd.wide_to_long(df, stubnames=['q','a1','a2','a3'],i='name',j='qa')
    df['index'] = list(map(lambda x :x[0],df.index))
    df['qa'] = list(map(lambda x :x[1],df.index))
    df['index'] = df['index'].astype('category')
    df['index'].cat.set_categories(file_names,inplace = True)
    df.sort_values(['index','qa'],ascending = True,inplace = True)
    return df,file_names

def answer_to_input(df_a):
    # ans_list = sorted(map(lambda word : word[0],FreqDist(df_a['a1'].append(df_a['a2']).append(df_a['a3'])).most_common(1000)))
    # ans_list = sorted(map(lambda word : word[0],FreqDist(df_a['a1']).most_common(1000)))
    ans_list = sorted(map(lambda word : word,FreqDist(df_a['a1'])))
    # pd.DataFrame(ans_list).to_csv('temp.csv',header=None,index=None)
    
    # df_a[['a1','a2','a3']] = df_a[['a1','a2','a3']].applymap(lambda x: x if x in ans_list else '0')
    # df_a['lable'] = df_a['a1']+','+df_a['a2']+','+df_a['a3']    
    
    # answer_input = df_a['a1'].str.get_dummies(sep = ',')[ans_list].values
    answer_input = df_a['a1'].apply(lambda x : ans_list.index(x))
    # print(list(answer_input))
    return np.array(answer_input)
    
def question_to_input(df_q1,df_q2):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df_q1 + df_q2)
    encoded_1 = tokenizer.texts_to_sequences(df_q1)
    encoded_2 = tokenizer.texts_to_sequences(df_q2)
    question_input_train = sequence.pad_sequences(encoded_1, maxlen=15)
    question_input_test = sequence.pad_sequences(encoded_2, maxlen=15)

    return question_input_train,question_input_test
def load_image(addr):  # A function to Load image
    result = np.zeros((3,40,40,3))
    for index,name in enumerate(os.listdir(addr)):
        img = io.imread(addr+'/'+name)
        img = cv2.resize(img, (40, 40), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 这里/255是为了将像素值归一化到[0，1]
        img = img / 255
        result[index] =  img.astype(np.float32)
    return result

# 将数据转化成对应的属性
def _int64_feature(value):  
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
 
 
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
 
 
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

 
train_txt_path = './VQADatasetA_20180815/train.txt'
test_txt_path = './VQADatasetA_20180815/test.txt'
df_txt_train,file_names_train = txt_to_df(train_txt_path)
df_txt_test,file_names_test = txt_to_df(test_txt_path)
df_train_q,df_test_q = question_to_input(list(map(str,df_txt_train['q'])),list(map(str,df_txt_test['q'])))

a1 = answer_to_input(df_txt_train)

# 因为我装的是CPU版本的，运行起来会有'warning'，解决方法入下，眼不见为净~
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
image_path = './image_train_72_3'
 
train_addrs = os.listdir(image_path)



 
# 下面这段就开始把数据写入TFRecods文件
 
train_filename = './train1.tfrecords'  # 输出文件地址
 
# 创建一个writer来写 TFRecords 文件
writer = tf.python_io.TFRecordWriter(train_filename)
 
for i in range(len(train_addrs)):
    # 这是写入操作可视化处理

    # 加载图片
    for j in range(5):
        img = load_image(image_path+'/'+list(file_names_train)[i])
     
        question = df_train_q[i+j]
        answer1 = a1[i+j]
     
        # 创建一个属性（feature）
        feature = {'train/answer': _int64_feature(answer1),
                    'train/question': _bytes_feature(tf.compat.as_bytes(question.tostring())),
                   'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
     
        # 创建一个 example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
     
        # 将上面的example protocol buffer写入文件
        writer.write(example.SerializeToString())
 
writer.close()
sys.stdout.flush()
