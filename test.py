# import tensorflow as tf
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


# from multiprocessing import Pool
# def work(x):
# 	return x+1
#
# pool = Pool(processes = 10)
#
# x = range(100000)
# result = pool.map(work,x)
#
# print(result)
#
#
# print(1111111111)
import  os
path = '../data/data1/Z123'
name = os.path.splitext(os.path.split(path)[-1])[0]
print(name)

# import pickle
# import  numpy as np
# import pandas as pd
# a = dict()
#
# a['a'] = np.arange(10)
# a['b'] = np.arange(10)
# a = pd.DataFrame(a)
# fr = open('1.txt','wb')
#
# pickle.dump(a,fr)
# fr.close()
#
# data = pickle.load(open('1.txt','rb'))
# print(data['a'])