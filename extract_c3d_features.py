import glob
import os
import  pickle
import tensorflow as tf
from tqdm import tqdm
from util.preprocess import VideoC3DExtractor


def extract_c3d(path):
    """Extract C3D features.

    Args :
        path : each video's path
    Return:
        name : vidoe's name
        feature : video's c3d features
    """

    name = os.path.splitext(os.path.split(path)[-1])[0]

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.visible_device_list = '0'

    with tf.Graph().as_default(), tf.Session(config=sess_config) as sess:
        extractor = VideoC3DExtractor(20, sess)
        features = extractor.extract(path)
    return name,features

def extract_video_c3d(path, save_path):
    """
    extract features an save feature

    :param path: videos path, one or more is ok
    :param save_path: save feature's path

    """
    video_c3d_features = dict()
    for video_path in tqdm([video for pa in path for video in glob.glob(pa + '*.mp4')], ncols=70):
        name,features =  extract_c3d(video_path)
        video_c3d_features[name] = features
    c3d_file = open(save_path,'wb')
    pickle.dump(video_c3d_features,c3d_file)
    c3d_file.close()


if __name__ == '__main__':
    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    video_c3d_feature_save_path = '../data/c3d_features.pkl'
    video_path = [
        "../data/VQADatasetA_20180815/VQADatasetA_20180815/train/",
        "../data/VQADatasetB_train_part1_20180919/train/",
        "../data/VQADatasetB_train_part2_20180919/train/",
        '../data/VQA_round2_DatasetA_20180927/semifinal_video_phase1/train/',
        '../data/VQA_round2_DatasetB_20181025/semifinal_video_phase2/train/'
        ]
    extract_video_c3d(video_path, video_c3d_feature_save_path)
