from util.preprocess import DenseNet_extractor
import glob
from tqdm import tqdm


def extract_video_Densenet(path, save_path):
    """
    extract and sava densenet video's features

    :param path: video's path , one or more is ok
    :param save_path:  feature's save_path
    :return: None
    """
    video_densenet_features = dict()
    for key_iamge_path in tqdm(map(lambda video: save_key_image(video, save_path, papers, img_size), \
                                   [video for pa in path for video in glob.glob(pa + '*.mp4')]), ncols=50):
        name, feature = extractor.extract(key_iamge_path)
        video_densenet_features[name] = feature
    densenet_file = open(save_path, 'wb')
    pickle.dump(video_densenet_features, densenet_file)
    densenet_file.close()


if __name__ == '__main__':
    extractor = DenseNet_extractor()
    video_Densenet_feature_save_path = '../data/Densenet_features.pkl'
    video_iamge_key_iamge_path = '../data/image_train_key_image/'
    extract_video_Densenet(video_iamge_key_iamge_path, video_Densenet_feature_save_path)
