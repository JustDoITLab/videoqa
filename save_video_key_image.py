import cv2
import os
from skimage import feature
from sklearn.cluster import KMeans
from tqdm import tqdm
import glob
from keras.layers import *


def get_hog_feature(iamge):
    gray_image = cv2.cvtColor(iamge, cv2.COLOR_BGR2GRAY)
    hog_feature = feature.hog(gray_image, orientations=8, pixels_per_cell=(16, 16),
                              cells_per_block=(1, 1))
    return list(hog_feature.ravel())

def extract_video_hog_feature(path, img_size=224):
    features = []
    frames = []
    cap = cv2.VideoCapture(path)
    frames_num = int(cap.get(7))
    step = frames_num // 250 + 1
    for i in range(0, frames_num, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        rval, frame = cap.read()
        frame = cv2.resize(frame, (img_size, img_size), fx=0, fy=0, interpolation=cv2.INTER_AREA)
        frames.append(frame)
        features.append(get_hog_feature(frame))
    cap.release()
    return features, frames


def save_key_image(path, save_path):
    name = os.path.splitext(os.path.split(path)[-1])[0]
    features, frames = extract_video_hog_feature(path)
    model = KMeans(n_clusters=n_cluster)
    model.fit(features)
    result = model.labels_

    new_path = save_path + name + '/'
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    labels = []
    key_frames = np.zeros((n_cluster, 224, 224, 3))
    for index, label in enumerate(result):
        if label not in labels:
            cv2.imwrite(new_path + str(index) + '.jpg', frames[index])
            key_frames[len(labels)] = frames[index]
            labels.append(label)


def save_key_images(path, save_path):
    if not os.path.exists(features_path):
        os.mkdir(save_path)
    for video_path in tqdm([video for pa in path for video in glob.glob(pa + '*.mp4')][::-1], ncols=70):
        save_key_image(video_path, save_path)


if __name__ == '__main__':
    os.chdir(os.path.split(os.path.realpath(__file__))[0])

    n_cluster = 20
    video_path = [
        "../data/VQADatasetA_20180815/VQADatasetA_20180815/train/",
        "../data/VQADatasetB_train_part1_20180919/train/",
        "../data/VQADatasetB_train_part2_20180919/train/",
        '../data/VQA_round2_DatasetA_20180927/semifinal_video_phase1/train/',
        '../data/VQA_round2_DatasetB_20181025/semifinal_video_phase2/train/'
        ]
    video_iamge_key_iamge_path = '../data/image_train_key_image/'
    save_key_images(video_path, video_iamge_key_iamge_path)

