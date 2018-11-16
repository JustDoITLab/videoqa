"""Preprocess the data for model."""
import os
import numpy as np
from PIL import Image
import skvideo.io
import tensorflow as tf
from keras.applications import DenseNet201
from keras.applications.vgg16 import preprocess_input
from keras.layers import GlobalAveragePooling2D

from .c3d import c3d


class DenseNet_extractor(object):
    """Select uniformly distributed frames and extract its VGG feature."""

    def __init__(self):
        """Load  DenseNet201 model.
        """
        self.base_model = DenseNet201(weights='imagenet', include_top=False)
        self.DenseNet201_model = GlobalAveragePooling2D()(base_model.output)
        self.DenseNet201_model = Model(self.base_model.input, DenseNet201_model)


    def extract(self, key_image_path):
        """Get DenseNet201  feature for video key images.

        Args:
            key_image_path: key_image_path of video.
        Returns:
            feature: [batch_size, 1920]
        """
        name = os.path.splitext(os.path.split(path)[-1])[0]
        frames_index = list(map(lambda x : x.split('.')[0],os.listdir(key_image_path)))
        frames_index.sort()
        frames_paths = list(map(lambda x: '{}/{}.jpg'.format(key_image_path,x),frames_index))
        images_input = [preprocess_input(image.img_to_array(image.load_img(im, target_size=(224, 224))), mode='torch')
                        for im in frames_paths]
        feature = self.DenseNet201_model.predict(images_input)
        return name,feature


class VideoC3DExtractor(object):
    """Select uniformly distributed clips and extract its C3D feature."""

    def __init__(self, clip_num, sess,steps = 1):
        """Load C3D model."""
        self.clip_num = clip_num
        self.steps = steps
        self.inputs = tf.placeholder(
            tf.float32, [self.clip_num, 16, 112, 112, 3])
        _, self.c3d_features = c3d(self.inputs, 1, clip_num)
        saver = tf.train.Saver()
        saver.restore(sess, '../data/sports1m_finetuning_ucf101.model')
        self.mean = np.load( '../data/crop_mean.npy')
        self.sess = sess

    def _select_clips(self, path):
        """Select self.batch_size clips for video. Each clip has 16 frames.

        Args:
            path: Path of video.
        Returns:
            clips: list of clips.
        """
        clips = list()
        # video_info = skvideo.io.ffprobe(path)
        video_data = skvideo.io.vread(path)
        total_frames = video_data.shape[0]

        stpes = self.steps
        for i in np.linspace(0, total_frames, self.clip_num + 2)[1:self.clip_num + 1]:
            # Select center frame first, then include surrounding frames
            clip_start = int(i) - 8*stpes 
            clip_end = int(i) + 8*stpes
            if clip_start < 0:
                clip_end = clip_end - clip_start
                clip_start = 0
            if clip_end > total_frames:
                clip_start = clip_start - (clip_end - total_frames)
                clip_end = total_frames
            clip = video_data[clip_start:clip_end:stpes]
            new_clip = []
            for j in range(16):
                frame_data = clip[j]
                img = Image.fromarray(frame_data)
                img = img.resize((112, 112), Image.BILINEAR)
                frame_data = np.array(img) * 1.0
                frame_data -= self.mean[j]
                new_clip.append(frame_data)
            clips.append(new_clip)
        return clips

    def extract(self, path):
        """Get 4096-dim activation as feature for video.

        Args:
            path: Path of video.
        Returns:
            feature: [self.batch_size, 4096]
        """
        clips = self._select_clips(path)
        feature = self.sess.run(
            self.c3d_features, feed_dict={self.inputs: clips})
        return feature

