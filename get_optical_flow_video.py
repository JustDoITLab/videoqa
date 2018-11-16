import cv2
import numpy as np
import os 
from  tqdm import tqdm
import glob
def optical_flow(video_path,save_path):
    name = os.path.splitext(os.path.split(video_path)[-1])[0]
    
    cap = cv2.VideoCapture(video_path)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    creat_out = True
    frames_num = cap.get(7)
    for i in range(0, int(frames_num) - 1, 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame2 = cap.read()
        try:
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        except:
            break

        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        if creat_out:
            out = cv2.VideoWriter(save_path + str(name)+ '.mp4',
                cv2.VideoWriter_fourcc('M','P','4','V'), 40, (rgb.shape[1],rgb.shape[0]))
        creat_out = False
        out.write(rgb)

        prvs = next
    out.release()
    cap.release()
    cv2.destroyAllWindows()


def save_key_images(path, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for video_path in tqdm([video for pa in path for video in glob.glob(pa + '*.mp4')][::-1],ncols = 70):
        optical_flow(video_path,save_path)
    # return list(map(lambda video: extract_c3d(video, save_path),
    #               [video for pa in path for video in glob.glob(pa + '*.mp4')]))

if __name__ == '__main__':
    os.chdir(os.path.split(os.path.realpath(__file__))[0])

    img_path_train = '../data/train_optical_flow_video/' 
    img_path_test = '../data/test_optical_flow_video/' 
    if not os.path.exists(img_path_train):
        path_train = ["../data/VQADatasetA_20180815/VQADatasetA_20180815/train/",
                      "../data/VQADatasetB_train_part1_20180919/train/",
                      "../data/VQADatasetB_train_part2_20180919/train/",
                      '../data/VQA_round2_DatasetA_20180927/semifinal_video_phase1/train/']
        save_key_images(path_train, img_path_train)

