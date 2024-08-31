import torch.utils.data as data
import numpy as np
import random
from utils import process_feat
import torch
import scipy.ndimage as ndi
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.FloatTensor')


class Dataset(data.Dataset):
    def __init__(self, args, transform=None, test_mode=False, unify=True):
        self.args = args
        self.modality = args.modality
        self.dataset = args.dataset
        self.unify = unify
        if self.dataset == 'shanghai':
            if test_mode:
                self.rgb_list_file = 'list/shanghai-i3d-test-10crop.list'
            else:
                self.rgb_list_file = 'list/shanghai-i3d-train-10crop.list'
        else:
            if test_mode:
                self.rgb_list_file = 'list/ucf-i3d-test.list'
            else:
                self.rgb_list_file = 'list/ucf-i3d.list'

        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None

    def _change_list(self):
        clip_list_q = []
        clip_list_k = []
        clip_list_test = []
        for i in range(len(self.list)):
            features = np.load(self.list[i].strip('\n'), allow_pickle=True)  # (59,10,2048)
            features = np.array(features, dtype=np.float32)

            # 每个视频只取32个视频段
            if self.unify:
                features = features.transpose(1, 0, 2)  # [10, T, F]
                divided_features = []
                for feature in features:
                    feature = process_feat(feature, self.args.num_clip)  # divide a video into 32 segments
                    divided_features.append(feature)
                features = np.array(divided_features, dtype=np.float32)  # (10,32,2048)

            # 将一个clip作为一个单位保存在np数组中
            if self.test_mode:
                features = np.mean(features, axis=1, keepdims=True)  # (28,1,2048)
                for j in range(len(features)):
                    clip_list_test.append(features[j])
            else:
                features = features.reshape((2, -1, self.args.num_clip, 2048))  # (2,5,32,2048)
                features_1 = features[0]  # (5,32,2048)
                features_2 = features[1]  # (5,32,2048)
                features_1 = np.mean(features_1, axis=0, keepdims=True)  # (1,32,2048)
                features_2 = np.mean(features_2, axis=0, keepdims=True)  # (1,32,2048)
                features_1 = features_1.transpose(1, 0, 2)  # [32, 1, 2048]
                features_2 = features_2.transpose(1, 0, 2)  # [32, 1, 2048]

                for j in range(len(features_1)):
                    clip_list_q.append(features_1[j])  # (5600,1,2048)SH  (25600,1,2048)UCF   (160000,1,2048)UCF
                    clip_list_k.append(features_2[j])  # (5600,1,2048)SH  (25600,1,2048)UCF   (160000,1,2048)UCF

        if self.test_mode:
            # self.list = np.array(clip_list, dtype=np.float32)
            self.list = clip_list_test  # UCF:69634
        else:
            self.list = clip_list_q  # UCF:160000
            self.list_k = clip_list_k  # UCF:160000

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.dataset == 'shanghai':
                self.list = self.list[63:]

                self._change_list()
                print('normal list for shanghai tech')
                # print(self.list)

            elif self.dataset == 'ucf':
                self.list = self.list[810:]

                self._change_list()
                print('normal list for ucf')
                # print(self.list)
        else:
            if self.dataset == 'shanghai':
                self._change_list()
                print('test list for shanghai tech')
                # print(self.list)

            elif self.dataset == 'ucf':
                self._change_list()
                print('test list for ucf')
                # print(self.list)

    # 高斯模糊
    def gaussian_filter(self, data, sigma=2):  # sigma是高斯滤波的标准差
        data = np.array(data)
        data_filtered = ndi.gaussian_filter1d(data, sigma=sigma)
        return data_filtered

    def __getitem__(self, index):

        if self.test_mode:
            features = self.list[index]  # (1,2048)
            features = np.array(features, dtype=np.float32)
            return features  # (1,2048)
        else:
            # Q部分
            features_q = self.list[index]  # (1,2048)
            features_q = np.array(features_q, dtype=np.float32)

            # K部分
            features_k = self.list_k[index]  # (1,2048)
            features_k = self.gaussian_filter(features_k, sigma=2)  # 高斯模糊增强
            features_k = np.array(features_k, dtype=np.float32)


            if self.tranform is not None:
                features_q = self.tranform(features_q)
                features_k = self.tranform(features_k)

            return features_q, features_k  # (1,2048)

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
