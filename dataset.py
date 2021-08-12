import paddle
import os
import cv2
import numpy as np
import pandas as pd
import PIL.Image as Image


def crop(img, top, left, height, width):
    return img[:, top:top + height, left:left + width]


def center_crop(img, output_size):

    _, h, w = img.shape
    _, th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)

class GAMMA_sub1_dataset(paddle.io.Dataset):
    """
    getitem() output:

    	fundus_img: RGB uint8 image with shape (3, image_size, image_size)

        oct_img:    Uint8 image with shape (256, oct_img_size[0], oct_img_size[1])
    """

    def __init__(self,
                 img_transforms,
                 oct_transforms,
                 dataset_root,
                 label_file='',
                 filelists=None,
                 num_classes=3,
                 mode='train'):

        self.dataset_root = dataset_root
        self.img_transforms = img_transforms
        self.oct_transforms = oct_transforms
        self.mode = mode.lower()
        self.num_classes = num_classes

        if self.mode == 'train':
            label = {row['data']: row[1:].values
                     for _, row in pd.read_excel(label_file).iterrows()}

            self.file_list = [[f, label[int(f)]] for f in os.listdir(dataset_root)]
        elif self.mode == "test":
            self.file_list = [[f, None] for f in os.listdir(dataset_root)]

        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]

    def __getitem__(self, idx):
        real_index, label = self.file_list[idx]

        fundus_img_path = os.path.join(self.dataset_root, real_index, real_index + ".jpg")
        oct_series_list = sorted(os.listdir(os.path.join(self.dataset_root, real_index, real_index)),
                                 key=lambda x: int(x.strip("_")[0]))
        # print(fundus_img_path)
        fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1]  # BGR -> RGB
        oct_series_0 = cv2.imread(os.path.join(self.dataset_root, real_index, real_index, oct_series_list[0]),
                                  cv2.IMREAD_GRAYSCALE)
        oct_img = np.zeros((len(oct_series_list), oct_series_0.shape[0], oct_series_0.shape[1], 1), dtype="uint8")

        for k, p in enumerate(oct_series_list):
            oct_img[k] = cv2.imread(
                os.path.join(self.dataset_root, real_index, real_index, p), cv2.IMREAD_GRAYSCALE)[..., np.newaxis]

        if self.img_transforms is not None:
            fundus_img = self.img_transforms(fundus_img)
            # print(fundus_img.shape)
            # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
        oct_img = oct_img.squeeze(-1)
        oct_img = center_crop(oct_img, [256, 512, 512])
        if self.oct_transforms is not None:
            oct_img = self.oct_transforms(oct_img)

        # normlize on GPU to save CPU Memory and IO consuming.
        # fundus_img = (fundus_img / 255.).astype("float32")
        # oct_img = (oct_img / 255.).astype("float32")

        fundus_img = fundus_img.transpose(2, 0, 1)  # H, W, C -> C, H, W
        # print(fundus_img.shape)
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
        # oct_img = oct_img.squeeze(-1)  # D, H, W, 1 -> D, H, W

        if self.mode == 'test':
            return fundus_img, oct_img, real_index
        if self.mode == "train":
            label = label.argmax()
            return fundus_img, oct_img, label

    def __len__(self):
        return len(self.file_list)