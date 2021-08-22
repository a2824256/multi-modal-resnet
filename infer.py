import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models import resnet34
from model import Model
import paddle.vision.transforms as trans
from  dataset import GAMMA_sub1_dataset
import warnings
warnings.filterwarnings('ignore')

batchsize = 1 # 4 patients per iter, i.e, 20 steps / epoch
oct_img_size = [512, 512]
image_size = 256
iters = 1000 # For demonstration purposes only, far from reaching convergence
val_ratio = 0.2 # 80 / 20
trainset_root = ""
test_root = "E:\\multi-modal-resnet\\multi-modality_images"
num_workers = 0
init_lr = 1e-4
optimizer_type = "adam"

best_model_path = "./best_model_0.8649/model.pdparams"
model = Model()
para_state_dict = paddle.load(best_model_path)
model.set_state_dict(para_state_dict)
model.eval()



img_test_transforms = trans.Compose([
    # trans.CropCenterSquare(),
    trans.Resize((image_size, image_size))
])

oct_test_transforms = trans.Compose([
    # trans.CenterCrop([256] + oct_img_size)
])

test_dataset = GAMMA_sub1_dataset(dataset_root=test_root,
                        img_transforms=img_test_transforms,
                        oct_transforms=None,
                        mode='test')

cache = []
for fundus_img, oct_img, idx in test_dataset:
    fundus_img = fundus_img[np.newaxis, ...]
    oct_img = oct_img[np.newaxis, ...]

    fundus_img = paddle.to_tensor((fundus_img / 255.).astype("float32"))
    oct_img = paddle.to_tensor((oct_img / 255.).astype("float32"))

    logits = model(fundus_img, oct_img)
    cache.append([idx, logits.numpy().argmax(1)])
    print(idx)

submission_result = pd.DataFrame(cache, columns=['data', 'dense_pred'])
submission_result['non'] = submission_result['dense_pred'].apply(lambda x: int(x[0] == 0))
submission_result['early'] = submission_result['dense_pred'].apply(lambda x: int(x[0] == 1))
submission_result['mid_advanced'] = submission_result['dense_pred'].apply(lambda x: int(x[0] == 2))
submission_result[['data', 'non', 'early', 'mid_advanced']].to_csv("./submission_sub1.csv", index=False)

