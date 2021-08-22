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

# import transforms as trans
import paddle.vision.transforms as trans
from ppcls.utils import logger

import warnings
warnings.filterwarnings('ignore')

from dataset import GAMMA_sub1_dataset
from model import Model, Model2
from ResNet200_vd_model import ResNet200_vd_model

logger.init_logger()
batchsize = 4  # 4 patients per iter, i.e, 20 steps / epoch
oct_img_size = [512, 512]
image_size = 256
oct_image_size = 512
iters = 2000  # For demonstration purposes only, far from reaching convergence
val_ratio = 0.2  # 80 / 20
trainset_root = "E:\\multi-modal-resnet\\training_data\\multi-modality_images"
trainset_label = "./training_data/glaucoma_grading_training_GT.xlsx"
test_root = ""
num_workers = 0
init_lr = 1e-4
optimizer_type = "adam"

filelists = os.listdir(trainset_root)
train_filelists, val_filelists = train_test_split(filelists, test_size=val_ratio, random_state=42)
print("Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))



img_train_transforms = trans.Compose([
    trans.RandomResizedCrop(image_size, scale=(0.90, 1.1), ratio=(0.90, 1.1)),
    trans.RandomHorizontalFlip(),
    trans.RandomVerticalFlip(),
    trans.RandomRotation(30)
])

oct_train_transforms = trans.Compose([
    # trans.CenterCrop([256] + oct_img_size),
    trans.RandomHorizontalFlip(),
    # trans.RandomVerticalFlip()
])

img_val_transforms = trans.Compose([
    # trans.CropCenterSquare(),
    trans.Resize((image_size, image_size))
])

# oct_val_transforms = trans.Compose([
#     trans.CenterCrop([256] + oct_img_size)
# ])

_train = GAMMA_sub1_dataset(dataset_root=trainset_root,
                        img_transforms=img_train_transforms,
                        oct_transforms=oct_train_transforms,
                        label_file=trainset_label, mode='train')

_val = GAMMA_sub1_dataset(dataset_root=trainset_root,
                        img_transforms=img_val_transforms,
                        # oct_transforms=oct_val_transforms,
                        oct_transforms=None,
                        label_file=trainset_label, mode='val')




def train(model, iters, train_dataloader, val_dataloader, optimizer, criterion, log_interval, eval_interval):
    iter = 0
    model.train()
    avg_loss_list = []
    avg_kappa_list = []
    best_kappa = 0.
    while iter < iters:
        for data in train_dataloader:
            iter += 1
            if iter > iters:
                break
            fundus_imgs = (data[0] / 255.).astype("float32")
            oct_imgs = (data[1] / 255.).astype("float32")
            labels = data[2].astype('int64')

            logits = model(fundus_imgs, oct_imgs)
            loss = criterion(logits, labels)
            # acc = paddle.metric.accuracy(input=logits, label=labels.reshape((-1, 1)), k=1)
            for p, l in zip(logits.numpy().argmax(1), labels.numpy()):
                avg_kappa_list.append([p, l])

            loss.backward()
            optimizer.step()

            model.clear_gradients()
            avg_loss_list.append(loss.numpy()[0])

            if iter % log_interval == 0:
                avg_loss = np.array(avg_loss_list).mean()
                avg_kappa_list = np.array(avg_kappa_list)
                avg_kappa = cohen_kappa_score(avg_kappa_list[:, 0], avg_kappa_list[:, 1], weights='quadratic')
                avg_loss_list = []
                avg_kappa_list = []
                print("[TRAIN] iter={}/{} avg_loss={:.4f} avg_kappa={:.4f}".format(iter, iters, avg_loss, avg_kappa))

            if iter % eval_interval == 0:
                avg_loss, avg_kappa = val(model, val_dataloader, criterion)
                print("[EVAL] iter={}/{} avg_loss={:.4f} kappa={:.4f}".format(iter, iters, avg_loss, avg_kappa))
                if avg_kappa >= best_kappa:
                    best_kappa = avg_kappa
                    paddle.save(model.state_dict(),
                                os.path.join("best_model_{:.4f}".format(best_kappa), 'model.pdparams'))
                model.train()


def val(model, val_dataloader, criterion):
    model.eval()
    avg_loss_list = []
    cache = []
    with paddle.no_grad():
        for data in val_dataloader:
            fundus_imgs = (data[0] / 255.).astype("float32")
            oct_imgs = (data[1] / 255.).astype("float32")
            labels = data[2].astype('int64')

            logits = model(fundus_imgs, oct_imgs)
            for p, l in zip(logits.numpy().argmax(1), labels.numpy()):
                cache.append([p, l])

            loss = criterion(logits, labels)
            # acc = paddle.metric.accuracy(input=logits, label=labels.reshape((-1, 1)), k=1)
            avg_loss_list.append(loss.numpy()[0])
    cache = np.array(cache)
    kappa = cohen_kappa_score(cache[:, 0], cache[:, 1], weights='quadratic')
    avg_loss = np.array(avg_loss_list).mean()

    return avg_loss, kappa

img_train_transforms = trans.Compose([
    trans.RandomResizedCrop(
        image_size, scale=(0.90, 1.1), ratio=(0.90, 1.1)),
    trans.RandomHorizontalFlip(),
    trans.RandomVerticalFlip(),
    trans.RandomRotation(30)
])

oct_train_transforms = trans.Compose([
    # 变成[256, 512, 512]的状态
    trans.Resize((oct_image_size, oct_image_size)),
    # trans.CenterCrop([256] + oct_img_size),
    trans.RandomHorizontalFlip(),
    # trans.RandomVerticalFlip()
])

img_val_transforms = trans.Compose([
    # trans.CropCenterSquare(),
    trans.Resize((image_size, image_size))
])

oct_val_transforms = trans.Compose([
    trans.Resize((oct_image_size, oct_image_size)),
])

train_dataset = GAMMA_sub1_dataset(dataset_root=trainset_root,
                        img_transforms=img_train_transforms,
                        oct_transforms=oct_train_transforms,
                        filelists=train_filelists,
                        label_file=trainset_label)

val_dataset = GAMMA_sub1_dataset(dataset_root=trainset_root,
                        img_transforms=img_val_transforms,
                        oct_transforms=oct_val_transforms,
                        filelists=val_filelists,
                        label_file=trainset_label)


train_loader = paddle.io.DataLoader(
    train_dataset,
    batch_sampler=paddle.io.DistributedBatchSampler(train_dataset, batch_size=batchsize, shuffle=True, drop_last=False),
    num_workers=num_workers,
    return_list=True,
    use_shared_memory=False
)

val_loader = paddle.io.DataLoader(
    val_dataset,
    batch_sampler=paddle.io.DistributedBatchSampler(val_dataset, batch_size=batchsize, shuffle=True, drop_last=False),
    num_workers=num_workers,
    return_list=True,
    use_shared_memory=False
)

model = ResNet200_vd_model()

if optimizer_type == "adam":
    scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[100, 1000, 1500],
                                                   values=[init_lr, init_lr * 0.1, init_lr * 0.01, init_lr * 0.001],
                                                   verbose=True)
    optimizer = paddle.optimizer.Adam(scheduler, parameters=model.parameters())

criterion = nn.CrossEntropyLoss()

train(model, iters, train_loader, val_loader, optimizer, criterion, log_interval=10, eval_interval=100)
