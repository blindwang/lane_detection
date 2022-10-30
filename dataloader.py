# -*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2021/4/24 10:44
"""
import os, cv2, torch, random, glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

ImgResize = (256, 512) # image size

train_transform = A.Compose([A.Resize(ImgResize[0], ImgResize[1]),
                             A.RandomBrightnessContrast(always_apply=True),
                             A.HueSaturationValue(always_apply=True),
                             A.ShiftScaleRotate(border_mode = cv2.BORDER_CONSTANT, always_apply=True),
                             A.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                             ToTensorV2()])
valid_transform = A.Compose([A.Resize(ImgResize[0], ImgResize[1]),
                             A.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                             ToTensorV2()])

class LaneDataSet(Dataset):
	def __init__(self, dataset, n_labels=10, stage=None):
		self._gt_img_list = []
		self._gt_label_binary_list = []
		self._gt_label_instance_list = []
		self.n_labels = n_labels
		if stage == 'train':
			self.transform = train_transform
		else:
			self.transform = valid_transform
		with open(dataset, 'r') as file:
			for _info in file:
				info_tmp = _info.strip(' ').split()
				self._gt_img_list.append(info_tmp[0])
				self._gt_label_binary_list.append(info_tmp[1])
				self._gt_label_instance_list.append(info_tmp[2])
		assert len(self._gt_img_list) == len(self._gt_label_binary_list) == len(self._gt_label_instance_list)

	def _split_instance_gt(self, label_instance_img):
		no_of_instances = self.n_labels
		ins = np.zeros((no_of_instances, label_instance_img.shape[0], label_instance_img.shape[1]))
		for _ch, label in enumerate(np.unique(label_instance_img)[1:]):
			x, y = np.where(label_instance_img == label)
			ins[_ch, x, y] = 1
		return ins

	def __len__(self):
		return len(self._gt_img_list)

	def __getitem__(self, idx):
		img = cv2.imread(self._gt_img_list[idx], cv2.IMREAD_COLOR)
		label_img = cv2.imread(self._gt_label_binary_list[idx], cv2.IMREAD_GRAYSCALE)
		# 背景为（0~50）黄色线为（51~200），白色线为（201~255）
		label_instance_img = cv2.imread(self._gt_label_instance_list[idx], cv2.IMREAD_GRAYSCALE)
		temp = np.zeros_like(label_instance_img)
		x, y = np.where(label_instance_img > 200)
		temp[x, y] = 2
		x, y = np.where((label_instance_img<=200)&(label_instance_img>50))
		temp[x, y] = 1
		label_instance_img = temp
		mask = cv2.merge([label_img, label_instance_img])
		trans_res = self.transform(image = img, mask = mask)
		img, mask = trans_res['image'], trans_res['mask']
		label_img, label_instance_img = mask[...,0]//255, mask[...,1]
		return img, label_img, label_instance_img

