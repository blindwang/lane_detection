# -*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2021/4/24 9:20
"""
import time, os, sys, cv2, warnings

import numpy as np

from dataloader import *
from model.model import LaneNet, compute_loss
from average_meter import *
warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cuda'
save_folder = 'seg_result'
os.makedirs(save_folder, exist_ok=True)


def gen_mask(rgb_img, seg_label):
	alpha = 0.6
	mixed_im = cv2.addWeighted(seg_label, alpha, rgb_img, 1, 0)
	return mixed_im


def get_img(dataset):
	gt_img_list = []
	with open(dataset, 'r') as file:
		for _info in file:
			info_tmp = _info.strip(' ').split()
			gt_img_list.append(info_tmp[0])
	return gt_img_list


if __name__ == '__main__':
	dataset = 'data/training_data_example'
	val_dataset_file = os.path.join(dataset, 'val.txt')
	img_lst = get_img(val_dataset_file)
	val_dataset = LaneDataSet(val_dataset_file, stage = 'val')
	val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
	model = torch.load('checkpoints/250.pth', map_location=DEVICE)
	model.eval()
	for batch_idx, (image_data, binary_label, instance_label) in enumerate(val_loader):
		image_data, binary_label, instance_label = image_data.to(DEVICE),binary_label.type(torch.FloatTensor).to(DEVICE),instance_label.to(DEVICE)
		with torch.set_grad_enabled(False):
			# 预测，并可视化
			# print(image_data.shape)  # torch.Size([1, 3, 256, 512])
			net_output = model(image_data)
			for k in net_output:
				print(net_output[k].shape)
			# print(np.unique(net_output["instance_seg_logits"].cpu().numpy()))
			break
			seg_logits = net_output["seg_logits"].cpu().numpy()[0]
			# 背景为（0~50）黄色线为（51~200），白色线为（201~255）
			result = (np.argmax(seg_logits, axis=0)*127).astype(np.uint8)       # 此处背景是0，黄色线是127，白色线是254
			result = result[:, :, np.newaxis]
			result = np.concatenate((result, result, result), axis=2).astype(np.uint8)
			rgb_img = cv2.imread(img_lst[batch_idx])
			rgb_img = cv2.resize(rgb_img, (512, 256), interpolation=cv2.INTER_NEAREST)
			# print(result.shape, rgb_img.shape)
			mixed_im = gen_mask(rgb_img, result)
			cv2.imwrite(os.path.join(save_folder, '{0:04d}.png'.format(batch_idx)), mixed_im)
			plt.figure()
			plt.imshow(result)
			plt.show()

