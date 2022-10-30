# coding: utf-8
"""
LaneNet model
https://arxiv.org/pdf/1807.01726.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.set_printoptions(suppress=True, precision=4)

try:
	from model.loss import DiscriminativeLoss
	from model.encoders import VGGEncoder
	from model.decoders import FCNDecoder
except:
	from loss import DiscriminativeLoss
	from encoders import *
	from decoders import *

# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class LaneNet(nn.Module):
	def __init__(self, arch="VGG", DEVICE='cpu'):
		super(LaneNet, self).__init__()
		# no of instances for segmentation
		self.DEVICE = DEVICE
		self.no_of_instances = 5
		encode_num_blocks = 5
		in_channels = [3, 64, 128, 256, 512]
		out_channels = in_channels[1:] + [512]
		self._arch = arch
		if self._arch == 'VGG':
			self._encoder = VGGEncoder(encode_num_blocks, in_channels, out_channels)
			self._encoder.to(self.DEVICE)

			decode_layers = ["pool5", "pool4", "pool3"]
			decode_channels = out_channels[:-len(decode_layers) - 1:-1]
			decode_last_stride = 8
			self._decoder = FCNDecoder(decode_layers, decode_channels, decode_last_stride, DEVICE=self.DEVICE)
			self._decoder.to('cpu')
		elif self._arch == 'ESPNet':
			raise NotImplementedError
		elif self._arch == 'ENNet':
			raise NotImplementedError

		self._pix_layer = nn.Conv2d(in_channels=64, out_channels=self.no_of_instances, kernel_size=1, bias=False).to(
			self.DEVICE)
		self.relu = nn.ReLU().to(self.DEVICE)

	def forward(self, input_tensor):
		encode_ret = self._encoder(input_tensor)
		decode_ret = self._decoder(encode_ret)
		decode_logits = decode_ret['logits']
		decode_deconv = decode_ret['deconv']        # BS*64*H*W
		decode_logits = decode_logits.to('cpu')
		binary_seg_ret = torch.argmax(F.softmax(decode_logits, dim=1), dim=1, keepdim=True)
		pix_embedding = self.relu(self._pix_layer(decode_deconv))
		return {
			'instance_seg_logits': pix_embedding,
			'binary_seg_pred': binary_seg_ret,
			'seg_logits': decode_logits
		}


def compute_loss(net_output, binary_label, instance_label):
	k_binary = 0.7
	k_instance = 0.3
	k_dist = 1.0
	ce_loss_fn = nn.CrossEntropyLoss()
	binary_seg_logits = net_output["binary_seg_logits"]     # ([1, 2, 256, 512])
	binary_loss = ce_loss_fn(binary_seg_logits, binary_label.type(torch.int64))
	pix_embedding = net_output["instance_seg_logits"]       # ([1, 5, 256, 512])
	ds_loss_fn = DiscriminativeLoss(0.5, 1.5, 1.0, 1.0, 0.001)
	var_loss, dist_loss, reg_loss = ds_loss_fn(pix_embedding, instance_label)
	binary_loss = binary_loss * k_binary
	instance_loss = var_loss * k_instance
	dist_loss = dist_loss * k_dist
	total_loss = binary_loss# + instance_loss + dist_loss
	out = net_output["binary_seg_pred"]                     # ([1, 1, 256, 512])
	iou = 0
	batch_size = out.size()[0]
	for i in range(batch_size):
		PR = out[i].squeeze(0).nonzero().size()[0]
		GT = binary_label[i].nonzero().size()[0]
		TP = (out[i].squeeze(0).type(torch.float) * binary_label[i]).nonzero().size()[0]
		union = PR + GT - TP
		iou += TP / union
	iou = iou / batch_size
	return total_loss, binary_loss, instance_loss, out, iou

