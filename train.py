# -*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2021/4/24 9:20
"""
import time, os, sys, cv2
from dataloader import *
from model.model import LaneNet, compute_loss
from average_meter import *
import logging, warnings
warnings.filterwarnings('ignore')

# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cpu'
os.makedirs('checkpoints', exist_ok=True)
logging.basicConfig(filename='log.txt', filemode='w', level=logging.WARNING, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d-%H:%M:%S')

if __name__ == '__main__':
	dataset = 'data/training_data_example'
	epochs = 1000
	bs = 2
	lr = 0.0005
	lr_decay = 0.5
	lr_decay_epoch = 250
	val_per_epoch = 2
	save_per_epoch = 250

	train_dataset_file = os.path.join(dataset, 'train.txt')
	val_dataset_file = os.path.join(dataset, 'val.txt')
	train_dataset = LaneDataSet(train_dataset_file, stage = 'train')
	train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
	val_dataset = LaneDataSet(val_dataset_file, stage = 'val')
	val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=True)
	model = LaneNet(DEVICE = DEVICE)
	model.to(DEVICE)

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_epoch, gamma=lr_decay)
	print(f"{epochs} epochs {len(train_dataset)} training samples\n")
	Loss = torch.nn.CrossEntropyLoss(weight=None, reduction='mean')  # weight可以对样本加权，解决样本不均衡问题

	for epoch in range(0, epochs):
		print("Epoch:{:03d}; Lr:{:.8f}".format(epoch, lr_scheduler.get_lr()[0]))
		model.train()
		total_losses, binary_losses, instance_losses, mean_ioues = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
		for batch_idx, (image_data, binary_label, instance_label) in enumerate(train_loader):
			image_data, binary_label, instance_label = image_data.to(DEVICE),binary_label.type(torch.FloatTensor).to(DEVICE),instance_label.to(DEVICE)
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				net_output = model(image_data)
				seg_logits = net_output["seg_logits"]     # ([1, 2, 256, 512])
				total_loss = Loss(seg_logits, instance_label.long())
				total_loss.backward()
				optimizer.step()
				total_losses.update(total_loss.item(), image_data.size()[0])
		lr_scheduler.step()
		print("Train---Epoch:{:03d}; Lr:{:.8f}; iter:{:03d}; total_loss: {:.4f}".format(
			epoch, lr_scheduler.get_lr()[0], batch_idx, total_losses.avg))
		logging.warning("Train---Epoch:{:03d}; Lr:{:.8f}; iter:{:03d}; total_loss: {:.4f}".format(
			epoch, lr_scheduler.get_lr()[0], batch_idx, total_losses.avg))

		if (epoch + 1)%val_per_epoch == 0:
			model.eval()
			total_losses, binary_losses, instance_losses, mean_ioues = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
			for batch_idx, (image_data, binary_label, instance_label) in enumerate(train_loader):
				image_data, binary_label, instance_label = image_data.to(DEVICE),binary_label.type(torch.FloatTensor).to(DEVICE),instance_label.to(DEVICE)
				with torch.set_grad_enabled(False):
					net_output = model(image_data)
					seg_logits = net_output["seg_logits"]
					total_loss = Loss(seg_logits, instance_label.long())
					total_losses.update(total_loss.item(), image_data.size()[0])
			print("Valid---Epoch:{:03d}; Lr:{:.8f}; iter:{:03d}; total_loss: {:.4f}".format(
				epoch, lr_scheduler.get_lr()[0], batch_idx, total_losses.avg))
			logging.warning("Valid---Epoch:{:03d}; Lr:{:.8f}; iter:{:03d}; total_loss: {:.4f}".format(
				epoch, lr_scheduler.get_lr()[0], batch_idx, total_losses.avg))

		if (epoch + 1)%save_per_epoch == 0:
			print(epoch)
			torch.save(model, 'checkpoints/{:03d}.pth'.format(epoch+1))


