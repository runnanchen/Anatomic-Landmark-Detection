from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
# (16, 11) for 600; (10, 8) for 300; (37, 30) for 1000
import lossFunction
import utils

class dilationInceptionModule(nn.Module):
	def __init__(self , inplanes, planes):
		super(dilationInceptionModule, self).__init__()

		fnum = int(planes / 4)
		self.temConv1 = nn.Sequential(
				nn.Conv2d(inplanes, fnum, kernel_size=(1, 1), stride=1, padding = 0),
				nn.BatchNorm2d(fnum, track_running_stats=False),
				nn.ReLU(inplace = True),
		)
		self.temConv2 = nn.Sequential(
				nn.Conv2d(inplanes, fnum, kernel_size=(3, 3), stride=1, padding = 1),
				nn.BatchNorm2d(fnum, track_running_stats=False),
				nn.ReLU(inplace = True),
		)
		self.temConv3 = nn.Sequential(
				nn.Conv2d(inplanes, fnum, kernel_size=(3, 3), stride=1, padding = 2, dilation = 2),
				nn.BatchNorm2d(fnum, track_running_stats=False),
				nn.ReLU(inplace = True),
		)
		self.temConv4 = nn.Sequential(
				nn.Conv2d(inplanes, fnum, kernel_size=(3, 3), stride=1, padding = 4, dilation = 4),
				nn.BatchNorm2d(fnum, track_running_stats=False),
				nn.ReLU(inplace = True),
		)
		self.conv = nn.Conv2d(4, 1, kernel_size=(1, 1), stride=1, padding = 0)

	def forward(self, x):
		x1 = self.temConv1(x)
		x2 = self.temConv2(x)
		x3 = self.temConv3(x)
		x4 = self.temConv4(x)
		y = torch.cat((x1, x2, x3, x4), 1)
		return y

class fusionResNet50(nn.Module):
	def __init__(self , model, batchSize, landmarksNum, useGPU, image_scale, R):
		super(fusionResNet50, self).__init__()


		para_list = list(model.children())
		self.relu = nn.ReLU(inplace = True)
		self.resnet_layer1 = nn.Sequential(*para_list[:5])
		self.resnet_layer2 = para_list[5]
		self.resnet_layer3 = para_list[6]
		self.resnet_layer4 = para_list[7]

		fnum = 96
		self.fnum = fnum
		self.f_conv4 = nn.Sequential(
			nn.Conv2d(2048, fnum, kernel_size=(1, 1), stride=1, padding = 0 ),
			nn.BatchNorm2d(fnum,track_running_stats=False),
			nn.ReLU(True),
		)

		self.f_conv3 = nn.Sequential(
			nn.Conv2d(1024, fnum, kernel_size=(1, 1), stride=1, padding = 0 ),
			nn.BatchNorm2d(fnum,track_running_stats=False),
			nn.ReLU(True),
		)

		self.f_conv2 = nn.Sequential(
			nn.Conv2d(512, fnum, kernel_size=(1, 1), stride=1, padding = 0 ),
			nn.BatchNorm2d(fnum,track_running_stats=False),
			nn.ReLU(True),
		)

		self.f_conv1 = nn.Sequential(
			nn.Conv2d(256, fnum, kernel_size=(1, 1), stride=1, padding = 0 ),
			nn.BatchNorm2d(fnum,track_running_stats=False),
			nn.ReLU(True),
		)

		self.avgPool8t = nn.AvgPool2d(8, 8)
		self.avgPool4t = nn.AvgPool2d(4, 4)
		self.avgPool2t = nn.AvgPool2d(2, 2)
		self.attentionLayer1 = nn.Sequential(
			nn.Linear(500, 128, bias = False),
			nn.BatchNorm1d(1,track_running_stats=False),
			nn.Tanh(),
			nn.Linear(128, landmarksNum*3, bias = False),
			#~ nn.BatchNorm1d(1,track_running_stats=False),
			nn.Softmax(dim = 0)
		)

		moduleList = []
		for i in range(landmarksNum*3):
			#~ temConv = dilationInceptionModule(fnum*4, 1)
			temConv = nn.Conv2d(fnum*4, 1, kernel_size=(1, 1), stride=1, padding = 0)
			moduleList.append(temConv)

		self.moduleList = nn.ModuleList(moduleList)

		scaleFactorList = []
		for i in range(landmarksNum*3):
			scaleFactorList.append(nn.Linear(1, 1, bias = False))
		self.scaleFactorList = nn.ModuleList(scaleFactorList)

		self.inception = dilationInceptionModule(fnum*4, fnum*4)
		self.prediction = nn.Conv2d(2048, landmarksNum*3, kernel_size=(1, 1), stride=1, padding = 0 )
		self.Upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
		self.Upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
		self.Upsample8 = nn.Upsample(scale_factor=8, mode='bilinear')
		self.Upsample16 = nn.Upsample(scale_factor=16, mode='bilinear')
		self.Upsample32 = nn.Upsample(scale_factor=32, mode='bilinear')

		self.landmarksNum = landmarksNum
		self.batchSize = batchSize
		self.useGPU = useGPU
		self.R2 = R

	def getCoordinate(self, outputs1):
		heatmaps = F.sigmoid(outputs1[:, 0:self.landmarksNum, :, :])
		heatmap_sum = torch.sum(heatmaps.view(self.batchSize, self.landmarksNum, -1), dim=2)

		Xmap1 = heatmaps * self.coordinateX
		Ymap1 = heatmaps * self.coordinateY

		Xmean1 = torch.sum(Xmap1.view(self.batchSize, self.landmarksNum, -1), dim=2) / heatmap_sum
		Ymean1 = torch.sum(Ymap1.view(self.batchSize, self.landmarksNum, -1), dim=2) / heatmap_sum

		coordinateMean1 = torch.stack([Xmean1, Ymean1]).permute(1, 2, 0)
		coordinateMean2 = 0

		XDevmap = torch.pow(self.coordinateX - Xmean1.view(self.batchSize, self.landmarksNum, 1, 1), 2)
		YDevmap = torch.pow(self.coordinateY - Ymean1.view(self.batchSize, self.landmarksNum, 1, 1), 2)

		XDevmap = heatmaps * XDevmap
		YDevmap = heatmaps * YDevmap

		coordinateDev = torch.sum((XDevmap + YDevmap).view(self.batchSize, self.landmarksNum, -1), dim=2) / heatmap_sum

		return coordinateMean1, coordinateMean2, coordinateDev

	def getAttention(self, bone, fnum):
		bone = self.avgPool8t(bone).view(fnum, -1)
		bone = bone.unsqueeze(1)
		y = self.attentionLayer1(bone).squeeze(1).transpose(1, 0)

		return y

	def predictionWithAttention(self, bone, attentions):
		featureNum, channelNum = attentions.size()[0], attentions.size()[1]
		attentionMaps = []
		for i in range(featureNum):
			attention = attentions[i, :]
			attention = attention.view(1, channelNum, 1, 1)
			attentionMap = attention * bone * channelNum
			attentionMaps.append(self.moduleList[i](attentionMap))
		attentionMaps = torch.stack(attentionMaps).squeeze().unsqueeze(0)
		return attentionMaps

	def forward(self, x):
		x = self.resnet_layer1(x)
		f1 = self.f_conv1(x)
		#~ print(x.size())
		x = self.resnet_layer2(x)
		f2 = self.f_conv2(x)
		#~ print(x.size())
		x = self.resnet_layer3(x)
		f3 = self.f_conv3(x)
		#~ print(x.size())
		x = self.resnet_layer4(x)
		f4 = self.f_conv4(x)

		f2 = self.Upsample2(f2)
		f3 = self.Upsample4(f3)
		f4 = self.Upsample8(f4)

		bone = torch.cat((f1, f2, f3, f4), 1)
		bone = self.inception(bone)
		attention = self.getAttention(bone, self.fnum*4)

		y = self.Upsample4(self.predictionWithAttention(bone, attention))
		coordinateMean1, coordinateMean2 = 0, 0

		return [y], coordinateMean1, coordinateMean2

class fusionVGG19(nn.Module):
	def __init__(self , model, config):
		super(fusionVGG19, self).__init__()
		para_list = list(model.children())[0]

		self.VGG_layer1 = nn.Sequential(*para_list[:14])
		self.VGG_layer2 = nn.Sequential(*para_list[14:27])
		self.VGG_layer3 = nn.Sequential(*para_list[27:40])
		self.VGG_layer4 = nn.Sequential(*para_list[40:])
		self.relu = nn.ReLU(inplace = True)
		
		fnum = 64
		self.fnum = fnum
		self.f_conv4 = nn.Sequential(
			nn.Conv2d(512, fnum, kernel_size=(1, 1), stride=1, padding = 0 ),
			nn.BatchNorm2d(fnum,track_running_stats=False),
			nn.ReLU(True),
		)
		
		self.f_conv3 = nn.Sequential(
			nn.Conv2d(512, fnum, kernel_size=(1, 1), stride=1, padding = 0 ),
			nn.BatchNorm2d(fnum,track_running_stats=False),
			nn.ReLU(True),
		)
		
		self.f_conv2 = nn.Sequential(
			nn.Conv2d(256, fnum, kernel_size=(1, 1), stride=1, padding = 0 ),
			nn.BatchNorm2d(fnum,track_running_stats=False),
			nn.ReLU(True),
		)
		
		self.f_conv1 = nn.Sequential(
			nn.Conv2d(128, fnum, kernel_size=(1, 1), stride=1, padding = 0 ),
			nn.BatchNorm2d(fnum,track_running_stats=False),
			nn.ReLU(True),
		)
		
		self.avgPool8t = nn.AvgPool2d(8, 8)
		self.avgPool4t = nn.AvgPool2d(4, 4)
		self.avgPool2t = nn.AvgPool2d(2, 2)
		self.attentionLayer1 = nn.Sequential(
			nn.Linear(500, 128, bias = False),
			nn.BatchNorm1d(1,track_running_stats=False),
			nn.Tanh(),
			nn.Linear(128, config.landmarkNum*3, bias = False),
			nn.Softmax(dim = 0)
		)
		
		moduleList = []
		for i in range(config.landmarkNum*3):
			temConv = nn.Conv2d(fnum*4, 1, kernel_size=(1, 1), stride=1, padding = 0)
			moduleList.append(temConv)
			
		self.moduleList = nn.ModuleList(moduleList)
		self.dilated_block = dilationInceptionModule(fnum*4, fnum*4)
		self.prediction = nn.Conv2d(fnum*4, config.landmarkNum*3, kernel_size=(1, 1), stride=1, padding = 0)
		self.Upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
		self.Upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
		self.Upsample8 = nn.Upsample(scale_factor=8, mode='bilinear')
		self.Upsample16 = nn.Upsample(scale_factor=16, mode='bilinear')
		self.Upsample32 = nn.Upsample(scale_factor=32, mode='bilinear')
		
		self.landmarksNum = config.landmarkNum
		self.batchSize = config.batchSize
		self.R2 = config.R2
		
		self.higth, self.width = config.image_scale
		
		self.coordinateX = torch.ones(self.batchSize, self.landmarksNum, self.higth, self.width).cuda(config.use_gpu)
		self.coordinateY = torch.ones(self.batchSize, self.landmarksNum, self.higth, self.width).cuda(config.use_gpu)
		
		for i in range(self.higth):
			self.coordinateX[:, :, i, :] = self.coordinateX[:, :, i, :] * i
			
		for i in range(self.width):
			self.coordinateY[:, :, :, i] = self.coordinateY[:, :, :, i] * i

		self.coordinateX, self.coordinateY = self.coordinateX / (self.higth - 1), self.coordinateY / (self.width - 1)

	def getCoordinate(self, outputs1):
		heatmaps = F.sigmoid(outputs1[:, 0:self.landmarksNum, :, :])
		heatmap_sum = torch.sum(heatmaps.view(self.batchSize, self.landmarksNum, -1), dim = 2)
		
		Xmap1 = heatmaps * self.coordinateX
		Ymap1 = heatmaps * self.coordinateY
		
		Xmean1 = torch.sum(Xmap1.view(self.batchSize, self.landmarksNum, -1), dim = 2) / heatmap_sum
		Ymean1 = torch.sum(Ymap1.view(self.batchSize, self.landmarksNum, -1), dim = 2) / heatmap_sum

		coordinateMean1 = torch.stack([Xmean1, Ymean1]).permute(1, 2, 0)
		coordinateMean2 = 0

		XDevmap = torch.pow(self.coordinateX - Xmean1.view(self.batchSize, self.landmarksNum, 1, 1), 2)
		YDevmap = torch.pow(self.coordinateY - Ymean1.view(self.batchSize, self.landmarksNum, 1, 1), 2)

		XDevmap = heatmaps * XDevmap
		YDevmap = heatmaps * YDevmap
		
		coordinateDev = torch.sum((XDevmap + YDevmap).view(self.batchSize, self.landmarksNum, -1), dim = 2) / heatmap_sum
		
		return coordinateMean1, coordinateMean2, coordinateDev

	def getAttention(self, bone, fnum):
		bone = self.avgPool8t(bone).view(fnum, -1)
		bone = bone.unsqueeze(1)
		y = self.attentionLayer1(bone).squeeze(1).transpose(1, 0)
		return y

	def predictionWithAttention(self, bone, attentions):
		featureNum, channelNum = attentions.size()[0], attentions.size()[1]

		attentionMaps = []
		for i in range(featureNum):
			attention = attentions[i, :]
			attention = attention.view(1, channelNum, 1, 1)
			attentionMap = attention * bone * channelNum
			attentionMaps.append(self.moduleList[i](attentionMap))

		attentionMaps = torch.stack(attentionMaps).squeeze().unsqueeze(0)
		return attentionMaps

	def forward(self, x):
		x = self.VGG_layer1(x)
		f1 = self.f_conv1(x)

		x = self.VGG_layer2(x)
		f2 = self.f_conv2(x)

		x = self.VGG_layer3(x)
		f3 = self.f_conv3(x)

		x = self.VGG_layer4(x)
		f4 = self.f_conv4(x)

		f2 = self.Upsample2(f2)
		f3 = self.Upsample4(f3)
		f4 = self.Upsample8(f4)
		bone = torch.cat((f1, f2, f3, f4), 1)

		# Attentive Feature Pyramid Fusion
		bone = self.dilated_block(bone)
		attention = self.getAttention(bone, self.fnum*4)
		y = self.Upsample4(self.predictionWithAttention(bone, attention))

		# predicting landmarks with the integral operation
		# coordinateMean1, coordinateMean2, coordinateDev = self.getCoordinate(y)

		return [y]



