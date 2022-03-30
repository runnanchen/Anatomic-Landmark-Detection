from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import utils

class fusionLossFunc_improved(nn.Module):
	def __init__(self, config):
		super(fusionLossFunc_improved, self).__init__()
		# .use_gpu, R1, R2, image_scale, batchSize, landmarkNum
		self.use_gpu = config.use_gpu
		self.R1 = config.R1
		self.width = config.image_scale[1]
		self.higth = config.image_scale[0]
		self.imageNum = config.batchSize
		self.landmarkNum = config.landmarkNum

		self.binaryLoss = nn.BCEWithLogitsLoss(None, True).cuda(config.use_gpu)
		self.l1Loss = torch.nn.L1Loss().cuda(config.use_gpu)

		self.offsetMapx = np.ones((self.higth*2, self.width*2))
		self.offsetMapy = np.ones((self.higth*2, self.width*2))
		
		self.HeatMap = np.zeros((self.higth*2, self.width*2))
		self.mask = np.zeros((self.higth*2, self.width*2))
		
		#~ self.binary_class_groundTruth = Variable(torch.zeros(imageNum, landmarkNum, h, w).cuda(self.use_gpu))
		self.offsetMapX_groundTruth = Variable(torch.zeros(self.imageNum, self.landmarkNum, self.higth, self.width).cuda(self.use_gpu))
		self.offsetMapY_groundTruth = Variable(torch.zeros(self.imageNum, self.landmarkNum, self.higth, self.width).cuda(self.use_gpu))
		self.binary_class_groundTruth1 = Variable(torch.zeros(self.imageNum, self.landmarkNum, self.higth, self.width).cuda(self.use_gpu))
		self.binary_class_groundTruth2 = Variable(torch.zeros(self.imageNum, self.landmarkNum, self.higth, self.width).cuda(self.use_gpu))
		self.offsetMask = Variable(torch.zeros(self.imageNum, self.landmarkNum, self.higth, self.width).cuda(self.use_gpu))
		
		rr = config.R1
		dev = 4
		referPoint = (self.higth, self.width)
		for i in range(referPoint[0] - rr, referPoint[0] + rr + 1):
			for j in range(referPoint[1] - rr, referPoint[1] + rr + 1):
				temdis = utils.Mydist(referPoint, (i, j))
				if temdis <= rr:
					self.HeatMap[i][j] = 1
		rr = config.R2
		referPoint = (self.higth, self.width)
		for i in range(referPoint[0] - rr, referPoint[0] + rr + 1):
			for j in range(referPoint[1] - rr, referPoint[1] + rr + 1):
				temdis = utils.Mydist(referPoint, (i, j))
				if temdis <= rr:
					self.mask[i][j] = 1
		
		for i in range(2*self.higth):
			self.offsetMapx[i, :] = self.offsetMapx[i, :] * i
			
		for i in range(2*self.width):
			self.offsetMapy[:, i] = self.offsetMapy[:, i] * i
			
		
		self.offsetMapx = referPoint[0] - self.offsetMapx
		self.offsetMapy = referPoint[1] - self.offsetMapy
		self.HeatMap = Variable(torch.from_numpy(self.HeatMap)).cuda(self.use_gpu).float()
		self.mask = Variable(torch.from_numpy(self.mask)).cuda(self.use_gpu).float()
		self.offsetMapx = Variable(torch.from_numpy(self.offsetMapx)).cuda(self.use_gpu).float() / config.R2
		self.offsetMapy = Variable(torch.from_numpy(self.offsetMapy)).cuda(self.use_gpu).float() / config.R2
		
		self.zeroTensor = torch.zeros((self.imageNum, self.landmarkNum, self.higth, self.width)).cuda(self.use_gpu)
		
		return
	
	def getOffsetMask(self, h, w, X, Y):
		for imageId in range(self.imageNum):
			for landmarkId in range(self.landmarkNum):
				self.offsetMask[imageId, landmarkId, :, :] = self.mask[h - X[imageId][landmarkId]: 2*h - X[imageId][landmarkId], w - Y[imageId][landmarkId]: 2*w - Y[imageId][landmarkId]]
		return self.offsetMask
		
	def forward(self, featureMaps, landmarks):
		h, w = featureMaps.size()[2], featureMaps.size()[3]
		X = np.round((landmarks[:, :, 0] * (h - 1)).numpy()).astype("int")
		Y = np.round((landmarks[:, :, 1] * (w - 1)).numpy()).astype("int")
		binary_class_groundTruth = self.binary_class_groundTruth1

		for imageId in range(self.imageNum):
			for landmarkId in range(self.landmarkNum):
				#~ self.binary_class_groundTruth[imageId, landmarkId, :, :] = self.HeatMap[h - X[imageId][landmarkId]: 2*h - X[imageId][landmarkId], w - Y[imageId][landmarkId]: 2*w - Y[imageId][landmarkId]]
				binary_class_groundTruth[imageId, landmarkId, :, :] = self.HeatMap[h - X[imageId][landmarkId]: 2*h - X[imageId][landmarkId], w - Y[imageId][landmarkId]: 2*w - Y[imageId][landmarkId]]
				self.offsetMapX_groundTruth[imageId, landmarkId, :, :] = self.offsetMapx[h - X[imageId][landmarkId]: 2*h - X[imageId][landmarkId], w - Y[imageId][landmarkId]: 2*w - Y[imageId][landmarkId]]
				self.offsetMapY_groundTruth[imageId, landmarkId, :, :] = self.offsetMapy[h - X[imageId][landmarkId]: 2*h - X[imageId][landmarkId], w - Y[imageId][landmarkId]: 2*w - Y[imageId][landmarkId]]

		indexs = binary_class_groundTruth > 0
		temloss = [[2*self.binaryLoss(featureMaps[imageId][landmarkId], binary_class_groundTruth[imageId][landmarkId]), \
					
					self.l1Loss(featureMaps[imageId][landmarkId + self.landmarkNum*1][indexs[imageId][landmarkId]], \
						self.offsetMapX_groundTruth[imageId][landmarkId][indexs[imageId][landmarkId]]) , \
						
					self.l1Loss(featureMaps[imageId][landmarkId + self.landmarkNum*2][indexs[imageId][landmarkId]], \
						self.offsetMapY_groundTruth[imageId][landmarkId][indexs[imageId][landmarkId]])]

					for imageId in range(self.imageNum)
					for landmarkId in range(self.landmarkNum)]
		loss1 = (sum([sum(temloss[ind]) for ind in range(self.imageNum * self.landmarkNum)]))/(self.imageNum * self.landmarkNum)

		return loss1
