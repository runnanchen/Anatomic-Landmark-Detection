from __future__ import print_function, division
import torch
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def get_statistical_results(offset, config):
	SDR = torch.zeros(config.landmarkNum, 5)
	SD = torch.zeros(config.landmarkNum)
	MRE = torch.mean(offset, 0)

	for landmarkId in range(config.landmarkNum):
		landmarkCol = offset[:, landmarkId].clone()
		train_mm = torch.tensor([landmarkCol[landmarkCol <= 1].size()[0], \
								 landmarkCol[landmarkCol <= 2].size()[0], \
								 landmarkCol[landmarkCol <= 2.5].size()[0], \
								 landmarkCol[landmarkCol <= 3.0].size()[0], \
								 landmarkCol[landmarkCol <= 4.0].size()[0]]).float()
		SDR[landmarkId, :] = train_mm / landmarkCol.shape[0]
		SD[landmarkId] = torch.sqrt(
			torch.sum(torch.pow(landmarkCol - MRE[landmarkId], 2)) / (landmarkCol.shape[0] - 1))
	return SDR, SD, MRE

def regression_voting(heatmaps, R):
	#print("11", time.asctime())
	topN = int(R * R * 3.1415926)
	heatmap = heatmaps[0]
	imageNum, featureNum, h, w = heatmap.size()
	landmarkNum = int(featureNum/3)
	heatmap = heatmap.contiguous().view(imageNum, featureNum, -1)
	
	predicted_landmarks = torch.zeros((imageNum, landmarkNum, 2))
	Pmap = heatmap[:, 0:landmarkNum, :].data
	Xmap = torch.round(heatmap[:, landmarkNum:landmarkNum*2, :].data * R).long() * w
	Ymap = torch.round(heatmap[:, landmarkNum*2:landmarkNum*3, :].data * R).long()
	topkP, indexs = torch.topk(Pmap, topN)
	#~ plt.imshow(Pmap.reshape(imageNum, landmarkNum, h,w)[0][0], cmap='gray', interpolation='nearest')
	for imageId in range(imageNum):
		for landmarkId in range(landmarkNum):
			
			topnXoff = Xmap[imageId][landmarkId][indexs[imageId][landmarkId]]				#offset in x direction
			topnYoff = Ymap[imageId][landmarkId][indexs[imageId][landmarkId]]				#offset in y direction
			VotePosi = (topnXoff + topnYoff + indexs[imageId][landmarkId]).cpu().numpy().astype("int")
			tem = VotePosi[VotePosi >= 0]
			maxid = 0
			if len(tem) >0:
				maxid = np.argmax(np.bincount(tem))
			x = maxid // w
			y = maxid - x * w
			x, y = x /(h - 1), y / (w - 1)
			predicted_landmarks[imageId][landmarkId] = torch.Tensor([x, y])
	return predicted_landmarks

def Mydist(a, b):
	x1, y1 = a
	x2, y2 = b
	return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_deviation(coordinates1, lables):
	coordinates1_b = coordinates1.clone()
	lables_b = lables.clone()
	
	coordinates1_b[:, :, 0] = coordinates1_b[:, :, 0] * 1934
	coordinates1_b[:, :, 1] = coordinates1_b[:, :, 1] * 2399
	
	lables_b[:, :, 0] = lables_b[:, :, 0] * 1934
	lables_b[:, :, 1] = lables_b[:, :, 1] * 2399
	
	tem_dist = torch.sqrt(torch.sum(torch.pow(coordinates1_b - lables_b, 2), 2))
	return tem_dist
