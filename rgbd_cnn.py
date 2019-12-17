import torch
from torchvision import models
import torch.nn as nn
from baseline_models import *

class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()
		
	def forward(self, x):
		return x

class RGBD_CNN(nn.Module):

	def __init__(self, backbone='r18', rgb_wts=None, depth_wts=None, classCount=38):

		super(RGBD_CNN, self).__init__()

		model_dict={'scnn':shallow_cnn(), 'r18':Resnet18(), 'r34':Resnet34(), 'r50':Resnet50(), 'r101':Resnet101(), 'r152':Resnet152()}
		
		self.rgb_backbone=model_dict[backbone]
		self.depth_backbone=model_dict[backbone]
		
		self.rgb_backbone.load_state_dict(rgb_wts)
		self.depth_backbone.load_state_dict(depth_wts)

		self.fusion_layer=nn.Linear(self.rgb_backbone.resnet.fc.in_features, classCount)

		self.rgb_backbone.resnet.fc=Identity()
		self.depth_backbone.resnet.fc=Identity()

	def forward(self, x, y):

		rgb_out=self.rgb_backbone(x)
		depth_out=self.depth_backbone(y)

		fusion=rgb_out*depth_out

		out=self.fusion_layer(fusion)

		return out