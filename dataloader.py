import os
import numpy as np
from PIL import Image
from matplotlib import cm
import torch
from torch.utils import data
import cv2

def resize_with_aspect_ratio(im, desired_size):
	
	old_size = im.shape[:2] # old_size is in (height, width) format

	ratio = float(desired_size)/max(old_size)
	new_size = tuple([int(x*ratio) for x in old_size])

	# new_size should be in (width, height) format
	im = cv2.resize(im, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)

	delta_w = desired_size - new_size[1]
	delta_h = desired_size - new_size[0]
	top, bottom = delta_h//2, delta_h-(delta_h//2)
	left, right = delta_w//2, delta_w-(delta_w//2)

	color = [0, 0, 0]
	new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
	
	return new_im

class Dataset(data.Dataset):
	
	def __init__(self, rgb_path=None, labels_path=None, depth_path=None, model_type='rgbd_cnn', resize_dim=256, transform=None):
		
		self.model_type=model_type
		self.resize_dim=resize_dim
		self.transform=transform
		
		self.rgb_path=rgb_path
		self.depth_path=depth_path
		self.labels_path=labels_path

		if rgb_path and self.model_type=='rgb_cnn':
			rgb_frame_list=os.listdir(self.rgb_path)
			self.list_IDs = rgb_frame_list
		
		if depth_path and self.model_type!='rgb_cnn':
			depth_frame_list=os.listdir(self.depth_path)
			self.list_IDs=depth_frame_list
		
	def __len__(self):
		
		return len(self.list_IDs)

	def __getitem__(self, index):

		frame=self.list_IDs[index]

		if self.model_type=='rgb_cnn':
			rgb_frame=np.load(self.rgb_path+frame)
			rgb_frame=resize_with_aspect_ratio(rgb_frame, self.resize_dim)
			rgb_pil=Image.fromarray(rgb_frame.astype('uint8'), 'RGB')
			X=self.transform(rgb_pil)
		
		if self.model_type=='depth_cnn':
			depth_frame=np.load(self.depth_path+frame)
			depth_frame=resize_with_aspect_ratio(depth_frame, self.resize_dim)
			jet_img=cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
			jet_pil=Image.fromarray(jet_img.astype('uint8'))
			X=self.transform(jet_pil)
		
		if self.model_type=='rgbd_cnn':
			rgb_frame=np.load(self.rgb_path+frame)
			rgb_frame=resize_with_aspect_ratio(rgb_frame, self.resize_dim)
			rgb_pil=Image.fromarray(rgb_frame.astype('uint8'), 'RGB')

			depth_frame=np.load(self.depth_path+frame)
			depth_frame=resize_with_aspect_ratio(depth_frame, self.resize_dim)
			jet_img=cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
			jet_pil=Image.fromarray(jet_img.astype('uint8'))

			X={'rgb':self.transform(rgb_pil), 'depth':self.transform(jet_pil)}

		y=np.load(self.labels_path+frame) 

		return X, y

class LSTM_seq_loader(data.Dataset):
	
	def __init__(self, rgb_path=None, labels_path=None, depth_path=None, model_type='rgbd_cnn', resize_dim=256, context_len=10, transform=None):
		
		self.model_type=model_type
		self.resize_dim=resize_dim
		self.context_len=context_len
		self.transform=transform
		
		self.rgb_path=rgb_path
		self.depth_path=depth_path
		self.labels_path=labels_path

		if rgb_path and self.model_type=='rgb_cnn':
			rgb_frame_list=os.listdir(self.rgb_path)
			self.list_IDs = [f for f in rgb_frame_list if int(f[5:-4])>self.context_len] # discard frames which cannot have 'context_len' # of frames
		
		if depth_path and self.model_type!='rgb_cnn':
			depth_frame_list=os.listdir(self.depth_path)
			self.list_IDs = [f for f in depth_frame_list if int(f[5:-4])>self.context_len] # discard frames which cannot have 'context_len' # of frames
		
	def __len__(self):
		
		return len(self.list_IDs)

	def __getitem__(self, index):

		frame=self.list_IDs[index]
		frame_num=int(frame[5:-4])

		if self.model_type=='rgb_cnn':
			seq_tensor=torch.zeros([self.context_len+1, 3, 224, 224])
			for i in range(self.context_len, -1, -1):
				rgb_frame=np.load(self.rgb_path+frame[:5]+str(frame_num-i)+frame[-4:])
				rgb_frame=resize_with_aspect_ratio(rgb_frame, self.resize_dim)
				rgb_pil=Image.fromarray(rgb_frame.astype('uint8'), 'RGB')
				seq_tensor[self.context_len-i]=self.transform(rgb_pil)
			X=seq_tensor
		
		if self.model_type=='depth_cnn':
			seq_tensor=torch.zeros([self.context_len+1, 3, 224, 224])
			for i in range(self.context_len, -1, -1):
				depth_frame=np.load(self.depth_path+frame[:5]+str(frame_num-i)+frame[-4:])
				depth_frame=resize_with_aspect_ratio(depth_frame, self.resize_dim)
				jet_img=cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
				jet_pil=Image.fromarray(jet_img.astype('uint8'))
				seq_tensor[self.context_len-i]=self.transform(jet_pil)
			X=seq_tensor
		
		if self.model_type=='rgbd_cnn':
			rgb_seq_tensor=torch.zeros([self.context_len+1, 3, 224, 224])
			for i in range(self.context_len, -1, -1):
				rgb_frame=np.load(self.rgb_path+frame[:5]+str(frame_num-i)+frame[-4:])
				rgb_frame=resize_with_aspect_ratio(rgb_frame, self.resize_dim)
				rgb_pil=Image.fromarray(rgb_frame.astype('uint8'), 'RGB')
				rgb_seq_tensor[self.context_len-i]=self.transform(rgb_pil)

			depth_seq_tensor=torch.zeros([self.context_len+1, 3, 224, 224])
			for i in range(self.context_len, -1, -1):
				depth_frame=np.load(self.depth_path+frame[:5]+str(frame_num-i)+frame[-4:])
				depth_frame=resize_with_aspect_ratio(depth_frame, self.resize_dim)
				jet_img=cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
				jet_pil=Image.fromarray(jet_img.astype('uint8'))
				depth_seq_tensor[self.context_len-i]=self.transform(jet_pil)

			X={'rgb':rgb_seq_tensor, 'depth':depth_seq_tensor}

		y=np.load(self.labels_path+frame) 

		return X, y