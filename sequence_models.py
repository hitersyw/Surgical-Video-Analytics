import torch
import torch.nn as nn
from torch.autograd import Variable

class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()
		
	def forward(self, x):
		return x

class RGBD_CNN(nn.Module):

	def __init__(self, cnn_backbone=None, classCount=38):

		super(RGBD_CNN, self).__init__()
		
		self.rgb_backbone=cnn_backbone
		self.depth_backbone=cnn_backbone
		
		self.fusion_layer=nn.Linear(self.rgb_backbone.resnet.fc.in_features, classCount)

		self.rgb_backbone.resnet.fc=Identity()
		self.depth_backbone.resnet.fc=Identity()

	def forward(self, x, y):

		rgb_out=self.rgb_backbone(x)
		depth_out=self.depth_backbone(y)

		fusion=rgb_out*depth_out

		out=self.fusion_layer(fusion)

		return out

class LSTM_Decoder(nn.Module):

	def __init__(self, cnn_backbone=None, model_type='rgb_cnn', hidden_dim_ratio=2, num_layers=1, bidir=False, drop_rate=0.0, classCount=38):

		super(LSTM_Decoder, self).__init__()
		
		self.encoder=cnn_backbone

		if model_type!='rgbd_cnn':
			self.embedding_dim=self.encoder.resnet.fc.in_features
			self.encoder.resnet.fc=Identity()
		else:
			self.embedding_dim=self.encoder.fusion_layer.in_features
			self.encoder.fusion_layer=Identity()

		self.model_type=model_type
		self.num_layers=num_layers
		self.hidden_dim=self.embedding_dim//hidden_dim_ratio
		self.score_layer=nn.Linear(self.hidden_dim, classCount)
		
		self.lstm=nn.LSTM(
			input_size=self.embedding_dim,
			hidden_size=self.hidden_dim,
			num_layers=num_layers,
			batch_first=True,
			dropout=drop_rate,
			bidirectional=bidir
		)

	def init_hidden(self, batch_size):
		# the weights are of the form (nb_layers, batch_size, nb_lstm_units)
		h_0 = torch.randn(self.num_layers, batch_size, self.hidden_dim)
		c_0 = torch.randn(self.num_layers, batch_size, self.hidden_dim)

		h_0 = Variable(h_0.cuda())
		c_0 = Variable(c_0.cuda())

		return (h_0, c_0)

	def forward(self, *inputs):

		x=inputs[0]

		self.hidden=self.init_hidden(x.shape[0])

		desired_shape=[-1] + list(x.shape[2:])
		x_5d_to_4d=x.view(desired_shape)

		if self.model_type=='rgbd_cnn':
			y_5d_to_4d=inputs[1].view(desired_shape)
			features=self.encoder(x_5d_to_4d, y_5d_to_4d)
		else:
			features=self.encoder(x_5d_to_4d)

		lstm_input=features.view(x.shape[0], x.shape[1], -1)
		lstm_out, (h_n, c_n) = self.lstm(lstm_input, self.hidden)
		
		out=self.score_layer(h_n[-1])

		return out