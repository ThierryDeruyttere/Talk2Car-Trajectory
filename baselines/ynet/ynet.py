import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.softargmax import SoftArgmax2D, create_meshgrid

def clones(module, N):
    '''Produce N identical layers'''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MLP(nn.Module):
	# From PECNET code
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_size=(1024, 512),
        activation="relu",
        discrim=False,
        dropout=-1,
    ):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(
                        min(0.1, self.dropout / 3) if i == 1 else self.dropout
                    )(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x

class YNetEncoder(nn.Module):
	def __init__(self, in_channels, channels=(64, 128, 256, 512, 512)):
		"""
		Encoder model
		:param in_channels: int, semantic_classes + obs_len
		:param channels: list, hidden layer channels
		"""
		super(YNetEncoder, self).__init__()
		self.stages = nn.ModuleList()

		# First block
		self.stages.append(nn.Sequential(
			nn.Conv2d(in_channels, channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True),
		))


		# Subsequent blocks, each starting with MaxPool
		for i in range(len(channels)-1):
			self.stages.append(nn.Sequential(
				nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
				nn.Conv2d(channels[i], channels[i+1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
				nn.ReLU(inplace=True),
				nn.Conv2d(channels[i+1], channels[i+1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
				nn.ReLU(inplace=True)))


		# Last MaxPool layer before passing the features into decoder
		self.stages.append(nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)))

	def forward(self, x):
		# Saves the feature maps Tensor of each layer into a list, as we will later need them again for the decoder
		features = []

		for i, stage in enumerate(self.stages):
			x = stage(x)
			features.append(x)

		return features


class YNetDecoder(nn.Module):
	def __init__(self, encoder_channels, decoder_channels, output_len, traj=False):
		"""
		Decoder models
		:param encoder_channels: list, encoder channels, used for skip connections
		:param decoder_channels: list, decoder channels
		:param output_len: int, pred_len
		:param traj: False or int, if False -> Goal and waypoint predictor, if int -> number of waypoints
		"""
		super(YNetDecoder, self).__init__()

		# The trajectory decoder takes in addition the conditioned goal and waypoints as an additional image channel
		if traj:
			encoder_channels = [channel+traj for channel in encoder_channels]
		encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder
		center_channels = encoder_channels[0]

		decoder_channels = decoder_channels

		# The center layer (the layer with the smallest feature map size)
		self.center = nn.Sequential(
			nn.Conv2d(center_channels, center_channels*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True),
			nn.Conv2d(center_channels*2, center_channels*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True)
		)

		# Determine the upsample channel dimensions
		upsample_channels_in = [center_channels*2] + decoder_channels[:-1]
		upsample_channels_out = [num_channel // 2 for num_channel in upsample_channels_in]

		# Upsampling consists of bilinear upsampling + 3x3 Conv, here the 3x3 Conv is defined
		self.upsample_conv = [
			nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			for in_channels_, out_channels_ in zip(upsample_channels_in, upsample_channels_out)]
		self.upsample_conv = nn.ModuleList(self.upsample_conv)

		# Determine the input and output channel dimensions of each layer in the decoder
		# As we concat the encoded feature and decoded features we have to sum both dims
		in_channels = [enc + dec for enc, dec in zip(encoder_channels, upsample_channels_out)]
		out_channels = decoder_channels

		self.decoder = [nn.Sequential(
			nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True))
			for in_channels_, out_channels_ in zip(in_channels, out_channels)]
		self.decoder = nn.ModuleList(self.decoder)


		# Final 1x1 Conv prediction to get our heatmap logits (before softmax)
		self.predictor = nn.Conv2d(in_channels=decoder_channels[-1], out_channels=output_len, kernel_size=1, stride=1, padding=0)

	def forward(self, features):
		# Takes in the list of feature maps from the encoder. Trajectory predictor in addition the goal and waypoint heatmaps
		features = features[::-1]  # reverse the order of encoded features, as the decoder starts from the smallest image
		center_feature = features[0]
		x = self.center(center_feature.float())
		for i, (feature, module, upsample_conv) in enumerate(zip(features[1:], self.decoder, self.upsample_conv)):
			x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # bilinear interpolation for upsampling
			x = upsample_conv(x)  # 3x3 conv for upsampling
			x = torch.cat([x, feature], dim=1)  # concat encoder and decoder features
			x = module(x)  # Conv
		x = self.predictor(x)  # last predictor layer
		return x


class YNetTorch(nn.Module):
	def __init__(
		self,
		pred_len,
		encoder_channels=[],
		decoder_channels=[],
		waypoints=1
	):
		"""
		Complete Y-net Architecture including semantic segmentation backbone, heatmap embedding and ConvPredictor
		:param pred_len: int, predicted timesteps
		:param encoder_channels: list, encoder channel structure
		:param decoder_channels: list, decoder channel structure
		:param waypoints: int, number of waypoints
		"""
		super(YNetTorch, self).__init__()

		self.encoder = YNetEncoder(in_channels=15, channels=encoder_channels)

		self.goal_decoder = YNetDecoder(encoder_channels, decoder_channels, output_len=pred_len)
		self.traj_decoder = YNetDecoder(encoder_channels, decoder_channels, output_len=pred_len, traj=waypoints)

		self.softargmax_ = SoftArgmax2D(normalized_coordinates=False)

	# Forward pass for goal decoder
	def pred_goal(self, features):
		goal = self.goal_decoder(features)
		return goal

	# Forward pass for trajectory decoder
	def pred_traj(self, features):
		traj = self.traj_decoder(features)
		return traj

	# Forward pass for feature encoder, returns list of feature maps
	def pred_features(self, x):
		features = self.encoder(x)
		return features

	# Softmax for Image data as in dim=NxCxHxW, returns softmax image shape=NxCxHxW
	def softmax(self, x):
		return nn.Softmax(2)(x.view(*x.size()[:2], -1)).view_as(x)

	# Softargmax for Image data as in dim=NxCxHxW, returns 2D coordinates=Nx2
	def softargmax(self, output):
		return self.softargmax_(output)

	def sigmoid(self, output):
		return torch.sigmoid(output)

	def softargmax_on_softmax_map(self, x):
		""" Softargmax: As input a batched image where softmax is already performed (not logits) """
		pos_y, pos_x = create_meshgrid(x, normalized_coordinates=False)
		pos_x = pos_x.reshape(-1)
		pos_y = pos_y.reshape(-1)
		x = x.flatten(2)

		estimated_x = pos_x * x
		estimated_x = torch.sum(estimated_x, dim=-1, keepdim=True)
		estimated_y = pos_y * x
		estimated_y = torch.sum(estimated_y, dim=-1, keepdim=True)
		softargmax_coords = torch.cat([estimated_x, estimated_y], dim=-1)
		return softargmax_coords

