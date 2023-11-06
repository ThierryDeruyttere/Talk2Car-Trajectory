import torch
import torch.nn as nn

from cnn_backbone.resnet import resnet
from utils.spline_path_torch import interpolate_waypoints_using_splines
import torch.nn.functional as F


class YNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, output_len, num_waypoints=0, height=192, width=288):
        """
        Decoder models
        :param encoder_channels: list, encoder channels, used for skip connections
        :param decoder_channels: list, decoder channels
        :param output_len: int, pred_len
        :param num_waypoints: int number of waypoints - for waypoint decoder
        """
        super(YNetDecoder, self).__init__()

        # The trajectory decoder takes in addition the conditioned goal and waypoints as an additional image channel
        if num_waypoints > 0:
            encoder_channels = [channel + num_waypoints for channel in encoder_channels]
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

        self.combiner = nn.Sequential(
            nn.Linear(512+768, 1024),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            # nn.Dropout(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, output_len * 2)
        )

        self.resnet = resnet(
                "ResNet-18", in_channels=68, num_classes=512
            )
        self.output_len = output_len

    def forward(self, features, command_embedding):
        # Takes in the list of feature maps from the encoder. Trajectory predictor in addition the goal and waypoint heatmaps
        features = features
        center_feature = features[0]
        x = self.center(center_feature)
        for i, (feature, module, upsample_conv) in enumerate(zip(features[1:], self.decoder, self.upsample_conv)):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # bilinear interpolation for upsampling
            x = upsample_conv(x)  # 3x3 conv for upsampling
            x = torch.cat([x, feature], dim=1)  # concat encoder and decoder features
            x = module(x)  # Conv
        x = self.combiner(torch.cat([self.resnet(x), command_embedding], dim=-1))  # last predictor layer
        x = self.predictor(x)

        return x.view(-1, self.output_len, 2)



class NeuralInterpolationHead(nn.Module):
    def __init__(self, num_waypoints, num_path_nodes, norm_feats=False, base_channels=15, neural_interpolation_type="FPN"):
        super(NeuralInterpolationHead, self).__init__()
        ### Interpolation head
        self.norm_feats = norm_feats
        self.num_path_nodes = num_path_nodes
        self.neural_interpolation_type = neural_interpolation_type
        extra_channels = num_waypoints

        encoder_dim = 512
        if self.neural_interpolation_type == "FPN":
            self.encoder = resnet(
                "ResNet-18", in_channels=base_channels + extra_channels, num_classes=512
            )
        elif self.neural_interpolation_type == "features":
            pass


        self.combiner = nn.Sequential(
            nn.Linear(encoder_dim + 768, 1024),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            # nn.Dropout(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, num_path_nodes * 2)
        )

    def forward(self, features, waypoints, command_embedding):
        features = torch.cat([features, waypoints], 1)

        x = self.encoder(features)
        x = self.combiner(torch.cat([x, command_embedding], dim=-1))
        x = self.predictor(x)

        return x.view(-1, self.num_path_nodes, 2)


class SplineInterpolationHead(nn.Module):
    def __init__(self):
        super(SplineInterpolationHead, self).__init__()

    def forward(self, waypoints, num_path_nodes: int):
        """
        Interpolates waypoints using hermetic splines
        Args:
            waypoints: waypoints [bs, num_waypoints, 2]
            num_path_nodes: desired number of nodes in path (int)
        Returns:
            trajectory: trajectory [bs, num_path_nodes, 2]
        """
        trajectory = interpolate_waypoints_using_splines(waypoints, num_path_nodes)
        return trajectory
