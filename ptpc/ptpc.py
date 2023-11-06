import torch
from torch import nn
import torch.distributions as D
import torch.nn.functional as F

from fpn_backbone.resnet_fpn import resnet_fpn_backbone
from combiner import Combiner
from interpolation_heads import NeuralInterpolationHead, SplineInterpolationHead, YNetDecoder


class Scale(nn.Module):
    def __init__(self, init=1.0):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor([init], dtype=torch.float32))

    def forward(self, input):
        return input * self.scale


def init_conv_kaiming(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_uniform_(module.weight, a=1)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_conv_std(module, std=0.01):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, std=std)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class PTPCHeadComm(nn.Module):
    def __init__(
        self,
        in_channel,
        n_conv=4,
        combine_at=2,
        num_scales=-1,
        command_information="channel_attention",
        upsampling_method="nearest",
        orig_img_size=(192, 288),
        norm_feats=False,
        num_waypoints=4,
        shared_command_fusion=True,
        kernel_size=1
    ):
        super().__init__()

        distr_tower_shared = []
        assert combine_at in range(0, n_conv - 1), "Impossible combiner placement"

        self.norm_feats = norm_feats
        self.orig_img_size = torch.Tensor(orig_img_size)
        self.command_information = command_information
        self.shared_command_fusion = shared_command_fusion
        if command_information == "none":
            self.combiner = None
        else:
            self.combiner = Combiner(
                command_dim=768,
                features_channels=in_channel,
                combination_method=command_information,
                shared_command_fusion=shared_command_fusion,
                num_scales=num_scales,
                kernel_size=kernel_size,
            )
        for i in range(combine_at):
            if i == 0:
                distr_tower_shared.append(
                    nn.Conv2d(256, in_channel, 3, padding=1, bias=False)
                )
            else:
                distr_tower_shared.append(
                    nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False)
                )
            distr_tower_shared.append(nn.GroupNorm(32, in_channel))
            distr_tower_shared.append(nn.ReLU())

        self.distr_tower_shared = nn.Sequential(*distr_tower_shared)

        mu_preds = []
        sigma_preds = []
        pi_preds = []
        distr_tower_scales = []
        self.num_waypoints = num_waypoints

        extra_channels = 0
        for scale_level in range(num_scales):
            distr_tower_scale = []
            for i in range(combine_at, n_conv):
                distr_tower_scale.append(
                    nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False)
                )
                distr_tower_scale.append(nn.GroupNorm(32, in_channel))
                distr_tower_scale.append(nn.ReLU())
            distr_tower_scale = nn.Sequential(*distr_tower_scale)
            distr_tower_scales.append(distr_tower_scale)
            """
            in_channel needs to be changed at each scale
            because higher res scale takes in the output of the lower res scale
            as conditioning.
            for i in len(scales):
                if i == 0:
                    in_channel = 256 (e.g.)
                else:
                    in_channel = 256 + len(level_points[i-1]) * (2 + 2 + 1) - if we just concatenate mu, sigma and pi (need to be upsampled somehow)
            """
            mu_preds.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channel, (in_channel + num_waypoints * 2) // 2, 3, padding=1
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        (in_channel + num_waypoints * 2) // 2,
                        2 * num_waypoints,
                        3,
                        padding=1,
                    ),
                )
            )
            sigma_preds.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channel, (in_channel + num_waypoints * 2) // 2, 3, padding=1
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        (in_channel + num_waypoints * 2) // 2,
                        2 * num_waypoints,
                        3,
                        padding=1,
                    ),
                )
            )
            pi_preds.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channel, (in_channel + num_waypoints * 2) // 2, 3, padding=1
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        (in_channel + num_waypoints * 2) // 2,
                        num_waypoints,
                        3,
                        padding=1,
                    ),
                )
            )

        ## Towers
        self.distr_tower_scales = nn.ModuleList(distr_tower_scales)
        self.mu_preds = nn.ModuleList(mu_preds)
        self.sigma_preds = nn.ModuleList(sigma_preds)
        self.pi_preds = nn.ModuleList(pi_preds)

        self.apply(init_conv_std)
        self.mu_scales = nn.ModuleList([Scale(1.0) for _ in range(num_scales)])
        self.sigma_scales = nn.ModuleList([Scale(1.0) for _ in range(num_scales)])
        self.upsampler = nn.Upsample(scale_factor=2, mode=upsampling_method)

    def forward(self, input, command_embedding):
        mus = []
        sigmas = []
        pis = []

        copy_command_embedding = command_embedding

        if self.shared_command_fusion and self.command_information == "text2conv":
            command_embedding = self.combiner.create_conv_filter(command_embedding)

        for ix, (feat, mu_scale, sigma_scale) in enumerate(
            list(zip(input, self.mu_scales, self.sigma_scales))
        ):

            if self.shared_command_fusion is False and self.command_information == "text2conv":
                command_embedding = self.combiner.create_conv_filter(copy_command_embedding, ix)

            feat = self.distr_tower_shared(feat)
            if self.combiner:
                feat = self.combiner(feat, command_embedding, ix=ix if self.shared_command_fusion is False else None)

            # Or do concatenation before this line
            feat = self.distr_tower_scales[ix](feat)

            mus.append(mu_scale(self.mu_preds[ix](feat)))
            sigmas.append(1 + F.elu(sigma_scale(self.sigma_preds[ix](feat))) + 1e-5)
            pis.append(self.pi_preds[ix](feat))

        return mus, sigmas, pis


class TrajectoryPredictor(nn.Module):
    def __init__(self, path_length, interpolation_method="neural",
                 base_channels=15, waypoints_ix=None, neural_interpolation_type="FPN"):
        super(TrajectoryPredictor, self).__init__()

        self.interpolation_method = interpolation_method
        self.path_length = path_length
        self.neural_interpolation_type = neural_interpolation_type

        if self.interpolation_method == "neural":
            if self.neural_interpolation_type == "FPN":
                self.interpolation_head = NeuralInterpolationHead(
                    len(waypoints_ix),
                    self.path_length,
                    base_channels=base_channels,
                )
            elif self.neural_interpolation_type == "features":
                self.downsample = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, padding=0)
                self.interpolation_head = YNetDecoder(
                    encoder_channels=[64 for _ in range(4)],
                    decoder_channels=[64 + len(waypoints_ix) for _ in range(4)],
                    output_len = self.path_length, num_waypoints=len(waypoints_ix),
                )


        elif self.interpolation_method == "spline":
            self.interpolation_head = SplineInterpolationHead()
        else:
            raise Exception("Unknown interpolation method", interpolation_method)

    def forward(
        self,
        waypoint_maps=None,
        layout=None,
        command=None,
        waypoint_samples=None,
        num_path_nodes=None,
        features=None,
    ):

        if self.interpolation_method == "neural":
            if self.neural_interpolation_type == "FPN":
                assert (
                    waypoint_maps is not None
                ), "waypoint_maps can't be None with interpolation method neural!"
                assert (
                    layout is not None
                ), "layout can't be None with interpolation method neural!"
                assert (
                    command is not None
                ), "command can't be None with interpolation method neural!"
                interpolated_path = self.interpolation_head(layout, waypoint_maps, command)

            elif self.neural_interpolation_type == "features":
                features = [self.downsample(feat) for feat in features]
                for ix, (feat, waypoints) in enumerate(zip(features, waypoint_maps)):
                    features[ix] = torch.cat([feat, waypoints], dim=1)
                interpolated_path = self.interpolation_head(features, command)

        elif self.interpolation_method == "spline":
            assert (
                waypoint_samples is not None
            ), "waypoint_samples can't be None with interpolation method spline!"
            assert (
                num_path_nodes is not None
            ), "num_path_nodes can't be None with interpolation method spline!"
            interpolated_path = self.interpolation_head(waypoint_samples, num_path_nodes)
        else:
            raise Exception("Unknown interpolation method", self.interpolation_method)

        return interpolated_path


class WaypointPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        width: int = 320,
        height: int = 224,
        n_conv: int = 4,
        combine_at: int = 2,
        command_information: str = "channel_attention",
        path_length: int = 20,
        norm_feats: bool = False,
        gt_sigma: float = 4.0,
        num_waypoints=4,
        shared_command_fusion=True,
        kernel_size=1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.width = width
        self.height = height
        self.n_conv = n_conv
        self.combine_at = combine_at
        self.path_length = path_length
        self.norm_feats = norm_feats
        self.gt_sigma = gt_sigma

        self.fpn = resnet_fpn_backbone(
            pretrained=False,
            norm_layer=nn.BatchNorm2d,
            trainable_layers=5,
            in_channels=self.in_channels,
        )

        self.fpn_strides = [4, 8, 16, 32]

        self.waypoints_head = PTPCHeadComm(
            64,
            self.n_conv,
            self.combine_at,
            num_scales=len(self.fpn_strides),
            command_information=command_information,
            orig_img_size=[height, width],
            norm_feats=norm_feats,
            num_waypoints=num_waypoints,
            shared_command_fusion=shared_command_fusion,
            kernel_size=kernel_size
        )
        self.image_dim = torch.tensor([self.height, self.width]).to(torch.float)

    def train(self, mode=True):
        super().train(mode)

        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

        self.apply(freeze_bn)

    def forward(
        self,
        input,
        command,
        gt_path,
        component_topk=0,
    ):
        mus, sigmas, pis, locations, separate_pis, separate_mus, separate_sigmas, features = self.generate_waypoint_mix_params(
            input, command, component_topk=component_topk, return_separate_pis=True
        )

        B, N_gt, num_path_nodes, _ = gt_path.shape

        return mus, sigmas, pis, locations, separate_pis, separate_mus, separate_sigmas, features

    def generate_waypoint_mix_params(
        self, input, command, component_topk=0, return_separate_pis=False
    ):
        features = self.fpn(input)
        features = [features[key] for key in features.keys() if key != "pool"]
        # mus, sigmas, pis = self.head(features)
        locations = self.compute_location(features)
        features = list(reversed(features))

        mus, sigmas, pis = self.waypoints_head(features, command)
        if return_separate_pis:
            separate_pis = pis
            separate_mus = mus
            separate_sigmas = sigmas

        mus, sigmas, pis, _ = self.prepare_outputs(
            mus, sigmas, pis, locations, component_topk
        )

        if return_separate_pis:
            return mus, sigmas, pis, locations, separate_pis, separate_mus, separate_sigmas, features
        else:
            return mus, sigmas, pis, locations

    def generate_waypoint_mix(
        self, input, command, component_topk=0
    ):
        (mus, sigmas, pis, location,) = self.generate_waypoint_mix_params(
            input, command, component_topk=component_topk, return_separate_pis=False
        )

        comp = D.Independent(D.Normal(loc=mus, scale=sigmas), 1)
        waypoint_mix = D.MixtureSameFamily(D.Categorical(logits=pis), comp)
        return waypoint_mix

    def compute_waypoints_feasibility(self, waypoint_samples, layout, command):
        """
        For each set of waypoints (one set per each trajectory), compute feasibility, one scalar
        Args:
            waypoint_samples: Waypoints for several paths [B, num_samples, num_waypoints, 2]
            layout: Layput [B, num_channels, H, W] - not necessarily used - alternativelly, use output features from waypoint head
            command: Command [B, command_dim] - not necessarily used - alternativelly, use output features from waypoint head
        Returns:
            waypoints_feaasibility: Feasibility for each set of waypoint nodes [B, num_paths]
        """
        pass

    def prepare_outputs(self, mus, sigmas, pis, locations, component_topk):
        mu_agg = []
        sigma_agg = []
        pi_agg = []
        scale_ind_agg = []
        num_waypoints = pis[0].shape[1]
        for i, (mu, sigma, pi, location) in enumerate(zip(mus, sigmas, pis, locations)):
            [B, _, H, W] = mu.shape

            # Maybe first do (location+mu) and then / self.image_dim
            locs = (
                location.t()
                .unsqueeze(0)
                .view(1, 2, H, W)
                .repeat(1, num_waypoints, 1, 1)
                .view(1, 2 * num_waypoints, H * W)
            )
            mu = (locs + mu.view(B, -1, H * W)) / self.image_dim.to(location).unsqueeze(
                0
            ).repeat(1, num_waypoints).unsqueeze(-1)
            mu = mu.view(B, 2 * num_waypoints, -1).permute(0, 2, 1)
            sigma = sigma.view(B, 2 * num_waypoints, -1).permute(0, 2, 1)
            pi = pi.view(B, num_waypoints, -1).permute(0, 2, 1)
            scale_ind = i * torch.ones_like(pi)

            mu_agg.append(mu)
            sigma_agg.append(sigma)
            pi_agg.append(pi)
            scale_ind_agg.append(scale_ind)

        mu_agg = torch.cat(mu_agg, dim=1)
        sigma_agg = torch.cat(sigma_agg, dim=1)
        pi_agg = torch.cat(pi_agg, dim=1)
        scale_ind_agg = torch.cat(scale_ind_agg, dim=1)

        mu_agg = list(mu_agg.chunk(num_waypoints, -1))
        sigma_agg = list(sigma_agg.chunk(num_waypoints, -1))
        pi_agg = list(pi_agg.chunk(num_waypoints, -1))

        if component_topk > 0:
            for i in range(num_waypoints):
                pi_agg[i], active_inds = pi_agg[i].topk(
                    min(component_topk, pi_agg[i].shape[1]), dim=1
                )
                inds = active_inds.repeat(1, 1, 2)
                mu_agg[i] = torch.gather(mu_agg[i], 1, inds)
                sigma_agg[i] = torch.gather(sigma_agg[i], 1, inds)

        mu_agg = torch.stack(mu_agg, 1)
        sigma_agg = torch.stack(sigma_agg, 1)
        pi_agg = torch.stack(pi_agg, 1)
        pi_agg = pi_agg.squeeze(-1)  # .softmax(-1)

        return mu_agg, sigma_agg, pi_agg, scale_ind_agg

    def compute_location(self, features):
        locations = []

        for i, feat in enumerate(features):
            _, _, height, width = feat.shape
            location_per_level = self.compute_location_per_level(
                height, width, self.fpn_strides[i], feat.device
            )
            locations.append(location_per_level)

        return list(reversed(locations))

    def compute_location_per_level(self, height, width, stride, device):
        shift_x = torch.arange(
            0, width * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y = torch.arange(
            0, height * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        location = torch.stack((shift_y, shift_x), 1) + stride // 2
        return location
