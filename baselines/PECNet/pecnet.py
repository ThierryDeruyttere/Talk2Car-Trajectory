import torch
import torch.nn as nn
from torch.nn import functional as F

from resnet import resnet
from utils_mask import create_neighborhood_mask


"""MLP model"""


class MLP(nn.Module):
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


class PECNet(nn.Module):
    def __init__(
        self,
        enc_layout_interm_size,
        enc_layout_latent_size,
        enc_layout_output_size,
        enc_command_latent_size,
        enc_command_output_size,
        enc_combined_latent_size,
        enc_combined_output_size,
        enc_dest_latent_size,
        enc_latent_size,
        dec_latent_size,
        predictor_latent_size,
        non_local_theta_size,
        non_local_phi_size,
        non_local_g_size,
        fdim,
        zdim,
        non_local_pools,
        non_local_dim,
        sigma,
        num_path_nodes,
        neighbor_dist_thresh,
        use_ref_obj=True,
        input_type="layout",
        layout_encoder_type="ResNet",
        verbose=False
    ):
        """
        Args:
            size parameters: Dimension sizes
            non_local_pools: Number of nonlocal pooling operations to be performed
            sigma: Standard deviation used for sampling N(0, sigma)
            past_length: Length of past history (number of timesteps)
            num_path_nodes: Length of future trajectory to be predicted
        """
        super(PECNet, self).__init__()

        self.zdim = zdim
        self.non_local_pools = non_local_pools
        self.sigma = sigma
        self.neighbor_dist_thresh = neighbor_dist_thresh

        self.layout_channels = 14  # 10 classes + egocar + 3 groundplan
        self.layout_loc_columns = 3
        if use_ref_obj:
            self.layout_channels += 1  # + referred
            self.layout_loc_columns += 1  # + referred

        assert input_type in {"locs", "layout"}, "Argument 'input_type' must be in  { 'locs', 'layout' }"
        self.input_type = input_type

        if input_type == "locs":
            self.encoder_layout = MLP(
                input_dim=64 * self.layout_loc_columns,
                hidden_size=enc_layout_latent_size,
                output_dim=enc_layout_output_size
            )
        else:
            if layout_encoder_type == "ResNet":
                layout_encoder_type = "ResNet-18"
            self.encoder_layout = resnet(
                layout_encoder_type,
                in_channels=self.layout_channels,
                num_classes=enc_layout_interm_size
            )

            self.encoder_layout = nn.Sequential(
                self.encoder_layout,
                nn.ReLU(),
                MLP(
                    input_dim=enc_layout_interm_size,
                    hidden_size=enc_layout_latent_size,
                    output_dim=enc_layout_output_size
                )
            )

        self.encoder_command = MLP(
            input_dim=768,
            hidden_size=enc_command_latent_size,
            output_dim=enc_command_output_size
        )
        self.encoder_combiner = MLP(
            input_dim=enc_layout_output_size + enc_command_output_size,
            hidden_size=enc_combined_latent_size,
            output_dim=enc_combined_output_size
        )
        self.encoder_dest = MLP(input_dim=2, hidden_size=enc_dest_latent_size, output_dim=fdim)

        self.encoder_latent = MLP(
            input_dim=enc_combined_output_size + fdim, hidden_size=enc_latent_size, output_dim=2 * zdim
        )
        self.decoder = MLP(input_dim=enc_combined_output_size + zdim, hidden_size=dec_latent_size, output_dim=2)  # layout + command + latent at input

        # Check social pooling
        self.non_local_theta = MLP(
            input_dim=enc_combined_output_size + fdim + 2,
            hidden_size=non_local_theta_size,
            output_dim=non_local_dim,
        )
        self.non_local_phi = MLP(
            input_dim=enc_combined_output_size + fdim + 2,
            hidden_size=non_local_phi_size,
            output_dim=non_local_dim,
        )
        self.non_local_g = MLP(
            input_dim=enc_combined_output_size + fdim + 2,
            hidden_size=non_local_g_size,
            output_dim=enc_combined_output_size + fdim + 2,
        )
        self.predictor = MLP(
            input_dim=enc_combined_output_size + fdim + 2,
            output_dim=2 * (num_path_nodes - 1),
            hidden_size=predictor_latent_size,
        )
        architecture = lambda net: [l.in_features for l in net.layers] + [
            net.layers[-1].out_features
        ]
        if verbose:
            print(
                "Layout Encoder architecture : {}".format(architecture(self.encoder_layout))
            )
            print(
                "Command Encoder architecture : {}".format(architecture(self.encoder_command))
            )
            print(
                "Combiner Encoder architecture : {}".format(architecture(self.encoder_combiner))
            )
            print(
                "Dest Encoder architecture : {}".format(architecture(self.encoder_dest))
            )
            print(
                "Latent Encoder architecture : {}".format(
                    architecture(self.encoder_latent)
                )
            )
            print("Decoder architecture : {}".format(architecture(self.decoder)))
            print("Predictor architecture : {}".format(architecture(self.predictor)))

            print(
                "Non Local Theta architecture : {}".format(
                    architecture(self.non_local_theta)
                )
            )
            print(
                "Non Local Phi architecture : {}".format(
                    architecture(self.non_local_phi)
                )
            )
            print(
                "Non Local g architecture : {}".format(architecture(self.non_local_g))
            )

    def non_local_social_pooling(self, feat, mask):
        # B - batch size, N - number of other objects, P - number of gt paths, C - feature dim
        # feat (B, (N + P), C)
        # mask (B, (N + P), (N + P))

        # B, (N + P), C
        theta_x = self.non_local_theta(feat)

        # B, C, (N + P)
        phi_x = self.non_local_phi(feat).permute(0, 2, 1)

        # f_ij = (theta_i)^T(phi_j)
        # (B, (N + P), (N + P))
        f = torch.matmul(theta_x, phi_x)

        # f_weights_i =  exp(f_ij)/(\sum_{j=1}^N exp(f_ij))
        f_weights = F.softmax(f, dim=-1)

        # setting weights of non neighbours to zero
        f_weights = f_weights * mask

        # rescaling row weights to 1
        f_weights = F.normalize(f_weights, p=1, dim=-1)

        # ith row of all_pooled_f = \sum_{j=1}^N f_weights_i_j * g_row_j
        pooled_f = torch.matmul(f_weights, self.non_local_g(feat))
        return pooled_f + feat

    def forward(self, layout, command, start_pos, other_obj_pos, dest=None):
        # layout (B, C, H, W) or (B, N * J)
        # command (B, 768)
        # dest (B, P, 2)
        # start_pos (B, P, 2)
        # other_obj_pos (B, N, 2)
        # mask (B, N + P, N + P)
        P = start_pos.shape[1]
        assert P == 1, "Only supports 1 gt. path per sample at a time."
        N = other_obj_pos.shape[1]
        # provide destination iff training
        # assert model.training
        assert self.training ^ (dest is None), "When training, destination needs to be provided."

        # encode
        layout_features = self.encoder_layout(layout).unsqueeze(1).repeat(1, P, 1)  # (B, P, layout_dim)
        command_features = self.encoder_command(command).unsqueeze(1).repeat(1, P, 1)  # (B, P, command_dim)
        combined_features = torch.cat((layout_features, command_features), dim=2)  # (B, P, combined_dim)
        combined_features = self.encoder_combiner(combined_features)

        if not self.training:
            z = torch.Tensor(layout.size(0), P, self.zdim).to(layout)
            z.normal_(0, self.sigma)
        else:
            # during training, use the destination to produce generated_dest and use it again to predict final future points

            # CVAE code
            dest_features = self.encoder_dest(dest)
            # dest_features - (B, P, fdim)

            features = torch.cat((combined_features, dest_features), dim=2)
            latent = self.encoder_latent(features)

            mu = latent[:, :, :self.zdim]  # 2-d array
            logvar = latent[:, :, self.zdim:]  # 2-d array

            var = logvar.mul(0.5).exp_()
            eps = torch.DoubleTensor(var.size()).normal_()
            eps = eps.to(layout)
            z = eps.mul(var).add_(mu)

        # z - (B, P, zdim)
        # combined_features - (B, P, combined_dim)

        decoder_input = torch.cat((combined_features, z), dim=2)
        # decoder_input - (B, P, combined_dim + zdim)
        generated_dest = self.decoder(decoder_input)
        # generated_dest - (B, P, 2)

        if self.training:
            # prediction in training, no best selection
            generated_dest_features = self.encoder_dest(generated_dest)
            # generated_dest_features - (B, P, 2)

            # combined_features (B, P, combined_dim)
            # command_features  (B, P, fdim)
            # generated_dest_features  (B, P, fdim)
            # start_pos (B, P, 2)
            # other_obj_pos (B, N, 2)

            all_obj_pos = torch.cat((start_pos, other_obj_pos), dim=1)
            # all_obj_pos (B, (N+P), 2)
            neighborhood_mask = create_neighborhood_mask(
                start_pos,
                other_obj_pos,
                neighbor_dist=self.neighbor_dist_thresh
            )
            # neighborhood_mask (B, (N+P), (N+P))

            combined_features = combined_features.repeat(1, N+P, 1)
            generated_dest_features = generated_dest_features.repeat(1, N+P, 1)

            prediction_features = torch.cat(
                (combined_features, generated_dest_features, all_obj_pos), dim=2
            )

            for i in range(self.non_local_pools):
                # non local social pooling
                prediction_features = self.non_local_social_pooling(
                    prediction_features, neighborhood_mask
                )

            generated_trajectory = self.predictor(prediction_features[:, :P])
            return generated_dest, mu, logvar, generated_trajectory
        return generated_dest

    # separated for forward to let choose the best destination
    def predict(self, layout, command, generated_dest, start_pos, other_obj_pos):
        P = start_pos.shape[1]
        assert P == 1, "Only supports 1 gt. path per sample at a time."
        N = other_obj_pos.shape[1]
        all_obj_pos = torch.cat((start_pos, other_obj_pos), dim=1)
        neighborhood_mask = create_neighborhood_mask(
            start_pos,
            other_obj_pos,
            neighbor_dist=self.neighbor_dist_thresh
        )

        layout_features = self.encoder_layout(layout)
        command_features = self.encoder_command(command)
        generated_dest_features = self.encoder_dest(generated_dest)

        layout_features = layout_features.unsqueeze(1).repeat(1, N + P, 1)
        command_features = command_features.unsqueeze(1).repeat(1, N + P, 1)
        combined_features = torch.cat((layout_features, command_features), dim=2)  # (B, P, combined_dim)
        combined_features = self.encoder_combiner(combined_features)

        generated_dest_features = generated_dest_features.repeat(1, N + P, 1)

        prediction_features = torch.cat(
            (combined_features, generated_dest_features, all_obj_pos), dim=2
        )
        for i in range(self.non_local_pools):
            # non local social pooling
            prediction_features = self.non_local_social_pooling(
                prediction_features, neighborhood_mask
            )

        generated_trajectory = self.predictor(prediction_features[:, :P])
        return generated_trajectory
