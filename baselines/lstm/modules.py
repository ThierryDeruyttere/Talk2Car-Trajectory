import torch
from torch import nn
from torch.nn import init
from resnet import resnet


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=None):
    if padding is None:
        padding_inside = (kernel_size - 1) // 2
    else:
        padding_inside = padding
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding_inside,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding_inside,
                bias=True,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )



class FlowNetS(nn.Module):
    def __init__(
        self, input_channels=5, batchNorm=True, input_width=320, input_height=576
    ):
        super(FlowNetS, self).__init__()

        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, input_channels, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)
        self.conv7 = conv(
            self.batchNorm, 1024, 1024, kernel_size=1, stride=1, padding=0
        )
        self.conv8 = conv(
            self.batchNorm, 1024, 1024, kernel_size=1, stride=1, padding=0
        )
        fc1_input_features = 40960  # (input_height * input_width) // 4 # 98304
        self.fc1 = nn.Linear(
            in_features=fc1_input_features, out_features=1024, bias=True
        )
        self.fc2 = nn.Linear(in_features=1024, out_features=1024, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        out_conv8 = self.conv8(self.conv7(out_conv6))  # SDD
        out_fc1 = nn.functional.relu(
            self.fc1(out_conv8.view(out_conv6.size(0), -1))
        )  # SDD
        out_fc2 = nn.functional.relu(self.fc2(out_fc1))
        return out_fc2


class InputEmbedding(nn.Module):
    """Linear embedding, ReLU non-linearity, input scaling.

    Input scaling is important for ReLU activation combined with the initial
    bias distribution. Otherwise some units will never be active.
    """
    def __init__(self, input_dim, embedding_dim, scale=4.0, preserve_specials=False):
        super(InputEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.preserve_specials = preserve_specials

        linear_embedding_dim = self.embedding_dim
        if preserve_specials:
            linear_embedding_dim -= 2
        self.input_embeddings = nn.Sequential(
            nn.Linear(input_dim, linear_embedding_dim),
            nn.ReLU(),
        )

    def forward(self, vel):
        x = self.input_embeddings(vel * self.scale)
        if self.preserve_specials:
            if len(vel.shape) == 3:
                vel = vel[:, :, 2:]
            else:
                vel = vel[:, 2:]
            x = torch.cat([x, vel], dim=2)
        return x


class InputAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(InputAttention, self).__init__()
        self.att_head = nn.MultiheadAttention(embedding_dim, num_heads=1)

    def forward(self, x, key_padding_mask):
        x = x.transpose(1, 0)
        attn_output, attn_output_weights = self.att_head(x, x, x, key_padding_mask=key_padding_mask)
        return attn_output.sum(dim=0)


class Hidden2Normal(nn.Module):
    def __init__(self, hidden_dim):
        super(Hidden2Normal, self).__init__()
        self.linear = nn.Linear(hidden_dim, 5)

    def forward(self, hidden_state):
        normal = self.linear(hidden_state)

        # numerically stable output ranges
        normal[:, 2] = 0.01 + 0.2 * torch.sigmoid(normal[:, 2])  # sigma 1
        normal[:, 3] = 0.01 + 0.2 * torch.sigmoid(normal[:, 3])  # sigma 2
        normal[:, 4] = 0.7 * torch.sigmoid(normal[:, 4])  # rho

        return normal


class Hidden2Coords(nn.Module):
    def __init__(self, hidden_dim):
        super(Hidden2Coords, self).__init__()
        self.linear = nn.Linear(hidden_dim, 2)

    def forward(self, hidden_state):
        coords = self.linear(hidden_state)
        return coords


class LSTM(torch.nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=128):
        """ Initialize the LSTM forecasting model

        Attributes
        ----------
        embedding_dim : Embedding dimension of location coordinates
        hidden_dim : Dimension of hidden state of LSTM
        pool : interaction module
        pool_to_input : Bool
            if True, the interaction vector is concatenated to the input embedding of LSTM [preferred]
            if False, the interaction vector is added to the LSTM hidden-state
        """

        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        scale = 4.0
        self.input_embedding = InputEmbedding(2, self.embedding_dim, scale, preserve_specials=False)
        self.lstm = torch.nn.LSTMCell(self.embedding_dim, self.hidden_dim)
        self.hidden2coords = Hidden2Coords(self.hidden_dim)

    def step(self, lstm, hidden_cell_state, inp):
        """Do one step of prediction: two inputs to one normal prediction.

        Parameters
        ----------
        lstm: torch nn module [Encoder / Decoder]
            The module responsible for prediction
        hidden_cell_state : tuple (hidden_state, cell_state)
            Current hidden_cell_state of the pedestrians
        inp : Tensor [batch_size, 2]
            Input

        Returns
        -------
        hidden_cell_state : tuple (hidden_state, cell_state)
            Updated hidden_cell_state of the pedestrians
        normals : Tensor [batch_size, 5]
            Parameters of a multivariate normal of the predicted position
            with respect to the current position
        """
        input_emb = self.input_embedding(inp)
        ## Masked Hidden Cell State
        hidden_cell_state = [
            torch.stack([h for h in hidden_cell_state[0]], dim=0),
            torch.stack([c for c in hidden_cell_state[1]], dim=0),
        ]

        # LSTM step
        hidden_cell_state = lstm(input_emb, hidden_cell_state)
        coords = self.hidden2coords(hidden_cell_state[0])

        return hidden_cell_state, coords

    def forward(self, hidden_init, start_inp, n_predict=20):
        """Forecast the entire sequence

        Parameters
        ----------
        hidden_init : Tuple ( Tensor [batch_size, hidden_dim] )
            Initial hidden state - given from encoder
        start_inp : Tensor [batch_size, 2]
            Tensor defining the split of the batch.
            Required to identify the tracks of to the same scene
        n_predict: Int
            Length of sequence to be predicted during test time

        Returns
        -------
        path_coords : Tensor [batch_size, num_preds, 2]
            Predicted positions relative to previous positions
        """

        # list of predictions
        path_coords = []  # predicted normal parameters for both phases

        hidden_cell_state = hidden_init
        inp = start_inp
        for i in range(n_predict):
            hidden_cell_state, coords = self.step(self.lstm, hidden_cell_state, inp)
            path_coords.append(coords.unsqueeze(1))
        path_coords = torch.stack(path_coords, dim=1)
        return path_coords


class LSTM_Model(torch.nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=512, use_ref_obj=True):
        """ Initialize the LSTM forecasting model

        Attributes
        ----------
        embedding_dim : Embedding dimension of location coordinates
        hidden_dim : Dimension of hidden state of LSTM
        pool : interaction module
        pool_to_input : Bool
            if True, the interaction vector is concatenated to the input embedding of LSTM [preferred]
            if False, the interaction vector is added to the LSTM hidden-state
        """

        super(LSTM_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.use_ref_obj = use_ref_obj

        self.lstm = torch.nn.LSTMCell(self.embedding_dim, self.hidden_dim)

        self.input_channels = 14  # 10 classes + egocar + 3 groundplan

        if self.use_ref_obj:
            self.input_channels += 1

        self.encoder = resnet(
            "ResNet-18",
            in_channels=self.input_channels,
            num_classes=self.hidden_dim
        )
        self.command_encoder = nn.Linear(768, self.hidden_dim)
        # self.combiner = nn.Sequential(
        #     nn.Linear(self.hidden_dim + 768, 1024),
        #     # nn.BatchNorm1d(num_features=1024),
        #     nn.ReLU(),
        #     # nn.Dropout(),
        #     nn.Linear(1024, 512),
        #     # nn.BatchNorm1d(num_features=512),
        #     nn.ReLU(),
        #     # nn.Dropout(),
        # )
        self.lstm = LSTM(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
        )

    def forward(self, layout, start_positions, command_embedding, n_predict=20):
        all_position_embedding = self.encoder(layout)
        command_embedding = self.command_encoder(command_embedding)

        # Put position embeddings into hidden state init, and command embedding into cell state init
        hidden_init = [
            all_position_embedding,
            command_embedding,
        ]

        path_coords = self.lstm(hidden_init, start_positions, n_predict=n_predict).transpose(2, 1)
        return path_coords