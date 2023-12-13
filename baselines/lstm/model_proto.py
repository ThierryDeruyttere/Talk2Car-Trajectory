import torch
import torch.nn as nn
from modules import LSTM, InputEmbedding, Hidden2Normal

from resnet import resnet
from loss import PredictionLoss, L2Loss

class LSTM_Model(torch.nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=512, use_ref_obj=True, loss_type="L2"):
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
        self.loss_type = loss_type

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
        self.criterion = L2Loss()

    def forward(self, layout, start_positions, command_embedding, n_predict=20):
        all_position_embedding = self.encoder(layout)
        command_embedding = self.command_encoder(command_embedding)

        # Put position embeddings into hidden state init, and command embedding into cell state init
        hidden_init = [
            all_position_embedding,
            command_embedding,
        ]

        path_coords = self.lstm(hidden_init, start_positions, n_predict=n_predict)
        return path_coords