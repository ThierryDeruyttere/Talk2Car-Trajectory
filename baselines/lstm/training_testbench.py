import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from talk2car import Talk2Car_Detector
from modules import LSTM_Model


batch_size = 16
path_length = 20

lr = 1e-3
weight_decay = 0.0

embedding_dim = 64
hidden_dim = 512
use_ref_obj = True
loss_type = "L2"
unrolled = False
shuffle = True

gpu_index = 1
root = "/cw/liir/NoCsBack/testliir/thierry/PathProjection/data_root"
device = torch.device(
    'cuda', index=gpu_index
) if torch.cuda.is_available() else torch.device('cpu')

dset = Talk2Car_Detector(
    split="test",
    dataset_root=root,
    height=300,
    width=200,
    unrolled=unrolled,
    use_ref_obj=use_ref_obj,
    path_normalization="fixed_length",
    path_length=path_length
)

loader = DataLoader(
    dataset=dset,
    batch_size=batch_size,
    shuffle=shuffle
)

model = LSTM_Model(
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    use_ref_obj=use_ref_obj
).to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)

criterion = nn.MSELoss(reduction="none")
to_meters = torch.tensor([120.0, 80.0])

for epoch in range(20):
    total_loss = 0.0
    total_ade = 0.0
    total_ade_endpoint = 0.0
    for batch in tqdm(loader):
        layout, gt_path_nodes, start_pos, command_embedding = batch["x"].float(), batch["path"], batch["start_pos"], batch["command_embedding"]
        layout = layout.to(device)
        gt_path_nodes = gt_path_nodes.to(device)
        start_pos = start_pos.to(device)
        command_embedding = command_embedding.to(device)

        path_nodes = model(layout, start_pos[:, 0], command_embedding, n_predict=path_length)
        loss = criterion(path_nodes, gt_path_nodes)
        loss = loss.mean()
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        path_unnormalized = start_pos.unsqueeze(2) + path_nodes.cumsum(dim=3)  # (B, self.num_path_hyps, self.num_path_nodes, 2)
        gt_path_unnormalized = start_pos.unsqueeze(2) + gt_path_nodes.cumsum(dim=3)  # (B, self.num_path_hyps, self.num_path_nodes, 2)

        endpoint = path_unnormalized[:, :, -1, :]  # (B, 3, 1, 2)
        gt_endpoint = gt_path_unnormalized[:, :, -1, :]  # (B, 3, 1, 2)

        avg_distances = (
                (path_unnormalized - gt_path_unnormalized) * to_meters.to(gt_path_nodes)
        ).norm(2, dim=-1).mean(dim=-1).min(-1)[0]

        avg_distances_endpoint = (
                (endpoint - gt_endpoint) * to_meters.to(gt_endpoint)
        ).norm(2, dim=-1).min(-1)[0]

        ade = avg_distances.mean()
        ade = ade.mean()
        total_ade += ade.item()

        ade_endpoint = avg_distances_endpoint.mean()
        ade_endpoint = ade_endpoint.mean()
        total_ade_endpoint += ade_endpoint.item()

    print(f"Loss: {total_loss / len(loader)}")
    print(f"Ade: {total_ade / len(loader):.2f}m")
    print(f"ADE Endpoint: {total_ade_endpoint / len(loader):.2f}m")




