import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from chamferdist import ChamferDistance

class PCN(nn.Module):
    def __init__(self, num_dense=2048):
        super().__init__()
        self.num_dense = num_dense

        # Encoder
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(256, 512, 1)

        # Decoder
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        self.fold1 = nn.Sequential(
            nn.Linear(512 + 2, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )
        self.fold2 = nn.Sequential(
            nn.Linear(512 + 3, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def build_grid(self, batch_size, device):
        grid_size = int(np.ceil(np.sqrt(self.num_dense)))
        linspace = torch.linspace(-0.3, 0.3, grid_size, device=device)
        grid_x, grid_y = torch.meshgrid(linspace, linspace, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)[:self.num_dense]
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, num_dense, 2)
        return grid

    def forward(self, x):  # x: (B, 3, N)
        B = x.size(0)
        feat = F.relu(self.conv1(x))
        feat = F.relu(self.conv2(feat))
        feat = F.relu(self.conv3(feat))
        global_feat = torch.max(feat, 2)[0]  # (B, 512)

        codeword = F.relu(self.fc1(global_feat))
        codeword = F.relu(self.fc2(codeword))

        codeword_expand = global_feat.unsqueeze(1).repeat(1, self.num_dense, 1)  # (B, num_dense, 512)
        grid = self.build_grid(B, x.device)

        fold1_input = torch.cat([codeword_expand, grid], dim=2)
        fold1 = self.fold1(fold1_input)

        fold2_input = torch.cat([codeword_expand, fold1], dim=2)
        fold2 = self.fold2(fold2_input)
        return fold2


class PCNDataset(Dataset):
    def __init__(self, data_dir, num_sparse=512, num_dense=2048):
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
        self.num_sparse = num_sparse
        self.num_dense = num_dense
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        pts = data['points'][:, :3]
        center = pts.mean(axis=0)
        pts = pts - center
        scale = np.max(np.linalg.norm(pts, axis=1))
        pts = pts / scale if scale > 0 else pts

        N = pts.shape[0]
        dense_idx = self.rng.choice(N, self.num_dense, replace=N < self.num_dense)
        sparse_idx = self.rng.choice(N, self.num_sparse, replace=N < self.num_sparse)
        dense = pts[dense_idx]
        sparse = pts[sparse_idx]
        return torch.tensor(sparse, dtype=torch.float32), torch.tensor(dense, dtype=torch.float32)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PCNDataset('./npz_seq1', num_sparse=512, num_dense=2048)
    train_len = int(len(dataset) * 0.8)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, len(dataset) - train_len])
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=4)

    model = PCN(num_dense=2048).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = ChamferDistance()

    for epoch in range(1, 31):
        model.train()
        train_loss = 0
        for sparse, dense in tqdm(train_loader, desc=f"Epoch {epoch}"):
            sparse = sparse.to(device).permute(0, 2, 1)  # (B, 3, N)
            dense = dense.to(device)
            pred = model(sparse)
            loss = criterion(pred, dense) / sparse.size(0)  # Normalize by batch size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sparse, dense in val_loader:
                sparse = sparse.to(device).permute(0, 2, 1)
                dense = dense.to(device)
                pred = model(sparse)
                val_loss += (criterion(pred, dense) / sparse.size(0)).item()

        print(f"Epoch {epoch}: Train Loss {train_loss / len(train_loader):.4f}, Val Loss {val_loss / len(val_loader):.4f}")

    torch.save(model.state_dict(), "pcn_model.pth")
    print("Model saved as pcn_model.pth")


if __name__ == "__main__":
    train()
