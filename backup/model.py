import torch
import torch.nn as nn

class InterpolationNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 출력: 보간된 2nd return 좌표
        )

    def forward(self, x):  # x: (B, N, 3)
        B, N, _ = x.shape
        x = x.view(B * N, -1)
        out = self.mlp(x)
        out = out.view(B, N, 3)
        return out
