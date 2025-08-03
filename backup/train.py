import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import InterpolationNet
from dataset import PointCloudFromPLYDataset  # ← 수정된 Dataset 사용

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 여러 폴더에서 ply 데이터 불러오기
    ply_dirs = [
        '/home/nsl/lidar_train/raw/seq1/raycast/ouster',
        '/home/nsl/lidar_train/raw/seq1/raycast/livox',
        '/home/nsl/lidar_train/raw/seq1/raycast/hesai'
    ]
    dataset = PointCloudFromPLYDataset(ply_dirs, N_points=4096)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    model = InterpolationNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    # 체크포인트 저장 폴더 생성
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(1, 101):
        model.train()
        total_loss = 0.0

        loop = tqdm(dataloader, desc=f"Epoch {epoch}")
        for first, second in loop:
            first = first.to(device)    # (B, N, 3)
            second = second.to(device)  # (B, N, 3)

            pred = model(first)         # (B, N, 3)
            loss = criterion(pred, second)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch}] Average Loss: {avg_loss:.6f}")

        # 모델 저장
        if epoch % 10 == 0:
            ckpt_path = f'checkpoints/model_epoch{epoch}.pt'
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

if __name__ == '__main__':
    train()
