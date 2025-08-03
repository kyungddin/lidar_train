import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PointCloudFromPLYDataset(Dataset):
    def __init__(self, ply_dirs, N_points=4096, normalize=True):
        self.ply_files = []
        for d in ply_dirs:
            self.ply_files += [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.ply')]
        self.N_points = N_points
        self.normalize = normalize

    def __len__(self):
        return len(self.ply_files)

    def __getitem__(self, idx):
        ply_path = self.ply_files[idx]

        # PLY 파일 읽기 (ascii 포맷 기준, 헤더를 건너뛰고 필요한 컬럼만 읽음)
        with open(ply_path, 'r') as f:
            lines = f.readlines()

        # 헤더 끝 위치 찾기
        end_header_idx = 0
        for i, line in enumerate(lines):
            if line.strip() == "end_header":
                end_header_idx = i
                break

        # 데이터 부분 읽기 (헤더 이후부터 끝까지)
        data_lines = lines[end_header_idx+1:]
        data = []
        for line in data_lines:
            vals = line.strip().split()
            # x,y,z=0,1,2 / nx,ny,nz=3,4,5 / intensity=6 / ring=7 / return=8 / label=9
            # 본 예시 ply는 property 순서가 다를 수 있으니 맞춰야 함!
            # 여기서는 user가 준 ply 헤더 기준 (x,y,z,nx,ny,nz,intensity,ring,return,label)
            x = float(vals[0])
            y = float(vals[1])
            z = float(vals[2])
            intensity = float(vals[6])
            ring = float(vals[7])
            ret = int(float(vals[8]))

            if ret == 2:
                data.append([x, y, z, intensity, ring])

        data = np.array(data, dtype=np.float32)

        # N_points 개수만큼 샘플링
        if len(data) >= self.N_points:
            indices = np.random.choice(len(data), self.N_points, replace=False)
        else:
            indices = np.random.choice(len(data), self.N_points, replace=True)
        sampled = data[indices]

        if self.normalize:
            # 각 컬럼별 평균, std 계산 (ring은 범주형처럼 작동할 수도 있으니 별도 처리 가능)
            mean = sampled.mean(axis=0)
            std = sampled.std(axis=0) + 1e-8
            sampled = (sampled - mean) / std

        return torch.tensor(sampled, dtype=torch.float32)
