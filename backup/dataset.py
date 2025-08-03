import os
import numpy as np
import torch
from torch.utils.data import Dataset
from plyfile import PlyData

class PointCloudFromPLYDataset(Dataset):
    def __init__(self, root_dirs, N_points=4096):
        self.N = N_points
        self.file_list = []
        for d in root_dirs:
            d = os.path.abspath(d)
            for f in os.listdir(d):
                if f.endswith('.ply'):
                    self.file_list.append(os.path.join(d, f))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        first_xyz, second_xyz = self._read_and_split_ply(path)

        first = self._pad_or_crop(first_xyz, self.N)
        second = self._pad_or_crop(second_xyz, self.N)

        # torch.Tensor로 변환
        first_tensor = torch.from_numpy(first)
        second_tensor = torch.from_numpy(second)

        return first_tensor, second_tensor

    def _pad_or_crop(self, arr, n):
        length = len(arr)
        if length >= n:
            # 무작위로 n개 샘플링 (replace=False: 중복 없이)
            idxs = np.random.choice(length, n, replace=False)
            sampled = arr[idxs]
            return sampled.astype(np.float32)
        else:
            # 부족할 땐 0으로 패딩
            pad = np.zeros((n, 3), dtype=np.float32)
            pad[:length] = arr
            return pad

    def _read_and_split_ply(self, file_path):
        # plyfile 라이브러리로 파싱 (헤더, 데이터 모두 안정적 처리)
        plydata = PlyData.read(file_path)
        vertex = plydata['vertex'].data

        # x, y, z 좌표 추출
        coords = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)

        # return 정보가 'return' 필드명이라고 가정
        # 실제 필드명은 ply 파일 구조에 따라 다를 수 있으니 확인 필요
        if 'return' in vertex.dtype.names:
            returns = vertex['return']
        elif 'returns' in vertex.dtype.names:
            returns = vertex['returns']
        else:
            raise ValueError("PLY 파일에 'return' 필드가 없습니다.")

        first = coords[returns == 1]
        second = coords[returns == 2]

        return first.astype(np.float32), second.astype(np.float32)
