import os
import numpy as np
import torch
from torch.utils.data import Dataset
from plyfile import PlyData

class PointCloudFromPLYDataset(Dataset):
    def __init__(self, root_dirs, N_points=4096, normalize=False):
        self.N = N_points
        self.normalize = normalize
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

        # normalize 시 빈 배열 방어 처리 포함
        if self.normalize:
            first_xyz = self._normalize_coords(first_xyz)
            second_xyz = self._normalize_coords(second_xyz)

        first = self._pad_or_crop(first_xyz, self.N)
        second = self._pad_or_crop(second_xyz, self.N)

        return torch.from_numpy(first), torch.from_numpy(second)

    def _normalize_coords(self, coords):
        if coords.shape[0] == 0:
            print("[경고] normalize: coords가 비어 있음. 그대로 반환합니다.")
            return coords
        mean = coords.mean(axis=0)
        std = coords.std(axis=0) + 1e-6  # 분산 0 방지
        return (coords - mean) / std

    def _pad_or_crop(self, arr, n):
        length = len(arr)
        if length >= n:
            idxs = np.random.choice(length, n, replace=False)
            return arr[idxs].astype(np.float32)
        else:
            pad = np.zeros((n, 3), dtype=np.float32)
            pad[:length] = arr
            return pad

    def _read_and_split_ply(self, file_path):
        plydata = PlyData.read(file_path)
        vertex = plydata['vertex'].data

        coords = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)

        if 'return' in vertex.dtype.names:
            returns = vertex['return']
        elif 'returns' in vertex.dtype.names:
            returns = vertex['returns']
        else:
            raise ValueError("PLY 파일에 'return' 필드가 없습니다.")

        first = coords[returns == 1]
        second = coords[returns == 2]

        if len(first) == 0:
            print(f"[경고] {os.path.basename(file_path)}: return==1 포인트 없음")
        if len(second) == 0:
            print(f"[경고] {os.path.basename(file_path)}: return==2 포인트 없음")

        return first.astype(np.float32), second.astype(np.float32)
