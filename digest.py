import numpy as np
import os
from tqdm import tqdm
from plyfile import PlyData

def read_ply_filtered(ply_path):
    ply = PlyData.read(ply_path)
    data = ply['vertex'].data

    x = data['x']
    y = data['y']
    z = data['z']
    intensity = data['intensity']
    ring = data['ring']
    returns = data['return']

    # 조건: return == 2.0
    mask = (returns == 2.0)
    if np.sum(mask) == 0:
        return None

    points = np.stack([x[mask], y[mask], z[mask], intensity[mask], ring[mask]], axis=-1).astype(np.float32)
    return points  # shape: (M, 5)

def convert_all_ply_dirs_filtered(base_dirs, save_root):
    os.makedirs(save_root, exist_ok=True)
    sample_idx = 0
    for dir_path in base_dirs:
        ply_files = sorted([f for f in os.listdir(dir_path) if f.endswith(".ply")])
        for f in tqdm(ply_files, desc=f"Processing {dir_path}"):
            ply_path = os.path.join(dir_path, f)

            points = read_ply_filtered(ply_path)
            if points is None:
                continue

            save_path = os.path.join(save_root, f"sample_{sample_idx:06d}.npz")
            np.savez(save_path, points=points)
            sample_idx += 1

# 예시 실행
base_dirs = [
    "/home/nsl/raw/seq1/raycast/livox",
    "/home/nsl/raw/seq1/raycast/ouster",
    "/home/nsl/raw/seq1/raycast/hesai"
]
save_dir = "./npz_seq1"
convert_all_ply_dirs_filtered(base_dirs, save_dir)