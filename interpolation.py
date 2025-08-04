#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import numpy as np
import open3d as o3d
import torch
from pcn_model import PCN  # 모델 경로 확인

# Parameters
VOXEL_SIZE = 0.03
MAX_DISTANCE = 5.0
INPUT_NUM = 512
OUTPUT_NUM = 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Publisher
pub_interp = None

# Load PCN model
model = PCN()
model.load_state_dict(torch.load("/home/nsl/lidar_train/pcn_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

def preprocess_points(msg):
    points = np.array([
        [p[0], p[1], p[2]]
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    ], dtype=np.float32)

    if points.shape[0] == 0:
        rospy.logwarn("No points received.")
        return None

    dist = np.linalg.norm(points, axis=1)
    mask = dist <= MAX_DISTANCE
    points = points[mask]

    if points.shape[0] == 0:
        rospy.logwarn("All points filtered by distance.")
        return None

    # voxel downsampling
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_down = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    points = np.asarray(pcd_down.points)

    # uniform sampling
    if points.shape[0] >= INPUT_NUM:
        indices = np.random.choice(points.shape[0], INPUT_NUM, replace=False)
    else:
        indices = np.random.choice(points.shape[0], INPUT_NUM, replace=True)

    return points[indices]

def inference_pcn(input_points):
    with torch.no_grad():
        input_tensor = torch.tensor(input_points, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(DEVICE)
        coarse = model(input_tensor)
        pred = coarse.squeeze(0).permute(1, 0).cpu().numpy()  # (2048, 3)
    return pred


def publish_pointcloud(points, header, topic, color=[0, 255, 0]):
    r, g, b = color
    rgb = (int(r) << 16) | (int(g) << 8) | int(b)
    rgb_float = np.frombuffer(np.uint32(rgb).tobytes(), dtype=np.float32)[0]

    points_rgb = [[x, y, z, rgb_float] for x, y, z in points]

    pc2_msg = pc2.create_cloud(
        header,
        fields=[
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.FLOAT32, 1),
        ],
        points=points_rgb
    )
    pub_interp.publish(pc2_msg)

def callback_2nd_return(msg):
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = msg.header.frame_id

    input_pts = preprocess_points(msg)
    if input_pts is None:
        return

    pred_pts = inference_pcn(input_pts)
    publish_pointcloud(pred_pts, header, "/pcn_interpolated_points2", color=[0, 255, 0])
    rospy.loginfo(f"[PCN] Interpolated {INPUT_NUM} → {OUTPUT_NUM} points and published.")

def listener():
    global pub_interp
    rospy.init_node("pcn_interpolation_node", anonymous=True)
    rospy.Subscriber("/ouster/points2", PointCloud2, callback_2nd_return)
    pub_interp = rospy.Publisher("/pcn_interpolated_points2", PointCloud2, queue_size=1)
    rospy.loginfo("PCN Interpolation node started, listening to /ouster/points2")
    rospy.spin()

if __name__ == '__main__':
    listener()
