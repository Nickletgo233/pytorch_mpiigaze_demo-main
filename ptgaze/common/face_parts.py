import enum  # 创建枚举库
from typing import Optional  # 明确标注变量、函数参数或返回值可能为None

import numpy as np
from scipy.spatial.transform import Rotation


class FacePartsName(enum.Enum):
    FACE = enum.auto()  # 自动生成值1.2.3.4这样
    REYE = enum.auto()
    LEYE = enum.auto()


class FaceParts:
    def __init__(self, name: FacePartsName):
        self.name = name
        self.center: Optional[np.ndarray] = None
        self.head_pose_rot: Optional[Rotation] = None
        self.normalizing_rot: Optional[Rotation] = None
        self.normalized_head_rot2d: Optional[np.ndarray] = None
        self.normalized_image: Optional[np.ndarray] = None

        self.normalized_gaze_angles: Optional[np.ndarray] = None
        self.normalized_gaze_vector: Optional[np.ndarray] = None
        self.gaze_vector: Optional[np.ndarray] = None

    @property
    def distance(self) -> float:
        return np.linalg.norm(self.center)

    def angle_to_vector(self) -> None:
        pitch, yaw = self.normalized_gaze_angles  # 俯仰角（pitch） 和 偏航角（yaw）
        self.normalized_gaze_vector = -np.array([  # 球坐标系转直角坐标系公式
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
            np.cos(pitch) * np.cos(yaw)
        ])

    def denormalize_gaze_vector(self) -> None:  # 反归一化向量
        normalizing_rot = self.normalizing_rot.as_matrix()
        # print('normalizing_rot:', normalizing_rot)
        # Here gaze vector is a row vector, and rotation matrices are
        # orthogonal, so multiplying the rotation matrix from the right is
        # the same as multiplying the inverse of the rotation matrix to the
        # column gaze vector from the left.
        # 这里凝视向量是行向量，旋转矩阵是正交的，所以从右乘旋转矩阵等于从左乘旋转矩阵的逆乘列凝视向量。
        self.gaze_vector = self.normalized_gaze_vector @ normalizing_rot  # 最终的输出的视线向量

    @staticmethod
    def vector_to_angle(vector: np.ndarray) -> np.ndarray:
        assert vector.shape == (3, )
        x, y, z = vector
        pitch = np.arcsin(-y)
        yaw = np.arctan2(-x, -z)
        # print('pitch:', pitch, 'yaw:', yaw)
        return np.array([pitch, yaw])
