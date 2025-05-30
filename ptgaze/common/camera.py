import dataclasses
from typing import Optional

import cv2
import numpy as np
import yaml


@dataclasses.dataclass()
class Camera:
    width: int = dataclasses.field(init=False)
    height: int = dataclasses.field(init=False)
    camera_matrix: np.ndarray = dataclasses.field(init=False)  # 相机内参矩阵
    dist_coefficients: np.ndarray = dataclasses.field(init=False)  # 畸变系数

    camera_params_path: dataclasses.InitVar[str] = None

    def __post_init__(self, camera_params_path):
        with open(camera_params_path) as f:
            data = yaml.safe_load(f)
        self.width = data['image_width']
        self.height = data['image_height']
        self.camera_matrix = np.array(data['camera_matrix']['data']).reshape(
            3, 3)
        self.dist_coefficients = np.array(
            data['distortion_coefficients']['data']).reshape(-1, 1)

    def project_points(self,  # 将3D点投影到2D图像平面，考虑相机内参和畸变
                       points3d: np.ndarray,  # 待投影的3D点。
                       rvec: Optional[np.ndarray] = None,  # 旋转向量
                       tvec: Optional[np.ndarray] = None) -> np.ndarray:  # 平移向量
        assert points3d.shape[1] == 3
        if rvec is None:
            rvec = np.zeros(3, dtype=float)
        if tvec is None:
            tvec = np.zeros(3, dtype=float)
        points2d, _ = cv2.projectPoints(points3d, rvec, tvec,
                                        self.camera_matrix,  # 相机内参矩阵（3×3）
                                        self.dist_coefficients)  # 畸变系数
        return points2d.reshape(-1, 2)
