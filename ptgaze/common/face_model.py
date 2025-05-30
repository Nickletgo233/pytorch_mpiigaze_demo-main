import dataclasses

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .camera import Camera
from .face import Face


@dataclasses.dataclass(frozen=True)
class FaceModel:
    LANDMARKS: np.ndarray
    REYE_INDICES: np.ndarray
    LEYE_INDICES: np.ndarray
    MOUTH_INDICES: np.ndarray
    NOSE_INDICES: np.ndarray
    CHIN_INDEX: int
    NOSE_INDEX: int

    def estimate_head_pose(self, face: Face, camera: Camera) -> None:
        """Estimate the head pose by fitting 3D template model."""
        """通过拟合3D模板模型估算头部姿态"""
        # If the number of the template points is small, cv2.solvePnP
        # becomes unstable, so set the default value for rvec and tvec
        # and set useExtrinsicGuess to True.
        # The default values of rvec and tvec below mean that the
        # initial estimate of the head pose is not rotated and the
        # face is in front of the camera.
        # 如果模板点数量较少，cv2.solvePnP变得不稳定，所以为rvec和tvec设置默认值，
        # 并将useExtrinsicGuess设置为True。下面的rvec和tvec的默认值表示头部位置的初始估计值
        rvec = np.zeros(3, dtype=float)
        tvec = np.array([0, 0, 1], dtype=float)
        # 通过最小化 3D 模板点与其在图像中对应 2D 关键点之间的重投影误差，求解头部的旋转矩阵（rvec）和平移向量（tvec）。
        _, rvec, tvec = cv2.solvePnP(self.LANDMARKS,  # 3D模板点（世界坐标系）
                                    face.landmarks,  # 2D检测点（图像坐标系）
                                    camera.camera_matrix,  # 相机内参矩阵
                                    camera.dist_coefficients,  # 畸变系数
                                    rvec,  # 初始旋转向量（用于迭代优化）
                                    tvec,  # 初始平移向量（用于迭代优化）
                                    useExtrinsicGuess=True,  # 使用初始值进行迭代优化
                                    flags=cv2.SOLVEPNP_ITERATIVE  # 使用迭代算法求解
        )
        rot = Rotation.from_rotvec(rvec)
        face.head_pose_rot = rot
        face.head_position = tvec
        face.reye.head_pose_rot = rot
        face.leye.head_pose_rot = rot
        R = np.round(rot.as_matrix(), 3).ravel()
        T = tvec
        # print('R:', R, 'T', T)
        return R, T

    def compute_3d_pose(self, face: Face) -> None:
        """Compute the transformed model."""
        rot = face.head_pose_rot.as_matrix()
        face.model3d = self.LANDMARKS @ rot.T + face.head_position

    def compute_face_eye_centers(self, face: Face, mode: str) -> None:
        """Compute the centers of the face and eyes.

        In the case of MPIIFaceGaze, the face center is defined as the
        average coordinates of the six points at the corners of both
        eyes and the mouth. In the case of ETH-XGaze, it's defined as
        the average coordinates of the six points at the corners of both
        eyes and the nose. The eye centers are defined as the average
        coordinates of the corners of each eye.
        """
        if mode == 'ETH-XGaze':
            face.center = face.model3d[np.concatenate(
                [self.REYE_INDICES, self.LEYE_INDICES,
                 self.NOSE_INDICES])].mean(axis=0)
        else:
            face.center = face.model3d[np.concatenate(
                [self.REYE_INDICES, self.LEYE_INDICES,
                 self.MOUTH_INDICES])].mean(axis=0)
        face.reye.center = face.model3d[self.REYE_INDICES].mean(axis=0)
        face.leye.center = face.model3d[self.LEYE_INDICES].mean(axis=0)
        # print('face center:', face.center)
        return face.center  # 此为相机坐标系下的人脸中心