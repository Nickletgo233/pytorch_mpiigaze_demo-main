import logging
from typing import List

import numpy as np
import torch
from omegaconf import DictConfig

from common import Camera, Face, FacePartsName
from head_pose_estimation import HeadPoseNormalizer, LandmarkEstimator
from models import create_model
from transforms import create_transform
from utils import get_3d_face_model
import torchsummary
from fvcore.nn import FlopCountAnalysis
from models.L2CS.L2CS import L2CS
import torchvision
import torch.nn as nn

logger = logging.getLogger(__name__)


class GazeEstimator:
    """
        MPIIGaze：基于双眼图像的视线估计。
        MPIIFaceGaze：基于全脸图像的视线估计。
        ETH-XGaze：基于全脸图像的高精度视线估计。
    """
    EYE_KEYS = [FacePartsName.REYE, FacePartsName.LEYE]

    def __init__(self, config: DictConfig):
        ""
        self._config = config

        self._face_model3d = get_3d_face_model(config)

        self.camera = Camera(config.gaze_estimator.camera_params)
        self._normalized_camera = Camera(
            config.gaze_estimator.normalized_camera_params)

        self._landmark_estimator = LandmarkEstimator(config)  # 估计人脸关键点
        self._head_pose_normalizer = HeadPoseNormalizer(
            self.camera, self._normalized_camera,
            self._config.gaze_estimator.normalized_camera_distance)
        if self._config.mode == 'L2CS':
            self._gaze_estimation_model = self._load_L2CS_model()
        else:
            self._gaze_estimation_model = self._load_model()
        self._transform = create_transform(config)

    def _load_model(self) -> torch.nn.Module:
        model = create_model(self._config)
        checkpoint = torch.load(self._config.gaze_estimator.checkpoint,
                                map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.to(torch.device(self._config.device))
        model.eval()
        # try:
            # torchsummary.summary(model, [3, 224, 224])  # 计算模型结构
            # input_tensor = torch.randn(1, 3, 224, 224)
            # flops = FlopCountAnalysis(model, input_tensor)
            # print(f"Total FLOPs: {flops.total() / 1e6:.2f}M")
        # except:
        #     pass

        return model

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        """检测人脸得到关键点"""
        return self._landmark_estimator.detect_faces(image)

    def estimate_gaze(self, image: np.ndarray, face: Face) -> None:
        # 计算头部姿态和3D关键点
        R, T = self._face_model3d.estimate_head_pose(face, self.camera)  # 得到s0->s1的R1T1矩阵
        self._face_model3d.compute_3d_pose(face)
        face_center = self._face_model3d.compute_face_eye_centers(face, self._config.mode)

        if self._config.mode == 'MPIIGaze':
            for key in self.EYE_KEYS:  # 处理双眼
                eye = getattr(face, key.name.lower())
                self._head_pose_normalizer.normalize(image, eye)  # 处理双眼
            self._run_mpiigaze_model(face)
        elif self._config.mode == 'MPIIFaceGaze':
            self._head_pose_normalizer.normalize(image, face)  # 归一化全脸
            self._run_mpiifacegaze_model(face)
        elif self._config.mode == 'ETH-XGaze':
            self._head_pose_normalizer.normalize(image, face)  # 归一化全脸
            self._run_ethxgaze_model(face)
        elif self._config.mode == 'L2CS':
            self._head_pose_normalizer.normalize(image, face)  # 归一化全脸
            self._run_L2CS_model(face)
        else:
            raise ValueError
        # print(R, T)
        return R, T, face_center
    @torch.no_grad()
    def _run_mpiigaze_model(self, face: Face) -> None:
        images = []
        head_poses = []
        for key in self.EYE_KEYS:  # 遍历左右眼
            eye = getattr(face, key.name.lower())
            image = eye.normalized_image  # 归一化后的眼睛图像
            normalized_head_pose = eye.normalized_head_rot2d  # 归一化后的头部姿态
            if key == FacePartsName.REYE:  # 右眼特殊处理（水平翻转）
                image = image[:, ::-1].copy()
                normalized_head_pose *= np.array([1, -1])
            image = self._transform(image)
            images.append(image)
            head_poses.append(normalized_head_pose)
        images = torch.stack(images)
        head_poses = np.array(head_poses).astype(np.float32)
        head_poses = torch.from_numpy(head_poses)

        device = torch.device(self._config.device)
        images = images.to(device)
        head_poses = head_poses.to(device)
        predictions = self._gaze_estimation_model(images, head_poses)
        predictions = predictions.cpu().numpy()

        for i, key in enumerate(self.EYE_KEYS):
            eye = getattr(face, key.name.lower())
            eye.normalized_gaze_angles = predictions[i]
            if key == FacePartsName.REYE:
                eye.normalized_gaze_angles *= np.array([1, -1])  # 右眼角度调整
            eye.angle_to_vector()  # 角度转单位向量
            eye.denormalize_gaze_vector()  # 反归一化到原始坐标系

    @torch.no_grad()
    def _run_mpiifacegaze_model(self, face: Face) -> None:
        image = self._transform(face.normalized_image).unsqueeze(0)

        device = torch.device(self._config.device)
        image = image.to(device)
        prediction = self._gaze_estimation_model(image)
        prediction = prediction.cpu().numpy()
        print('pitch, yaw:', prediction[0])

        face.normalized_gaze_angles = prediction[0]
        face.angle_to_vector()
        face.denormalize_gaze_vector()

    @torch.no_grad()
    def _run_ethxgaze_model(self, face: Face) -> None:
        image = self._transform(face.normalized_image).unsqueeze(0)

        device = torch.device(self._config.device)
        image = image.to(device)
        prediction = self._gaze_estimation_model(image)
        prediction = prediction.cpu().numpy()

        face.normalized_gaze_angles = prediction[0]
        face.angle_to_vector()
        face.denormalize_gaze_vector()

    def _load_L2CS_model(self):
        # Save input parameters
        weights = 'C:\pixtalks\pytorch_mpiigaze_demo-main\L2CSNet_gaze360.pkl'
        device = 'cpu'

        # Create L2CS model
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], 90)
        model.load_state_dict(torch.load(weights, map_location=device))
        model.to(device)
        model.eval()

        return model


    def _run_L2CS_model(self, face: Face) -> None:
        self.softmax = nn.Softmax(dim=1)
        self.idx_tensor = [idx for idx in range(90)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor).to('cpu')

        image = self._transform(face.normalized_image).unsqueeze(0)

        device = torch.device(self._config.device)
        image = image.to(device)

        # Predict
        gaze_pitch, gaze_yaw = self._gaze_estimation_model(image)
        pitch_predicted = self.softmax(gaze_pitch)
        yaw_predicted = self.softmax(gaze_yaw)

        # Get continuous predictions in degrees.
        pitch_predicted = torch.sum(pitch_predicted.data * self.idx_tensor, dim=1) * 4 - 180
        yaw_predicted = torch.sum(yaw_predicted.data * self.idx_tensor, dim=1) * 4 - 180
        print('pitch, yaw:', pitch_predicted, yaw_predicted)

        pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
        yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0



        # print(pitch_predicted[0], yaw_predicted[0])
        # print(np.array([pitch_predicted[0], yaw_predicted[0]]), type(np.array([pitch_predicted[0], yaw_predicted[0]])))
        # for i, j in zip(pitch_predicted, yaw_predicted):
        #     print(type(i), type(j))

        # face.normalized_gaze_angles = np.array([pitch_predicted[0], yaw_predicted[0]])
        face.normalized_gaze_angles = np.array([yaw_predicted[0], pitch_predicted[0]])
        face.angle_to_vector()
        face.denormalize_gaze_vector()






