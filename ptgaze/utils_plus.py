import numpy as np
import cv2
import time
import math

from numba.cuda.libdevice import sqrtf


def coord_trans(src_coords, dst_coords, R, T, is_inv=0):
    # 验证输入维度
    src_coords = np.asarray(src_coords)
    R = np.asarray(R).reshape(3, 3)
    T = np.asarray(T)

    if src_coords.shape != (3,):
        raise ValueError("源坐标必须是长度为3的数组")
    if R.shape != (3, 3):
        raise ValueError("旋转矩阵必须是3x3的矩阵")
    if T.shape != (3,):
        raise ValueError("平移向量必须是长度为3的数组")
    # print(R, T)

    # 执行坐标转换
    if is_inv != 0:
        # 反向转换: R^(-1) * (src - T)
        dst_coords = np.dot(R.T, src_coords - T)
    else:
        # 正向转换: R * src + T
        dst_coords = np.dot(R, src_coords) + T

    return dst_coords

def uv_in_screen(p1_s0, p2_s0, p1_s2, p2_s2, u_vis, v_vis, u_world, v_world, R, T):
    p1_s2 = coord_trans(p1_s0, p1_s2, R, T)
    p2_s2 = coord_trans(p2_s0, p2_s2, R, T)
    # print(p1_s2)
    # print(p2_s2)
    x_screen = p1_s2[0] - (p1_s2[0] - p2_s2[0]) * p1_s2[2] / (p1_s2[2] - p2_s2[2])
    y_screen = p1_s2[1] - (p1_s2[1] - p2_s2[1]) * p1_s2[2] / (p1_s2[2] - p2_s2[2])
    # print('x y screen:', x_screen, y_screen)
    u = x_screen * u_vis / u_world + u_vis / 2
    v = y_screen * v_vis / v_world + v_vis / 2

    # clamped_x = max(0, min(u, u_vis - 1))
    # clamped_y = max(0, min(v, v_vis - 1))

    return u, v


def draw_fps(img, start_time, end_time, history):  # 计算 FPS（使用 OpenCV 高精度计时）
    # global history
    elapsed_time = (end_time - start_time) / cv2.getTickFrequency()
    current_fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0
    # 平滑 FPS
    history.append(current_fps)
    if len(history) > 20:
        history.pop(0)
    smoothed_fps = sum(history) / len(history)
    # 绘制 FPS
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.putText(
            img,f"FPS: {smoothed_fps:.2f}",(10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7,(0, 255, 0),2, cv2.LINE_AA
        )
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except:
        pass

    return img


def angles_to_vector(pitch: float, yaw: float) -> np.ndarray:
    """
    将俯仰角(pitch)和偏航角(yaw)转换为三维单位向量。

    参数:
        pitch: 俯仰角（绕X轴），单位为弧度
        yaw: 偏航角（绕Y轴），单位为弧度

    返回:
        三维单位向量 [x, y, z]
    """
    # 计算方向向量
    x = np.sin(yaw) * np.cos(pitch)
    y = np.sin(pitch)
    z = np.cos(yaw) * np.cos(pitch)

    # 归一化为单位向量
    vector = np.array([x, y, z])
    return vector / np.linalg.norm(vector)


def R21_to_R(R1, T1, R2, T2):
    R = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    T = [0, 0, 0]
    for i in range(3):
        T[i] = R2[i * 3] * T1[0] + R2[i * 3 + 1] * T1[1] + R2[i * 3 + 2] * T1[2] + T2[i]
        for j in range(3):
            R[i * 3 + j] = R2[i * 3] * R1[j] + R2[i * 3 + 1] * R1[j + 3] + R2[i * 3 + 2] * R1[j + 6]
    return R, T


def det_world_wh(inch, w, h):
    w, h = float(w), float(h)
    diagonal_pixels = math.hypot(w, h)
    unit_length = (inch * 25.4) / diagonal_pixels
    w_world = w * unit_length
    h_world = h * unit_length
    return w_world, h_world


def normalize_vector(vector):
    """归一化3维向量"""
    x, y, z = vector
    magnitude = math.sqrt(x ** 2 + y ** 2 + z ** 2)

    # 处理零向量情况
    if magnitude == 0:
        return (0.0, 0.0, 0.0)

    return [x / magnitude, y / magnitude, z / magnitude]


class Move_Filter():
    def __init__(self):
        self.acc_u = 0.0
        self.acc_v = 0.0
        self.thre =  20.0
        self.decay = 0.9

    def update(self, u, v):
        self.acc_u += u
        self.acc_v += v
        ret_u, ret_v = 0, 0
        m = math.sqrt(self.acc_u**2 + self.acc_v**2)

        if m > self.thre:
            r = self.thre / m
            ret_u = self.acc_u * (1 - r)
            ret_v = self.acc_v * (1 - r)
            self.acc_u *= r
            self.acc_v *= r

        self.acc_u *= self.decay
        self.acc_v *= self.decay
        # print(self.acc_u, u)

        return ret_u, ret_v


import numpy as np


def right_handed_euler_to_rotation(yaw, pitch, roll):
    """
    将Yaw、Pitch、Roll欧拉角转换为右手坐标系下的旋转矩阵（ZYX顺序）
    坐标系定义：x轴向左，y轴向下，z轴向前

    参数:
    - yaw: 偏航角（绕Z轴旋转），弧度制
    - pitch: 俯仰角（绕Y轴旋转），弧度制
    - roll: 翻滚角（绕X轴旋转），弧度制

    返回:
    - R: 3x3旋转矩阵
    """
    # 绕X轴旋转（Roll） - x轴向左，y轴向下，z轴向前
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],  # 右手系标准旋转（x轴向左）
        [0, np.sin(roll), np.cos(roll)]
    ])

    # 绕Y轴旋转（Pitch） - x轴向左，y轴向下，z轴向前
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],  # 右手系标准旋转（y轴向下）
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # 绕Z轴旋转（Yaw） - x轴向左，y轴向下，z轴向前
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],  # 右手系标准旋转（z轴向前）
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # 组合旋转矩阵（ZYX顺序）
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

yaw = np.pi
pitch = 0
roll = 0
R_ = right_handed_euler_to_rotation(yaw, pitch, roll)
print(R_.reshape(-1).tolist())