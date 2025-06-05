import argparse
import cv2
import numpy as np
import os
import sys
from demo import Demo
from omegaconf import OmegaConf
import pathlib


def euler_to_rotation_matrix(angles):
    """
    Convert the Euler Angle into a rotation matrix
    Parameters：
    angles: A list or array containing the rotation angles around the x, y, and z axes
    return：
    Rotation matrix
    """
    alpha, beta, gamma = angles  # Obtain the rotation angles around the x, y, and z axes respectively
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    cos_b, sin_b = np.cos(beta), np.sin(beta)
    cos_g, sin_g = np.cos(gamma), np.sin(gamma)

    # Construct the rotation matrix
    matrix_x = np.array([[1, 0, 0],
                         [0, cos_a, -sin_a],
                         [0, sin_a, cos_a]])

    matrix_y = np.array([[cos_b, 0, sin_b],
                        [0, 1, 0],
                         [-sin_b, 0, cos_b]])

    matrix_z = np.array([[cos_g, -sin_g, 0],
                         [sin_g, cos_g, 0],
                         [0, 0, 1]])

    rotation_matrix = np.dot(np.dot(matrix_z, matrix_y), matrix_x)

    return rotation_matrix


def bk_img(u, v):
    bk = np.zeros(shape=(v, u, 3), dtype=np.uint8)
    return bk


def coord_rt(coord, rt):
    p1 = np.concatenate([coord, np.array([1], dtype=rt.dtype)], axis=0)
    p2 = np.matmul(rt, p1)
    # Or use the @ operator: P2 = M @ P1
    x2, y2, z2, _ = p2
    return [x2, y2, z2]


def coord2uv(coord, w_screen, h_screen, u_screen, v_screen, scale, center=(0.5, 0.5)):
    x, y, z = coord
    print(f'x:{x}, y:{y}')
    x = x * 1.1 + w_screen * center[0]
    y = -y * 1.1 + h_screen * center[1]
    u = x / w_screen * u_screen
    v = y / h_screen * v_screen
    return int(u), int(v)


def main():
    u_screen, v_screen = 1920, 1080
    w_screen, h_screen = 597, 336
    scale = 2
    center = [0.5, 0.5]
    R = np.array([-1.0, -1.2246467991473532e-16, 0.0, 1.2246467991473532e-16, -1.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
    T = np.array([-30.04763, 70.02763, 0.]).reshape(3, 1) * scale
    RT_3x4 = np.hstack((R, T))  # Concatenate R and T to obtain a 3x4 matrix
    RT_4x4 = np.vstack((RT_3x4, [0, 0, 0, 1]))  # Add homogeneous coordinate rows to obtain a 4x4 matrix
    RT_screen_camera = RT_4x4  # 4x4
    rt_4x4 = np.array(RT_screen_camera)
    bk = bk_img(u_screen, v_screen)

    package_root = pathlib.Path(__file__).parent.resolve()
    print('package_root:', package_root)
    config = OmegaConf.load(package_root / 'depth_estimation_mpiifacegaze.yaml')
    config.PACKAGE_ROOT = package_root.as_posix()  # 转斜杠\为/q
    # print(OmegaConf.to_yaml(config))  # 打印完整配置内容
    # print(config.keys())  # 查看配置顶层键
    demo = Demo(config)
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        demo._wait_key()
        ok, frame = cam.read()
        if not ok:
            break
        try:
            faces, vector, vector2, R1, T1, face_center = demo._process_image(frame)
            # print('demo.visualizer.image:', demo.visualizer.image.shape)
            for face in faces:
                landmark = face.landmarks
                # print(list(landmark[36]), landmark[39], type(landmark[36]))
                land, land2 = list(landmark[36]), list(landmark[45]) # 相机坐标系640x480 右上角为原点
                land = [int(i) for i in land]
                print(land)
                land2 = [int(i) for i in land2]
                land_s, land_s2 = [0, 0], [0, 0]


                for pt in [land, land2]:
                    # print(pt)
                    cv2.circle(frame, center=pt, radius=3, color=(0, 0, 255))
                    pt[0] = pt[0] - 320
                    pt[1] = pt[1] - 240
                # demo.visualizer.draw_points(landmark[36:39:3])

                for i, j in [0, 1], [1, 0]:
                    land_s[i] = land[j]
                    land_s2[i] = land2[j]
            land.append(1)
            # print(land)
            left_elm = coord_rt(land, rt_4x4)
            bk[:, :, :] = 0
            for pt in [left_elm]:
                u, v = coord2uv(pt, w_screen, h_screen, u_screen, v_screen, scale, center)
                cv2.circle(bk, (u, v), 5, (0, 0, 255), 5)
                cv2.putText(bk, str(round(pt[2], 1)), (u, v), 5, 5, (0, 0, 255), )
                print("--> u v: ", u, v)
            cv2.imshow('FullScreen', bk)

            if cv2.waitKey(10) == 27:
                break

            cv2.imshow('frame2', frame)

        except:
            continue
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
