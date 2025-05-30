from omegaconf import OmegaConf
import pathlib
from demo import Demo
from ptgaze.utils import check_path_all
from utils_plus import *

alpha = 0.2
smooth_a = 0
smooth_b = 0
smooth_u = 0
smooth_v = 0

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cam = cv2.VideoCapture(1)

package_root = pathlib.Path(__file__).parent.resolve()
print('package_root:', package_root)
config = OmegaConf.load(package_root / 'depth_estimation_mpiifacegaze.yaml')
# config = OmegaConf.load(package_root / 'depth_estimation_L2CS.yaml')
# config = OmegaConf.load(package_root / 'depth_estimation_et h-xgaze.yaml')
config.PACKAGE_ROOT = package_root.as_posix()  # 转斜杠\为/q
check_path_all(config)
# GazeEstimator(config)
demo = Demo(config)
temp_p, temp_y = 0, 0
fps_history = []
p1_s0 = [0., 0., 0.]
p2_s0 = [0., 0., 1.]
p1_s1 = [0., 0., 0.]
p2_s1 = [0., 0., 0.]
p1_s2 = [0., 0., 0.]
p2_s2 = [0., 0., 0.]
x_adjust = 0.
y_adjust = 0.
vector_adjust = [0., 0., 0.]
R2 = [0.01982, -0.99974, 0.01064, 0.99771, 0.01909, -0.06479, 0.06457, 0.01190, 0.99784]
T2 = [0.04763, 30.02763, 0.99784]
w_vis, h_vis = 1920, 1080
w_world, h_world = det_world_wh(27, 16, 9)
test_point = [0., 0., 0.]
fillter = Move_Filter()


while True:
    demo._wait_key()
    ok, frame = cam.read()
    # print(frame)
    if not ok:
        break

    # 水平镜像翻转（参数 1 表示水平翻转）
    mirrored_frame = cv2.flip(frame, 1)
    fps_start_time = cv2.getTickCount()
    # print('1')
    try:
        faces, vector, head_vector, R1, T1, face_center = demo._process_image(frame)
        # print('faces:', faces, 'vector:', vector, 'face_center', face_center)

        for face in faces:
            depth = 10 / ((face.bbox[1][0] - face.bbox[0][0])/640)

            p1_s1 = face_center  # x,y轴为归一化后的结果，x轴范围[0.5, -0.5]
            # print('p1_s1:', p1_s1)
            # p12_vec_s0 = face.normalized_gaze_vector
            p12_vec_s1 = face.gaze_vector
            # print('p12_vec_s1:', p12_vec_s1)

            # sight_line = normalize_vector(vector_adjust)
            # sight_line = vector_adjust
            for i in range(3):
                p2_s1[i] = p1_s1[i] + p12_vec_s1[i]
            for i in [p1_s1, p2_s1]:  # 将归一化向量转化为真实物理大小
                i[0] = i[0] * 640 * 1.8
                i[1] = i[1] * 480 * 2.1
                i[2] = i[2] * 690
            # for i in range(3):
            #     p2_s1[i] = p2_s1[i] + sight_line[i]
            # print(p2_s1, p1_s1)

            u_temp, v_temp = uv_in_screen(p1_s1, p2_s1, p1_s2, p2_s2, w_vis, h_vis, w_world, h_world, R2, T2)
            # u, v = u_temp, v_temp
            u, v = fillter.update(u_temp, v_temp)
            # print('u:', u, 'v:', v)

            demo.visualizer.image = cv2.cvtColor(demo.visualizer.image, cv2.COLOR_BGR2RGB)
            cv2.putText(
                demo.visualizer.image,
                f"depth: {depth:.2f}cm",
                (150, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,(255, 255, 0),1, cv2.LINE_AA,
            )
            cv2.putText(
                demo.visualizer.image,
                f"pitch: {vector[0]:.2f} yaw:{vector[1]:.3f}",
                (150, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,(255, 255, 0), 1, cv2.LINE_AA,
            )
            cv2.putText(
                demo.visualizer.image,
                f"head_pitch: {head_vector[0]:.2f} head_yaw:{head_vector[1]:.3f}",
                (150, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 0), 1, cv2.LINE_AA,
            )
            demo.visualizer.image = cv2.cvtColor(demo.visualizer.image, cv2.COLOR_RGB2BGR)

    except:
        continue

    # 获取屏幕分辨率（可根据你的屏幕分辨率手动设置）
    screen_width = 1920
    screen_height = 1080
    # 创建黑色背景
    background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8) # 先高后宽

    # 将a,b从cm映射到像素（假设中心为画布中心）
    # 你可以根据实际最大距离调整缩放因子
    scale = 35  # 每cm对应5个像素（可调）
    center_x = screen_width // 2
    center_y = screen_height // 2

    # 应用指数平滑
    smooth_u = alpha * u + (1 - alpha) * smooth_u
    smooth_v = alpha * v + (1 - alpha) * smooth_v

    point_x = int(smooth_v)
    point_y = int(smooth_u)
    point_x = screen_width - 1 - point_x  # X方向不变（向右为正）
    point_y = screen_height - 1 - point_y  # Y方向翻转（向上为正）

    cv2.line(background, [int(center_x) - 3, int(center_y)], [int(center_x) + 3, int(center_y)], (255, 0, 0), 2)
    cv2.line(background, [int(center_x), int(center_y) + 3], [int(center_x), int(center_y) - 3], (255, 0, 0), 2)
    # cv2.line(background, int(center_y) - 10, int(center_y) + 10, (255, 0, 0), 2)

    # 将坐标限制在图像边界内
    clamped_x = max(0, min(point_x, screen_width - 1))
    clamped_y = max(0, min(point_y, screen_height - 1))

    # 在裁剪后的坐标处绘制圆点
    cv2.circle(background, (clamped_x, clamped_y), 15, (0, 0, 255), -1)
    cv2.putText(
        background,
        f"u: {u:.2f} v:{v:.3f}",
        (900, 500),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 255, 0), 1, cv2.LINE_AA,
    )
    cv2.putText(
        background,
        f"adjust vector: [{vector_adjust[0]:.2f}, {vector_adjust[1]:.2f}, {vector_adjust[2]:.2f}]",
        (900, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 255, 255), 1, cv2.LINE_AA,
    )

    # 创建窗口并设为全屏
    cv2.namedWindow("Fullscreen Background", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Fullscreen Background", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    resized_frame = cv2.resize(frame, (200, 153))  # 自定义缩放大小

    # 将缩小的frame贴到background左上角
    h, w, _ = resized_frame.shape
    background[0:h, screen_width - w:screen_width] = resized_frame

    end_time = cv2.getTickCount()
    demo.visualizer.image = draw_fps(demo.visualizer.image, fps_start_time, end_time, fps_history)
    background = draw_fps(background, fps_start_time, end_time, fps_history)
    # 显示图像
    cv2.imshow("Fullscreen Background", background)
    cv2.imshow('frame', demo.visualizer.image)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        temp_p, temp_y = vector[0], vector[1]
        print('已重新标定视觉中心')
    elif key & 0xFF == ord('r'):
        temp_p, temp_y = 0, 0
        print('重置')
    elif key & 0xFF == ord('j'):
        # x_adjust += 0.01
        vector_adjust[0] += 1
    elif key & 0xFF == ord('l'):
        # x_adjust -= 0.01
        vector_adjust[0] -= 1
    elif key & 0xFF == ord('i'):
        # y_adjust += 0.01
        vector_adjust[1] += 1
    elif key & 0xFF == ord('k'):
        # y_adjust -= 0.01
        vector_adjust[1] -= 1
    elif key & 0xFF == ord('o'):
        vector_adjust = [0., 0., 0.]

# 释放摄像头并关闭所有窗口
cam.release()
cv2.destroyAllWindows()
