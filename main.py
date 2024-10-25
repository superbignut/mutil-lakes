
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def check_red_func(cloth_img):
    """ cloth_img = cv2.imread('cloth.jpg') # np.ndarray """

    image_rgb = cv2.cvtColor(cloth_img, cv2.COLOR_BGR2RGB) # 转为 rgb格式

    lower_red = np.array([100, 0, 0])   # 红色的最低范围
    upper_red = np.array([255, 80, 80]) # 红色的最高范围

    # 创建掩码
    red_mask = cv2.inRange(image_rgb, lower_red, upper_red) # 在范围内的是255 其余变成0

    # 计算红色区域的像素数量
    red_pixels = cv2.countNonZero(red_mask) # 计算非零区域

    # 计算总像素数量
    total_pixels = image_rgb.shape[0] * image_rgb.shape[1] # 统计总像素数

    # 计算红色占比
    red_ratio = red_pixels / total_pixels

    print(f"红色占比: {red_ratio * 100:.2f}%")


def video_demo():
    capture = cv2.VideoCapture(0)  # 0为电脑内置摄像头

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定义视频编解码器
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))  # 创建视频写入对象，参数依次为输出文件名、编解码器、帧率、帧大小

    while (True):
        ret, frame = capture.read()  # 摄像头读取,ret为是否成功打开摄像头,true,false。 frame为视频的每一帧图像
        if ret:
            out.write(frame)  # 写入帧到视频文件
            frame = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示。
            # print(frame.shape)
            cv2.imshow("video", frame)
            check_red_func(frame)
            c = cv2.waitKey(50)
            if c == 27: # ESC
                break
    capture.release()  # 释放捕获设备
    out.release()  # 释放视频写入对象

video_demo()
cv2.destroyAllWindows()




