import time
import os
import numpy as np
import joblib
import cv2
import mediapipe as mp
from gaze_module.asset import detect_faces_mediapipe, GazeEstimator, gaze2point, mm2px
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import socket
import json

def _gazeEstimate(face):
    x, y = gaze2point(face.center * 1e3, face.gaze_vector)
    x, y = mm2px((x, y))
    return int(x), int(y)

def _noseEstimate(face):
    x, y = gaze2point(face.nose_pos * 1e3, face.gaze_vector)
    x, y = mm2px((x, y))
    return int(x), int(y)

def _chinEstimate(face):
    x, y = gaze2point(face.chin_pos * 1e3, face.gaze_vector)
    x, y = mm2px((x, y))
    return int(x), int(y)

def estimate_gaze():
    estimator = GazeEstimator(
        checkpoint_path=r"gaze_module\data\resnet18\eth\Iter_10_resnet18.pth",
        camera_params=r"gaze_module\data\sample_params.yaml",
        normalized_camera_params=r"gaze_module\data\eth-xgaze.yaml",
        normalized_camera_eye_params=r"gaze_module\data\eye_normalized.yaml",
        model_name='resnet18',
        device='npu'
    )
    detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, 
        max_num_faces=1, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    pos_tag = 0
    # recv_flag常开，保持数据采集
    while True:
        # 拍摄一张图片
        ret, frame = cap.read()
        local_ts = time.time()
        # 如果没有拍摄到图片，或者不需要上传数据，则重拍
        if not ret:
            continue
        # 检测人脸
        faces = detect_faces_mediapipe(detector, frame)
        # 如果没有检测到人脸，重拍
        if len(faces) == 0:
            continue
        features = []
        for face in faces:
            # 估计注视点
            estimator.estimate(face, frame)
            # 可采集其他特征
            gaze_x, gaze_y = _gazeEstimate(face)
            nose_x, nose_y = _noseEstimate(face)
            chin_x, chin_y = _chinEstimate(face)
            pitch, yaw, roll = face.head_pose_rot.as_euler('xyz', degrees=True)
            feat = [
                gaze_x, gaze_y, 
                nose_x, nose_y, 
                chin_x, chin_y, 
                pitch, yaw, roll
            ]
            features.append(feat)
        
        yield features, local_ts

if __name__ == '__main__':
    # 开放端口
    host = "127.0.0.1"
    port = 20020
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (host, port)
    
    # 开启服务
    gaze_generator = estimate_gaze()
    while True:
        try:
            # 获取视线估计结果
            gaze_result = next(gaze_generator)
            # 将结果转换为JSON格式
            response = json.dumps(gaze_result)
            # 发送数据到客户端
            sock.sendto(response.encode('utf-8'), server_address)
        except StopIteration:
            break
        except Exception as e:
            print(e)
            continue
