import cv2
import cv2
import numpy as np
import timm
import torch
import torchvision.transforms as T
from scipy.spatial.transform import Rotation
import os
import requests
from gaze_module.face_camera import Face, FaceParts, FacePartsName, FaceModelMediaPipe, Camera

transform = T.Compose([
    T.Lambda(lambda x: cv2.resize(x, (224, 224))),
    T.Lambda(lambda x: x[:, :, ::-1].copy()),  # BGR -> RGB
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                 0.225]),  # RGB
])

import mediapipe as mp


def detect_faces_mediapipe(detector, image: np.ndarray):
    h, w = image.shape[:2]
    predictions = detector.process(image[:, :, ::-1])
    detected = []
    #  multi_face_landmarks 0-454号关键点坐标，可通过predictions.multi_face_landmarks[0].landmark[454].x .y来获取第1个人的脸的第454号关键点坐标（改坐标是比值）
    # 通过乘以图像的h、w获取该关键点的实际像素坐标 .x*w  .y*h
    if predictions.multi_face_landmarks:
        i = 0
        for prediction in predictions.multi_face_landmarks:
            i += 1
            pts = np.array([(pt.x * w, pt.y * h)  # pt.x和pt.y是像素点位置的比值，乘以图像的h和w就对应该点在图像上的像素位置坐标
                            for pt in prediction.landmark],
                           dtype=np.float64)
            bbox = np.vstack([pts.min(axis=0), pts.max(axis=0)])
            bbox = np.round(bbox).astype(np.int32)
            detected.append(Face(bbox, pts))
    return detected

# 归一化成单位向量
def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)

class HeadPoseNormalizer:
    def __init__(self, camera: Camera, normalized_camera: Camera, normalized_camera_eye: Camera,
                 normalized_distance: float):
        self.camera = camera
        self.normalized_camera = normalized_camera
        self.normalized_distance = normalized_distance
        self.normalized_camera_eye = normalized_camera_eye  # 111

    # 归一化面部图像和头部姿势   normalize(frame, face)
    def normalize(self, image: np.ndarray, eye_or_face: FaceParts) -> None:
        #旋转矩阵，面部和眼睛是相同的，且以面部为中心点计算得到的旋转矩阵
        eye_or_face.normalizing_rot = self._compute_normalizing_rotation(  # 计算相机坐标系到归一化虚拟坐标系的旋转矩阵 R
            eye_or_face.center, eye_or_face.head_pose_rot)

        eye_or_face.leye.normalizing_rot = self._compute_normalizing_rotation(  # 计算左眼的相机坐标系到归一化虚拟坐标系的旋转矩阵 R
            eye_or_face.leye.center, eye_or_face.leye.head_pose_rot)
        eye_or_face.reye.normalizing_rot = self._compute_normalizing_rotation(  # 右眼的旋转矩阵 R
            eye_or_face.reye.center, eye_or_face.reye.head_pose_rot)

        self._normalize_image(image, eye_or_face)  # 归一化面部图像
        #self._normalize_image_all(image, eye_or_face, self.normalized_camera)  # 归一化面部图像
        self._normalize_image_all(image, eye_or_face.leye,self.normalized_camera_eye)  # 归一化左眼
        self._normalize_image_all(image, eye_or_face.reye,self.normalized_camera_eye)  # 归一化右眼

        self._normalize_head_pose(eye_or_face)  # 归 一化头部姿势

    # # 新加的，归一化眼睛图像  同时归一化面部和眼睛图像时，左右眼睛的旋转矩阵与面部的旋转矩阵相同R(可参考FaceMode类)，头部姿势也是用面部中心计算的
    # # 所以同时归一化面部和眼睛图像时，该函数只用于对眼睛图像做放射变换
    # def normalize_faceeye(self, image: np.ndarray, eye_or_face: FaceParts) -> None:
    #     eye_or_face.normalizing_rot = self._compute_normalizing_rotation(  # 计算相机坐标系到归一化虚拟坐标系的旋转矩阵 R
    #         eye_or_face.center, eye_or_face.head_pose_rot)

    # 归一化眼睛图像
    def _normalize_image_all(self, image: np.ndarray,
                         eye_or_face: FaceParts, normalized_camera: Camera) -> None:
        camera_matrix_inv = np.linalg.inv(self.camera.camera_matrix)  # np.linalg.inv求逆矩阵  求(通过标定得到的)新相机的内参矩阵C_r的逆矩阵
        normalized_camera_matrix = normalized_camera.camera_matrix  # 眼睛归一化投影内参矩阵C_n

        scale = self._get_scale_matrix(eye_or_face.distance)  # 图像归一化需要的放缩矩阵 S, 左右眼的scale不一样
        conversion_matrix = scale @ eye_or_face.normalizing_rot.as_matrix()  # 图像归一化的变换矩阵 M = SR
        # 变换矩阵M 描述了3D缩放和旋转，将面部中心归一化到相机坐标系中的固定位置，并用于原始相机坐标系和归一化相机坐标系之间的3D位置之间的相互转换

        projection_matrix = normalized_camera_matrix @ conversion_matrix @ camera_matrix_inv  # 透视变换矩阵 W = C_s @ M @ C_r的逆

        # 对图像进行透视变换，就是变形 传入图像、透视变换矩阵W,和变换后（归一化后）的图像大小w、h， 得到归一化后的面部图像224*224
        normalized_image = cv2.warpPerspective(
            image, projection_matrix,
            (normalized_camera.width, normalized_camera.height))

        # 将眼睛图像灰度化，原始代码这儿未注释
        if eye_or_face.name in {FacePartsName.REYE, FacePartsName.LEYE}:
            normalized_image = cv2.cvtColor(normalized_image,
                                            cv2.COLOR_BGR2GRAY)
            normalized_image = cv2.equalizeHist(normalized_image)

        eye_or_face.normalized_image = normalized_image

    # 归一化（面部或眼睛）图像 原来的
    def _normalize_image(self, image: np.ndarray,
                         eye_or_face: FaceParts) -> None:
        camera_matrix_inv = np.linalg.inv(self.camera.camera_matrix)  # np.linalg.inv求逆矩阵  求(通过标定得到的)新相机的内参矩阵C_r的逆矩阵
        normalized_camera_matrix = self.normalized_camera.camera_matrix  # 面部归一化投影内参矩阵，即ETH-XGaze的内参矩阵 C_s

        scale = self._get_scale_matrix(eye_or_face.distance)  # 图像归一化需要的放缩矩阵 S
        conversion_matrix = scale @ eye_or_face.normalizing_rot.as_matrix()  # 图像归一化的变换矩阵 M = SR
        # 变换矩阵M 描述了3D缩放和旋转，将面部中心归一化到相机坐标系中的固定位置，并用于原始相机坐标系和归一化相机坐标系之间的3D位置之间的相互转换

        projection_matrix = normalized_camera_matrix @ conversion_matrix @ camera_matrix_inv  # 透视变换矩阵 W = C_s @ M @ C_r的逆

        # 对图像进行透视变换，就是变形 传入图像、透视变换矩阵W,和变换后（归一化后）的图像大小w、h， 得到归一化后的面部图像224*224
        normalized_image = cv2.warpPerspective(
            image, projection_matrix,
            (self.normalized_camera.width, self.normalized_camera.height))

        # 将眼睛图像灰度化，原始代码这儿未注释
        if eye_or_face.name in {FacePartsName.REYE, FacePartsName.LEYE}:
            normalized_image = cv2.cvtColor(normalized_image,
                                            cv2.COLOR_BGR2GRAY)
            normalized_image = cv2.equalizeHist(normalized_image)

        eye_or_face.normalized_image = normalized_image

    # 计算归一化头部姿势向量 pitch、yaw
    @staticmethod
    def _normalize_head_pose(eye_or_face: FaceParts) -> None:
        # 通过solvePnP得到的头部旋转向量R_r * 归一化旋转向量R得到在归一化虚拟相机坐标系下的归一化头部旋转向量（姿态），由于在归一化坐标系下，z轴坐标始终为0，所以头部姿态向量就由三维变成2维的（水平和垂直方向）
        # 计算归一化头部旋转向量R_n，按照MPIIGaze论文上的求法是 R_n = M @ R_r,R_r是由solvePnP求出来的旋转变量,M为图像归一化的变换矩阵  M = SR
        normalized_head_rot = eye_or_face.head_pose_rot * eye_or_face.normalizing_rot  # 此处是R_n = R_r * R ，向量计算而不是矩阵计算
        euler_angles2d = normalized_head_rot.as_euler('XYZ')[:2]  # 求欧拉角 pitch、yaw（不要roll）
        eye_or_face.normalized_head_rot2d = euler_angles2d * np.array([1, -1])

    # 计算相机坐标系到归一化虚拟坐标系的旋转矩阵 R  通过 M=SR可求归一化的变换矩阵M，S=diag(1,1,d_n/d)为放缩矩阵,d_n按照MPIIGaze论文上是0.6m
    @staticmethod
    def _compute_normalizing_rotation(center: np.ndarray,
                                      head_rot: Rotation) -> Rotation:
        # 旋转后的相机坐标系的z轴z_axis定义为从相机到参考点的直线，其中的参考点通常设置为脸中心或眼中心
        z_axis = _normalize_vector(center.ravel())  # np.ravel() 将数组维度拉成一维数组
        # 旋转的x轴x_axis被定义为头部坐标系统的x轴head_x_axis，这样旋转的相机捕捉到的外观是面向前方的。
        #Dprint("head_rot",head_rot)
        head_rot = head_rot.as_matrix()
        head_x_axis = head_rot[:, 0]

        # 两个向量叉积结果为二者构成平面的法向量，即叉积生成的向量垂直与两个向量构成的平面
        y_axis = _normalize_vector(np.cross(z_axis, head_x_axis))  # 向量叉积 y_axis = z_axis × head_x_axis
        x_axis = _normalize_vector(np.cross(y_axis, z_axis))  # 向量叉积 x_axis = y_axis × z_axis
        return Rotation.from_matrix(np.vstack([x_axis, y_axis, z_axis]))

    def _get_scale_matrix(self, distance: float) -> np.ndarray:
        return np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, self.normalized_distance / distance],
        ], dtype=np.float64)


class GazeEstimator:
    def __init__(self, checkpoint_path: str, 
                 camera_params: str, normalized_camera_params: str, 
                 normalized_camera_eye_params:str,
                 model_name: str = "resnet18",
                 normalized_distance: float = 0.6, device: str = None):  # 设置归一化的距离为0.6米，即设置归一化的图像为面部到摄像头的距离为0.6米
        camera = Camera(camera_params)  # 实时的相机内参矩阵
        normalized_camera = Camera(normalized_camera_params)  # ETH-XGaze归一化的相机内参矩阵，用于面部
        normalized_camera_eye = Camera(normalized_camera_eye_params)  # eye_normalized的虚拟相机内参矩阵  111
        self.camera = camera
        self.normalized_camera = normalized_camera
        self.normalized_camera_eye = normalized_camera_eye  # 111
        self.face_model_3d = FaceModelMediaPipe()
        self.head_pose_normalizer = HeadPoseNormalizer(camera, normalized_camera, normalized_camera_eye,
                                                       normalized_distance=normalized_distance)  # normalized_distance参考MPIIGaze论文为0.6m
        
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.gaze_estimation_model = timm.create_model(model_name, num_classes=2)
            try:
                assert os.path.isfile(checkpoint_path), f"checkpoint file not found: {checkpoint_path}"
                
                print("Starting loading！！！！")

                #自己训练模型的pth加载
                checkpoint = torch.load(checkpoint_path,map_location = self.device)
                self.gaze_estimation_model.load_state_dict(checkpoint)
                print("model successfully loaded.")
            except:
                print("model loading failed.")
            self.gaze_estimation_model.to(device)
            self.gaze_estimation_model.eval()
        elif self.device == 'npu':
            self.session = requests.session()
            
    def estimate(self, face, frame):
        self.face_model_3d.estimate_head_pose(face,
                                              self.camera)
        self.face_model_3d.compute_3d_pose(face)
        self.face_model_3d.compute_face_eye_centers(face, 'ETH-XGaze')
        self.head_pose_normalizer.normalize(frame, face)
        
        if self.device == 'npu':
            face.normalized_gaze_angles = self.estimateByAtlas200DK(face)
        else:
            image = transform(face.normalized_image).unsqueeze(0)
            image = image.to(self.device)
            
            prediction = self.gaze_estimation_model(image)
            prediction = prediction.detach().cpu().numpy()
            face.normalized_gaze_angles = prediction[0]
        face.angle_to_vector()
        face.denormalize_gaze_vector()

def gaze2point(center, gaze_vector):
    """
    x = gaze_vector[0] * t + center[0]
    y = gaze_vector[1] * t + center[1]
    z = gaze_vector[2] * t + center[2]

    solve it for z=0 :
    """
    t = - center[2] / gaze_vector[2]
    # print("center[2]",center[2],"gaze_vector[2]",gaze_vector[2],"t",t)
    x = gaze_vector[0] * t + center[0]
    y = gaze_vector[1] * t + center[1]
    return x, y

def px2mm(coords, width=1920, height=1080,
          width_mm=310, height_mm=174):
    x = (coords[0] / width) * width_mm
    x = - x + width_mm / 2
    y = (coords[1] / height) * height_mm
    return x, y


def mm2px(point, width=1920, height=1080,
          width_mm=310, height_mm=174):  # 14寸 width_mm=310, height_mm=174 27寸 width_mm=600, height_mm=330
    x, y = point
    # print("before %d,%d"%(x,y))
    # print("b_point",point)
    x = - x + width_mm / 2
    x = (x / width_mm) * width
    y = (y / height_mm) * height
    # print("a_point",(round(x),round(y)))
    # round(x)四舍五入变为整数
    return round(x), round(y)