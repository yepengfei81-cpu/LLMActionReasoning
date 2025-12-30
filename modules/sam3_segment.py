"""
SAM3 砖块分割模块 (重构版)
- 通用相机基类
- SAM3 分割器
- 环境相机 / 手眼相机
"""

import sys
SAM3_PROJECT_PATH = "/home/ypf/sam3-main"
if SAM3_PROJECT_PATH not in sys.path:
    sys.path.insert(0, SAM3_PROJECT_PATH)

import torch
import numpy as np
from PIL import Image
import cv2
import threading
import time
from typing import Tuple, Optional, Dict, List, Union
from abc import ABC, abstractmethod
import gc
import pybullet as p

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# 可视化颜色
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
    (0, 165, 255), (255, 255, 0), (128, 0, 128), (0, 128, 128),
]


# ============================================================================
#                          通用相机基类
# ============================================================================

class CameraBase(ABC):
    """
    相机基类 - 提供通用的内参、外参和坐标转换功能
    """
    def __init__(self,
                 width: int = 640,
                 height: int = 480,
                 fov: float = 60.0,
                 near: float = 0.01,
                 far: float = 2.0,
                 use_opengl: bool = True):
        
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far
        self.renderer = p.ER_BULLET_HARDWARE_OPENGL if use_opengl else p.ER_TINY_RENDERER
        
        # 计算投影矩阵
        aspect = width / height
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov, aspect=aspect, nearVal=near, farVal=far
        )
        
        # 计算相机内参
        fov_rad = np.radians(fov)
        self.fy = height / (2.0 * np.tan(fov_rad / 2.0))
        self.fx = self.fy  # 假设正方形像素
        self.cx = width / 2.0
        self.cy = height / 2.0
        
        self._running = False
    
    def get_intrinsics(self) -> Dict[str, float]:
        """获取相机内参"""
        return {
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'width': self.width,
            'height': self.height,
            'fov': self.fov,
            'near': self.near,
            'far': self.far
        }
    
    @abstractmethod
    def get_camera_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取相机在世界坐标系中的位姿
        Returns:
            position: 相机位置 (3,)
            orientation: 相机姿态四元数 (4,)
        """
        pass
    
    @abstractmethod
    def get_view_matrix(self) -> List[float]:
        """获取视图矩阵"""
        pass
    
    def get_extrinsics(self) -> Dict[str, np.ndarray]:
        """
        获取相机外参（世界到相机变换）
        Returns:
            dict with 'R' (3x3 rotation), 't' (3, translation), 'pose' (4x4 matrix)
        """
        view_matrix = self.get_view_matrix()
        # PyBullet 的 view_matrix 是列主序的 4x4 矩阵
        view_matrix_np = np.array(view_matrix).reshape(4, 4, order='F')
        
        # view_matrix 是 world_to_camera 变换
        # 取逆得到 camera_to_world (相机位姿)
        camera_pose = np.linalg.inv(view_matrix_np)
        
        R = camera_pose[:3, :3]  # 相机在世界坐标系的旋转
        t = camera_pose[:3, 3]   # 相机在世界坐标系的位置
        
        return {
            'R': R,
            't': t,
            'pose': camera_pose,
            'view_matrix': view_matrix_np
        }
    
    def pixel_to_camera(self, u: float, v: float, depth: float) -> np.ndarray:
        """
        像素坐标 + 深度 -> 相机坐标系中的 3D 点
        
        Args:
            u: 像素 x 坐标
            v: 像素 y 坐标  
            depth: 深度值 (米)
        
        Returns:
            相机坐标系中的 3D 点 (3,)
        """
        # 标准针孔相机模型
        x_cam = (u - self.cx) * depth / self.fx
        y_cam = (v - self.cy) * depth / self.fy
        z_cam = depth
        
        return np.array([x_cam, y_cam, z_cam])
    
    def pixel_to_world(self, u: float, v: float, depth: float) -> np.ndarray:
        """
        像素坐标 + 深度 -> 世界坐标系中的 3D 点
        
        Args:
            u: 像素 x 坐标
            v: 像素 y 坐标
            depth: 深度值 (米)
        
        Returns:
            世界坐标系中的 3D 点 (3,)
        """
        # 先转到相机坐标系
        # 注意：PyBullet 相机的 Z 轴指向场景（与 OpenCV 惯例相同）
        # 但 Y 轴朝下，需要处理符号
        x_cam = (u - self.cx) * depth / self.fx
        y_cam = -(v - self.cy) * depth / self.fy  # 注意负号
        z_cam = -depth  # 相机看向 -Z
        
        point_cam = np.array([x_cam, y_cam, z_cam])
        
        # 获取外参并转换到世界坐标
        extrinsics = self.get_extrinsics()
        R = extrinsics['R']
        t = extrinsics['t']
        
        point_world = R @ point_cam + t
        return point_world
    
    def capture_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        捕获 RGB 和深度图像
        
        Returns:
            rgb_bgr: BGR 格式图像 (H, W, 3)
            depth: 深度图 (H, W)，单位米
        """
        view_matrix = self.get_view_matrix()
        
        _, _, rgb, depth_buffer, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=self.renderer
        )
        
        # 转换 RGB
        rgb = np.array(rgb, dtype=np.uint8).reshape(self.height, self.width, 4)
        rgb_bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
        
        # 转换深度
        depth_buffer = np.array(depth_buffer, dtype=np.float32).reshape(self.height, self.width)
        depth = self.far * self.near / (self.far - (self.far - self.near) * depth_buffer)
        
        return rgb_bgr, depth
    
    def start(self):
        """启动相机"""
        self._running = True
    
    def stop(self):
        """停止相机"""
        self._running = False
    
    def close(self):
        """关闭相机"""
        self.stop()


# ============================================================================
#                          固定位置环境相机
# ============================================================================

class FixedCamera(CameraBase):
    """
    固定位置的环境相机
    """
    def __init__(self,
                 camera_position: Tuple[float, float, float] = (0.0, 0.0, 2.0),
                 camera_target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 camera_up: Optional[Tuple[float, float, float]] = None,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.camera_position = np.array(camera_position)
        self.camera_target = np.array(camera_target)
        
        # 自动检测是否垂直俯视
        look_dir = self.camera_target - self.camera_position
        look_dir_normalized = look_dir / (np.linalg.norm(look_dir) + 1e-8)
        self.is_vertical = abs(look_dir_normalized[2]) > 0.9
        
        # 设置上向量
        if camera_up is not None:
            self.camera_up = camera_up
        else:
            self.camera_up = (0.0, 1.0, 0.0) if self.is_vertical else (0.0, 0.0, 1.0)
        
        # 预计算视图矩阵
        self._view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_position,
            cameraTargetPosition=camera_target,
            cameraUpVector=self.camera_up
        )
        
        print(f"[FixedCamera] 位置: {camera_position}")
        print(f"[FixedCamera] 目标: {camera_target}")
        print(f"[FixedCamera] 垂直俯视: {self.is_vertical}")
    
    def get_camera_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取相机位姿"""
        extrinsics = self.get_extrinsics()
        R = extrinsics['R']
        t = extrinsics['t']
        
        # 从旋转矩阵计算四元数
        from scipy.spatial.transform import Rotation
        quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
        
        return t, quat
    
    def get_view_matrix(self) -> List[float]:
        """获取预计算的视图矩阵"""
        return self._view_matrix


# ============================================================================
#                          手眼相机 (Eye-in-Hand)
# ============================================================================
class EyeInHandCamera(CameraBase):
    """
    手眼相机 - 固定在机械臂末端（TCP 坐标系）
    """
    def __init__(self,
                 robot_model,
                 local_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 local_orientation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 **kwargs):
        super().__init__(**kwargs)
        
        self.rm = robot_model
        self.local_pos = list(local_position)
        self.local_orn = p.getQuaternionFromEuler(local_orientation_rpy)
        
        # 分割结果缓存
        self._lock = threading.Lock()
        self._cached_frame: Optional[np.ndarray] = None
        self._cached_masks: Optional[np.ndarray] = None
        self._cached_results: List[Dict] = []
        self._last_segment_time: float = 0
        self._segment_display_duration: float = 3.0
        
        print(f"[EyeInHand] 初始化完成 (基于 TCP 坐标系)")
        print(f"   local_pos: {self.local_pos}")
        print(f"   local_orn_rpy: {local_orientation_rpy}")
        print(f"   分辨率: {self.width}x{self.height}, FOV: {self.fov}°")
        
    def get_camera_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        tcp_pos, tcp_orn = self.rm.tcp_world_pose()
        camera_pos, camera_orn = p.multiplyTransforms(
            tcp_pos, tcp_orn,
            self.local_pos, self.local_orn
        )
        return np.array(camera_pos), np.array(camera_orn)
    
    def get_view_matrix(self) -> List[float]:
        cam_pos, cam_orn = self.get_camera_pose()
        rot_matrix = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
        forward = -rot_matrix[:, 2]
        up = -rot_matrix[:, 1]
        target = cam_pos + forward * 0.5
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist()
        )
        return view_matrix
    
    def update_segmentation_cache(self, frame: np.ndarray, masks: Optional[np.ndarray], results: List[Dict] = None):
        """更新分割结果缓存"""
        with self._lock:
            self._cached_frame = frame.copy() if frame is not None else None
            self._cached_masks = masks
            self._cached_results = results or []
            self._last_segment_time = time.time()
    
    def _draw_segmentation(self, frame: np.ndarray, masks: Optional[np.ndarray]) -> np.ndarray:
        """绘制分割结果"""
        if masks is None or len(masks) == 0:
            return frame
        
        result = frame.copy()
        height, width = frame.shape[:2]
        
        for i, mask in enumerate(masks):
            if len(mask.shape) == 3:
                mask = mask[0]
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.float32), (width, height),
                                 interpolation=cv2.INTER_NEAREST)
            
            mask_bool = mask > 0.5
            if not np.any(mask_bool):
                continue
            
            color = COLORS[i % len(COLORS)]
            overlay = result.copy()
            overlay[mask_bool] = color
            result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
            
            contours, _ = cv2.findContours((mask_bool * 255).astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, color, 2)
            
            ys, xs = np.where(mask_bool)
            if len(xs) > 0:
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                cv2.circle(result, (cx, cy), 5, color, -1)
                cv2.circle(result, (cx, cy), 8, (255, 255, 255), 2)
        
        return result
    
    def get_display_frame(self) -> np.ndarray:
        """获取用于显示的帧（支持分割结果显示）"""
        frame, _ = self.capture_frame()
        tcp_pos, _ = self.rm.tcp_world_pose()
        
        with self._lock:
            show_segmentation = (
                self._cached_masks is not None and 
                self._cached_frame is not None and
                (time.time() - self._last_segment_time) < self._segment_display_duration
            )
            
            if show_segmentation:
                display_frame = self._draw_segmentation(self._cached_frame, self._cached_masks)
                num_detected = len(self._cached_masks) if self._cached_masks is not None else 0
                cv2.putText(display_frame, f"Eye-in-Hand SAM3: {num_detected} detected", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if self._cached_results:
                    for i, result in enumerate(self._cached_results[:3]):
                        pos = result['position']
                        y_offset = 50 + i * 20
                        cv2.putText(display_frame, 
                                   f"#{i+1}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            else:
                display_frame = frame
                cv2.putText(display_frame, "Eye-in-Hand (TCP)", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(display_frame, f"TCP: ({tcp_pos[0]:.2f}, {tcp_pos[1]:.2f}, {tcp_pos[2]:.2f})", 
                   (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return display_frame
    
    def clear_segmentation_cache(self):
        """清除分割缓存"""
        with self._lock:
            self._cached_frame = None
            self._cached_masks = None
            self._cached_results = []
            self._last_segment_time = 0


# ============================================================================
#                          SAM3 分割器 (通用)
# ============================================================================

class SAM3Segmenter:
    """
    SAM3 分割器 - 与相机解耦，可用于任何相机
    """
    _instance = None
    _model = None
    _processor = None
    
    def __init__(self,
                 checkpoint_path: str = "/home/ypf/sam3-main/checkpoint/sam3.pt",
                 text_prompt: str = "red brick",
                 sam_resolution: int = 1008,
                 confidence_threshold: float = 0.5):
        
        self.text_prompt = text_prompt
        self.sam_resolution = sam_resolution
        self.confidence_threshold = confidence_threshold
        
        # 单例模式：避免重复加载模型
        if SAM3Segmenter._model is None:
            print(f"[SAM3Segmenter] 加载模型...")
            SAM3Segmenter._model = build_sam3_image_model(checkpoint_path=checkpoint_path)
            SAM3Segmenter._processor = Sam3Processor(
                SAM3Segmenter._model, 
                resolution=sam_resolution, 
                confidence_threshold=confidence_threshold
            )
            print(f"[SAM3Segmenter] 模型加载完成!")
            torch.cuda.empty_cache()
            gc.collect()
        
        self.model = SAM3Segmenter._model
        self.processor = SAM3Segmenter._processor
        
        self._lock = threading.Lock()
    
    def segment(self, image_bgr: np.ndarray, prompt: Optional[str] = None) -> Optional[np.ndarray]:
        """
        对图像进行分割
        
        Args:
            image_bgr: BGR 格式图像
            prompt: 文本提示（可选，默认使用初始化时的提示）
        
        Returns:
            masks: 分割掩码数组 (N, H, W) 或 None
        """
        with self._lock:
            prompt = prompt or self.text_prompt
            
            # BGR -> RGB -> PIL
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # 分割
            inference_state = self.processor.set_image(pil_image)
            output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
            
            masks = output["masks"].cpu().numpy() if output["masks"] is not None else None
            
            torch.cuda.empty_cache()
            
            return masks
    
    def segment_async(self, image_bgr: np.ndarray, callback, prompt: Optional[str] = None):
        """
        异步分割
        
        Args:
            image_bgr: BGR 格式图像
            callback: 回调函数，签名 callback(masks)
            prompt: 文本提示
        """
        def worker():
            masks = self.segment(image_bgr, prompt)
            callback(masks)
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread


# ============================================================================
#                       砖块位置计算器 (通用)
# ============================================================================

class BrickPositionCalculator:
    """
    砖块位置计算器 - 从分割掩码计算 3D 位置
    """
    def __init__(self, brick_height: float = 0.06):
        self.brick_height = brick_height
    
    def compute_positions(self,
                          camera: CameraBase,
                          masks: np.ndarray,
                          depth: np.ndarray,
                          adjust_to_centroid: bool = True) -> List[Dict]:
        """
        从分割掩码计算砖块 3D 位置
        
        Args:
            camera: 相机实例（提供内参外参）
            masks: 分割掩码 (N, H, W)
            depth: 深度图 (H, W)
            adjust_to_centroid: 是否将表面点调整到质心
        
        Returns:
            positions: 砖块位置列表
        """
        results = []
        height, width = depth.shape
        cam_pos, _ = camera.get_camera_pose()
        
        for mask in masks:
            # 处理掩码维度
            if len(mask.shape) == 3:
                mask = mask[0]
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.float32), (width, height),
                                 interpolation=cv2.INTER_NEAREST)
            
            mask_bool = mask > 0.5
            if not np.any(mask_bool):
                continue
            
            # 计算掩码中心
            ys, xs = np.where(mask_bool)
            cx, cy = np.mean(xs), np.mean(ys)
            
            # 计算深度（使用中心区域的中值）
            kernel = np.ones((5, 5), np.uint8)
            center_mask = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=2)
            mask_depths = depth[center_mask > 0] if np.any(center_mask > 0) else depth[mask_bool]
            avg_depth = np.median(mask_depths)
            
            # 像素 -> 世界坐标
            surface_pos = camera.pixel_to_world(cx, cy, avg_depth)
            
            # 调整到砖块质心（从表面向下偏移半个砖块高度）
            if adjust_to_centroid:
                centroid_pos = surface_pos.copy()
                centroid_pos[2] = max(
                    surface_pos[2] - self.brick_height / 2.0,
                    self.brick_height / 2.0
                )
            else:
                centroid_pos = surface_pos
            
            results.append({
                'position': centroid_pos,
                'surface_position': surface_pos,
                'pixel_center': (cx, cy),
                'depth': avg_depth,
                'distance': np.linalg.norm(centroid_pos - cam_pos),
                'mask_area': np.sum(mask_bool)
            })
        
        # 按距离排序
        results.sort(key=lambda x: x['distance'])
        return results
    
    def match_with_ground_truth(self,
                                 detected_positions: List[Dict],
                                 brick_body_ids: List[int],
                                 max_distance: float = 0.1) -> List[Dict]:
        """
        将检测结果与真实砖块匹配
        
        Args:
            detected_positions: 检测到的位置列表
            brick_body_ids: PyBullet 砖块 body IDs
            max_distance: 最大匹配距离
        
        Returns:
            matched: 匹配结果列表
        """
        # 获取真实位置
        gt_positions = []
        for body_id in brick_body_ids:
            try:
                pos, _ = p.getBasePositionAndOrientation(body_id)
                gt_positions.append({
                    'id': body_id,
                    'position': np.array(pos)
                })
            except:
                pass
        
        matched = []
        used_gt = set()
        
        for det in detected_positions:
            det_pos = det['position']
            best_match = None
            best_dist = float('inf')
            
            for gt in gt_positions:
                if gt['id'] in used_gt:
                    continue
                
                dist = np.linalg.norm(det_pos - gt['position'])
                if dist < best_dist and dist < max_distance:
                    best_dist = dist
                    best_match = gt
            
            if best_match is not None:
                used_gt.add(best_match['id'])
                matched.append({
                    'detected': det,
                    'ground_truth': best_match,
                    'error': best_dist,
                    'error_xyz': det_pos - best_match['position']
                })
            else:
                matched.append({
                    'detected': det,
                    'ground_truth': None,
                    'error': None,
                    'error_xyz': None
                })
        
        return matched


# ============================================================================
#                       SAM3 砖块分割系统 (集成类)
# ============================================================================

class SAM3BrickSegmenter:
    """
    SAM3 砖块分割系统 - 集成相机、分割器和位置计算
    （保持向后兼容的接口）
    """
    def __init__(self,
                 camera_position: Tuple[float, float, float] = (0.0, 0.0, 2.0),
                 camera_target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 width: int = 640,
                 height: int = 480,
                 fov: float = 60.0,
                 near: float = 0.1,
                 far: float = 5.0,
                 checkpoint_path: str = "/home/ypf/sam3-main/checkpoint/sam3.pt",
                 text_prompt: str = "red brick",
                 sam_resolution: int = 1008,
                 confidence_threshold: float = 0.5,
                 use_opengl: bool = True,
                 brick_body_ids: List[int] = None,
                 brick_height: float = 0.06):
        
        # 创建固定相机
        self.camera = FixedCamera(
            camera_position=camera_position,
            camera_target=camera_target,
            width=width,
            height=height,
            fov=fov,
            near=near,
            far=far,
            use_opengl=use_opengl
        )
        
        # 创建分割器
        self.segmenter = SAM3Segmenter(
            checkpoint_path=checkpoint_path,
            text_prompt=text_prompt,
            sam_resolution=sam_resolution,
            confidence_threshold=confidence_threshold
        )
        
        # 创建位置计算器
        self.position_calculator = BrickPositionCalculator(brick_height=brick_height)
        
        # 配置
        self.brick_body_ids = brick_body_ids or []
        self.brick_height = brick_height
        self.text_prompt = text_prompt
        
        # 向后兼容的属性
        self.width = width
        self.height = height
        self.camera_position = np.array(camera_position)
        
        # 缓存
        self._running = False
        self._lock = threading.Lock()
        self._cached_frame: Optional[np.ndarray] = None
        self._cached_masks: Optional[np.ndarray] = None
        self._cached_results: List[Dict] = []
        self._last_segment_time: float = 0
        self._segment_display_duration: float = 3.0
        self._segment_pending = False
    
    def capture_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        """捕获图像"""
        return self.camera.capture_frame()
    
    def get_display_frame(self) -> np.ndarray:
        """获取显示帧"""
        frame, _ = self.capture_frame()
        
        with self._lock:
            show_segmentation = (
                self._cached_masks is not None and 
                (time.time() - self._last_segment_time) < self._segment_display_duration
            )
            
            if show_segmentation:
                display_frame = self._draw_segmentation(self._cached_frame, self._cached_masks)
                num_detected = len(self._cached_masks) if self._cached_masks is not None else 0
                cv2.putText(display_frame, f"SAM3: {num_detected} bricks", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                display_frame = frame
                cv2.putText(display_frame, "Live View (press 's' to segment)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        return display_frame
    
    def _draw_segmentation(self, frame: np.ndarray, masks: Optional[np.ndarray]) -> np.ndarray:
        """绘制分割结果"""
        if masks is None or len(masks) == 0:
            return frame
        
        result = frame.copy()
        for i, mask in enumerate(masks):
            if len(mask.shape) == 3:
                mask = mask[0]
            if mask.shape != (self.height, self.width):
                mask = cv2.resize(mask.astype(np.float32), (self.width, self.height),
                                 interpolation=cv2.INTER_NEAREST)
            
            mask_bool = mask > 0.5
            if not np.any(mask_bool):
                continue
            
            color = COLORS[i % len(COLORS)]
            overlay = result.copy()
            overlay[mask_bool] = color
            result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
            
            contours, _ = cv2.findContours((mask_bool * 255).astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, color, 2)
        
        return result
    
    def segment_and_localize(self,
                              camera: Optional[CameraBase] = None,
                              prompt: Optional[str] = None,
                              verbose: bool = True) -> List[Dict]:
        """
        通用分割和定位方法
        
        Args:
            camera: 要使用的相机（默认使用内置相机）
            prompt: 文本提示
            verbose: 是否打印结果
        
        Returns:
            positions: 砖块位置列表
        """
        cam = camera or self.camera
        prompt = prompt or self.text_prompt
        
        # 捕获图像
        frame, depth = cam.capture_frame()
        
        # 分割
        masks = self.segmenter.segment(frame, prompt)
        
        if masks is None or len(masks) == 0:
            if verbose:
                print("[SAM3] 未检测到砖块")
            # 清除手眼相机缓存
            if isinstance(cam, EyeInHandCamera):
                cam.update_segmentation_cache(frame, None, [])
            return []
        
        # 计算位置
        results = self.position_calculator.compute_positions(cam, masks, depth)
        
        # 匹配真实位置并打印
        if self.brick_body_ids and verbose:
            matched = self.position_calculator.match_with_ground_truth(
                results, self.brick_body_ids
            )
            self._print_results(results, matched, cam)
        
        # 更新缓存 - 根据相机类型
        if isinstance(cam, EyeInHandCamera):
            cam.update_segmentation_cache(frame, masks, results)
        else:
            with self._lock:
                self._cached_frame = frame.copy()
                self._cached_masks = masks
                self._cached_results = results
                self._last_segment_time = time.time()
        
        return results
    
    def _print_results(self, results: List[Dict], matched: List[Dict], camera: CameraBase):
        """打印位置结果"""
        cam_pos, _ = camera.get_camera_pose()
        
        print(f"\n{'='*80}")
        print(f"SAM3 砖块定位结果 | {time.strftime('%H:%M:%S')}")
        print(f"相机类型: {camera.__class__.__name__}")
        print(f"相机位置: ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})")
        print(f"{'='*80}")
        
        print(f"\n[检测结果] ({len(results)} 个)")
        print(f"{'序号':<4} {'X':>8} {'Y':>8} {'Z':>8} │ {'误差X':>8} {'误差Y':>8} {'误差Z':>8} │ {'总误差':>8}")
        print("-" * 80)
        
        for i, m in enumerate(matched):
            det = m['detected']
            pos = det['position']
            
            if m['ground_truth'] is not None:
                err = m['error_xyz']
                total_err = m['error']
                print(f"{i+1:<4} {pos[0]:>8.4f} {pos[1]:>8.4f} {pos[2]:>8.4f} │ "
                      f"{err[0]:>+8.4f} {err[1]:>+8.4f} {err[2]:>+8.4f} │ {total_err:>8.4f}")
            else:
                print(f"{i+1:<4} {pos[0]:>8.4f} {pos[1]:>8.4f} {pos[2]:>8.4f} │ "
                      f"{'N/A':>8} {'N/A':>8} {'N/A':>8} │ {'N/A':>8}")
        
        print(f"{'='*80}\n")
    
    def trigger_segment(self):
        """触发异步分割"""
        if self._segment_pending:
            print("[SAM3] 分割正在进行中，跳过...")
            return
        
        self._segment_pending = True
        
        def callback(masks):
            if masks is not None:
                frame, depth = self.camera.capture_frame()
                results = self.position_calculator.compute_positions(self.camera, masks, depth)
                
                with self._lock:
                    self._cached_masks = masks
                    self._cached_results = results
                    self._last_segment_time = time.time()
                
                if results and self.brick_body_ids:
                    matched = self.position_calculator.match_with_ground_truth(
                        results, self.brick_body_ids
                    )
                    self._print_results(results, matched, self.camera)
            
            self._segment_pending = False
        
        frame, _ = self.camera.capture_frame()
        self._cached_frame = frame.copy()
        self.segmenter.segment_async(frame, callback)
        print("[SAM3] 触发分割...")
    
    def start(self):
        """启动"""
        self._running = True
        self.camera.start()
        print("[SAM3] 启动完成")
    
    def stop(self):
        """停止"""
        self._running = False
        self.camera.stop()
    
    def close(self):
        """关闭"""
        self.stop()
        torch.cuda.empty_cache()
        gc.collect()
        print("[SAM3] 已关闭")
    
    def get_positions(self) -> List[np.ndarray]:
        """获取缓存的位置"""
        with self._lock:
            return [r['position'] for r in self._cached_results]
    
    def get_num_detected(self) -> int:
        """获取检测数量"""
        with self._lock:
            return len(self._cached_results)
    
    def set_brick_body_ids(self, body_ids: List[int]):
        """设置砖块 ID"""
        self.brick_body_ids = body_ids
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.close()
        return False


# ============================================================================
#                       显示管理器
# ============================================================================

class CameraDisplayManager:
    """统一的相机显示管理器"""
    def __init__(self, 
                 sam3_segmenter: Optional[SAM3BrickSegmenter] = None,
                 eye_in_hand: Optional[EyeInHandCamera] = None,
                 display_fps: int = 30,
                 combined_view: bool = True):
        
        self.sam3 = sam3_segmenter
        self.eye_in_hand = eye_in_hand
        self.display_fps = display_fps
        self.combined_view = combined_view
        
        self._running = False
        self._display_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._key_queue: List[int] = []
        self._key_lock = threading.Lock()
        
        print(f"[DisplayManager] 初始化完成")
        print(f"   SAM3: {'已连接' if sam3_segmenter else '未连接'}")
        print(f"   手眼相机: {'已连接' if eye_in_hand else '未连接'}")
        print(f"   显示帧率: {display_fps} FPS")
    
    def _display_thread_func(self):
        """显示线程"""
        window_name = "Camera View"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        frame_interval = 1.0 / self.display_fps
        last_frame_time = 0
        
        while self._running:
            current_time = time.time()
            
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.001)
                continue
            
            last_frame_time = current_time
            
            try:
                frames = []
                
                if self.sam3 and self.sam3._running:
                    try:
                        sam3_frame = self.sam3.get_display_frame()
                        frames.append(("SAM3 View", sam3_frame))
                    except:
                        pass
                
                if self.eye_in_hand and self.eye_in_hand._running:
                    try:
                        eih_frame = self.eye_in_hand.get_display_frame()
                        frames.append(("Eye-in-Hand", eih_frame))
                    except:
                        pass
                
                if not frames:
                    time.sleep(0.01)
                    continue
                
                if self.combined_view and len(frames) >= 2:
                    h1, w1 = frames[0][1].shape[:2]
                    h2, w2 = frames[1][1].shape[:2]
                    
                    target_h = min(h1, h2)
                    if h1 != target_h:
                        scale = target_h / h1
                        frames[0] = (frames[0][0], cv2.resize(frames[0][1], None, fx=scale, fy=scale))
                    if h2 != target_h:
                        scale = target_h / h2
                        frames[1] = (frames[1][0], cv2.resize(frames[1][1], None, fx=scale, fy=scale))
                    
                    combined = np.hstack([frames[0][1], frames[1][1]])
                    h, w = combined.shape[:2]
                    cv2.line(combined, (w//2, 0), (w//2, h), (255, 255, 255), 2)
                    cv2.putText(combined, "Press 's' to segment | 'q' to quit", 
                               (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
                    cv2.imshow(window_name, combined)
                    cv2.resizeWindow(window_name, w, h)
                    
                elif len(frames) == 1:
                    cv2.imshow(window_name, frames[0][1])
                
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    with self._key_lock:
                        self._key_queue.append(key)
                    
                    if key == ord('q'):
                        print("[DisplayManager] 用户按下 'q'，准备退出...")
                        self._running = False
                    elif key == ord('s'):
                        if self.sam3:
                            self.sam3.trigger_segment()
                
            except Exception as e:
                print(f"[DisplayManager] 显示错误: {e}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
        print(f"[DisplayManager] 显示线程结束")
    
    def start(self):
        """启动"""
        if self._running:
            return
        
        self._running = True
        self._display_thread = threading.Thread(target=self._display_thread_func, daemon=True)
        self._display_thread.start()
        time.sleep(0.3)
        print(f"[DisplayManager] 启动完成")
    
    def stop(self):
        """停止"""
        self._running = False
        if self._display_thread:
            self._display_thread.join(timeout=2.0)
        cv2.destroyAllWindows()
    
    def is_running(self) -> bool:
        """检查运行状态"""
        return self._running
    
    def get_key(self) -> Optional[int]:
        """获取按键"""
        with self._key_lock:
            if self._key_queue:
                return self._key_queue.pop(0)
        return None
    
    def close(self):
        """关闭"""
        self.stop()
        print(f"[DisplayManager] 已关闭")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.close()
        return False