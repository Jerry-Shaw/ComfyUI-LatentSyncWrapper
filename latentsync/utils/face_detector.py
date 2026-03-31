from insightface.app import FaceAnalysis
import numpy as np
import torch
import os
import sys
import traceback


def get_comfyui_base_dir():
    """获取 ComfyUI 的根目录"""
    try:
        import folder_paths
        comfy_base = getattr(folder_paths, 'base_path', None)
        if comfy_base and os.path.exists(comfy_base):
            return comfy_base
    except:
        pass
    
    comfy_base = os.environ.get('ComfyUIBaseDir')
    if comfy_base and os.path.exists(comfy_base):
        return comfy_base
    
    return None


def get_insightface_root():
    """获取 insightface 根目录 - 保持原有路径 ComfyUI/models/insightface"""
    comfy_base = get_comfyui_base_dir()
    if comfy_base:
        # 使用原有的 insightface 路径
        insightface_root = os.path.join(comfy_base, "models", "insightface")
        print(f"InsightFace root: {insightface_root}")
        return insightface_root
    
    # 回退到节点目录
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    fallback_path = os.path.join(cur_dir, "..", "checkpoints", "auxiliary")
    print(f"InsightFace root (fallback): {fallback_path}")
    return fallback_path


INSIGHTFACE_DETECT_SIZE = 512


class FaceDetector:
    def __init__(self, device="cuda"):
        self.device = device
        insightface_root = get_insightface_root()
        print(f"\nInitializing FaceDetector with device={device}")
        
        # 检查 buffalo_l 是否存在
        buffalo_path = os.path.join(insightface_root, "models", "buffalo_l")
        if not os.path.exists(buffalo_path):
            print(f"\n⚠️  Warning: buffalo_l model not found at: {buffalo_path}")
            print("   Face detection may not work correctly.")
            print("   Please download buffalo_l from:")
            print("   https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip")
            print("   and extract to: " + os.path.join(insightface_root, "models"))
            print("   Expected location: " + buffalo_path + "\n")
        
        # 设置提供者
        if device == "cuda" and torch.cuda.is_available():
            providers = ["CUDAExecutionProvider"]
            ctx_id = 0
            print("✓ Using CUDA for face detection")
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1
            print("⚠️ Using CPU for face detection (slower)")
        
        try:
            self.app = FaceAnalysis(
                allowed_modules=["detection", "landmark_2d_106"],
                root=insightface_root,
                providers=providers,
            )
            self.app.prepare(ctx_id=ctx_id, det_size=(INSIGHTFACE_DETECT_SIZE, INSIGHTFACE_DETECT_SIZE))
            print("✓ Face detector initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize face detector: {e}")
            self.app = None
            raise

    def __call__(self, frame, threshold=0.5):
        if self.app is None:
            print("Face detector not initialized")
            return None, None
        
        if frame is None:
            return None, None
        
        try:
            f_h, f_w, _ = frame.shape
            
            # 确保 frame 是 numpy 数组
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            
            faces = self.app.get(frame)

            get_face_store = None
            max_size = 0

            if len(faces) == 0:
                return None, None
            else:
                for face in faces:
                    bbox = face.bbox.astype(np.int_).tolist()
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    if w < 50 or h < 80:
                        continue
                    if w / h > 1.5 or w / h < 0.2:
                        continue
                    if face.det_score < threshold:
                        continue
                    size_now = w * h

                    if size_now > max_size:
                        max_size = size_now
                        get_face_store = face

            if get_face_store is None:
                return None, None
            else:
                face = get_face_store
                lmk = np.round(face.landmark_2d_106).astype(np.int_)

                # Calculate landmarks for alignment
                try:
                    pt_left_eye = np.mean(lmk[[43, 48, 49, 51, 50]], axis=0)
                    pt_right_eye = np.mean(lmk[101:106], axis=0)
                    pt_nose = np.mean(lmk[[74, 77, 83, 86]], axis=0)
                    
                    landmarks3 = np.round([pt_left_eye, pt_right_eye, pt_nose])
                    
                    # Calculate expanded bounding box
                    halk_face_coord = np.mean([lmk[74], lmk[73]], axis=0)
                    sub_lmk = lmk[LMK_ADAPT_ORIGIN_ORDER]
                    halk_face_dist = np.max(sub_lmk[:, 1]) - halk_face_coord[1]
                    upper_bond = halk_face_coord[1] - halk_face_dist

                    x1, y1, x2, y2 = (np.min(sub_lmk[:, 0]), int(upper_bond), np.max(sub_lmk[:, 0]), np.max(sub_lmk[:, 1]))

                    if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
                        x1, y1, x2, y2 = face.bbox.astype(np.int_).tolist()

                    y2 += int((x2 - x1) * 0.1)
                    x1 -= int((x2 - x1) * 0.05)
                    x2 += int((x2 - x1) * 0.05)

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(f_w, x2)
                    y2 = min(f_h, y2)

                    return (x1, y1, x2, y2), lmk
                    
                except Exception as e:
                    print(f"Error calculating landmarks: {e}")
                    # Fallback to basic bbox
                    bbox = face.bbox.astype(np.int_).tolist()
                    return (bbox[0], bbox[1], bbox[2], bbox[3]), lmk

        except Exception as e:
            print(f"Error during face detection: {e}")
            return None, None


def cuda_to_int(cuda_str: str) -> int:
    """
    Convert the string with format "cuda:X" to integer X.
    """
    if cuda_str == "cuda":
        return 0
    device = torch.device(cuda_str)
    if device.type != "cuda":
        return -1  # CPU mode
    return device.index


LMK_ADAPT_ORIGIN_ORDER = [
    1,
    10,
    12,
    14,
    16,
    3,
    5,
    7,
    0,
    23,
    21,
    19,
    32,
    30,
    28,
    26,
    17,
    43,
    48,
    49,
    51,
    50,
    102,
    103,
    104,
    105,
    101,
    73,
    74,
    86,
]