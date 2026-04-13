import os
import tempfile
import torchaudio
import uuid
import sys
import shutil
from collections.abc import Mapping
from datetime import datetime

# Function to find ComfyUI directories
def get_comfyui_base_dir():
    """获取 ComfyUI 的根目录"""
    try:
        import folder_paths
        comfy_base = getattr(folder_paths, 'base_path', None)
        if comfy_base and os.path.exists(comfy_base):
            return comfy_base
        
        comfy_dir = os.path.dirname(os.path.dirname(os.path.abspath(folder_paths.__file__)))
        if os.path.exists(comfy_dir):
            return comfy_dir
    except:
        pass
    
    comfy_base = os.environ.get('ComfyUIBaseDir')
    if comfy_base and os.path.exists(comfy_base):
        return comfy_base
    
    return None

def get_latentsync_models_dir():
    """获取 LatentSync 模型的统一存放目录"""
    comfy_base = get_comfyui_base_dir()
    if comfy_base:
        latentsync_models_dir = os.path.join(comfy_base, "models", "LatentSyncModels")
        os.makedirs(latentsync_models_dir, exist_ok=True)
        return latentsync_models_dir
    return None

def get_latentsync_model_path(model_subpath):
    """获取 LatentSync 模型文件的完整路径"""
    models_dir = get_latentsync_models_dir()
    if models_dir:
        return os.path.join(models_dir, model_subpath)
    return None

def get_comfyui_temp_dir():
    """获取 ComfyUI 的临时目录"""
    comfy_base = get_comfyui_base_dir()
    if comfy_base:
        temp_dir = os.path.join(comfy_base, "temp")
        return temp_dir
    return tempfile.gettempdir()

def cleanup_comfyui_temp_directories():
    """清理 ComfyUI 临时目录"""
    comfyui_temp = get_comfyui_temp_dir()
    if os.path.exists(comfyui_temp):
        try:
            shutil.rmtree(comfyui_temp)
            print(f"Removed ComfyUI temp directory: {comfyui_temp}")
        except Exception as e:
            print(f"Could not remove {comfyui_temp}: {str(e)}")
            try:
                backup_name = f"{comfyui_temp}_backup_{uuid.uuid4().hex[:8]}"
                os.rename(comfyui_temp, backup_name)
                print(f"Renamed {comfyui_temp} to {backup_name}")
            except:
                pass

def init_temp_directories():
    """初始化临时目录设置"""
    cleanup_comfyui_temp_directories()
    
    system_temp = tempfile.gettempdir()
    unique_id = str(uuid.uuid4())[:8]
    temp_base_path = os.path.join(system_temp, f"latentsync_{unique_id}")
    os.makedirs(temp_base_path, exist_ok=True)
    
    os.environ['TMPDIR'] = temp_base_path
    os.environ['TEMP'] = temp_base_path
    os.environ['TMP'] = temp_base_path
    tempfile.tempdir = temp_base_path
    
    print(f"Set up system temp directory: {temp_base_path}")
    return temp_base_path

def module_cleanup():
    """模块退出时清理资源"""
    global MODULE_TEMP_DIR
    
    if MODULE_TEMP_DIR and os.path.exists(MODULE_TEMP_DIR):
        try:
            shutil.rmtree(MODULE_TEMP_DIR, ignore_errors=True)
            print(f"Cleaned up module temp directory: {MODULE_TEMP_DIR}")
        except:
            pass
    
    cleanup_comfyui_temp_directories()

MODULE_TEMP_DIR = init_temp_directories()

import atexit
atexit.register(module_cleanup)

import math
import torch
import random
import torchaudio
import folder_paths
import numpy as np
import platform
import subprocess
import importlib.util
import importlib.machinery
import argparse
from omegaconf import OmegaConf
from PIL import Image
from decimal import Decimal, ROUND_UP

if hasattr(folder_paths, "get_temp_directory"):
    original_get_temp = folder_paths.get_temp_directory
    folder_paths.get_temp_directory = lambda: MODULE_TEMP_DIR
else:
    setattr(folder_paths, 'get_temp_directory', lambda: MODULE_TEMP_DIR)

def import_inference_script(script_path):
    """导入推理脚本"""
    if not os.path.exists(script_path):
        raise ImportError(f"Script not found: {script_path}")

    module_name = "latentsync_inference"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None:
        raise ImportError(f"Failed to create module spec for {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        del sys.modules[module_name]
        raise ImportError(f"Failed to execute module: {str(e)}")

    return module

def check_ffmpeg():
    """检查 ffmpeg 是否可用"""
    try:
        if platform.system() == "Windows":
            ffmpeg_path = shutil.which("ffmpeg.exe")
            if ffmpeg_path is None:
                possible_paths = [
                    os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "ffmpeg", "bin"),
                    os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "ffmpeg", "bin"),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg", "bin"),
                ]
                for path in possible_paths:
                    if os.path.exists(os.path.join(path, "ffmpeg.exe")):
                        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
                        return True
                print("FFmpeg not found. Please install FFmpeg and add it to PATH")
                return False
            return True
        else:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg not found. Please install FFmpeg")
        return False

def check_and_install_dependencies():
    """检查并安装依赖"""
    if not check_ffmpeg():
        raise RuntimeError("FFmpeg is required but not found")

    required_packages = [
        'omegaconf',
        'pytorch_lightning',
        'transformers',
        'accelerate',
        'huggingface_hub',
        'einops',
        'diffusers',
        'ffmpeg-python',
        'imageio[ffmpeg]',
        'soundfile',
        'opencv-python'
    ]

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dependencies_installed_flag = os.path.join(cur_dir, ".latentsync_dependencies_installed")
    
    # 如果已经安装过，跳过
    if os.path.exists(dependencies_installed_flag):
        print("Dependencies already installed from previous run")
        return
    
    def is_package_installed(package_name):
        """检查包是否已安装 - 改进版"""
        # 处理带方括号的包名（如 imageio[ffmpeg]）
        if '[' in package_name:
            package_name = package_name.split('[')[0]
        
        # 处理带连字符的包名（如 ffmpeg-python）
        import_name = package_name.replace('-', '_')
        
        try:
            # 方法1: 尝试直接导入
            importlib.import_module(import_name)
            return True
        except ImportError:
            pass
        
        try:
            # 方法2: 使用 find_spec
            if importlib.util.find_spec(import_name) is not None:
                return True
        except:
            pass
        
        try:
            # 方法3: 检查 pip list
            import subprocess
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'show', package_name],
                capture_output=True, text=True
            )
            return result.returncode == 0
        except:
            pass
        
        return False

    def install_package(package):
        python_exe = sys.executable
        print(f"Installing {package}...")
        try:
            subprocess.check_call(
                [python_exe, '-m', 'pip', 'install', package],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"✓ Successfully installed {package}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Error installing {package}: {e}")
            return False

    # 检查缺失的包
    missing_packages = []
    for package in required_packages:
        pkg_name = package.split('[')[0] if '[' in package else package
        if not is_package_installed(pkg_name):
            missing_packages.append(package)
            print(f"  {pkg_name} - NOT FOUND")
        else:
            print(f"  {pkg_name} - ✓ INSTALLED")
    
    # 只安装缺失的包
    if missing_packages:
        print(f"\nInstalling missing dependencies: {', '.join(missing_packages)}")
        all_success = True
        for package in missing_packages:
            if not install_package(package):
                all_success = False
                print(f"Warning: Failed to install {package}")
        
        # 只有所有包都安装成功才创建标记文件
        if all_success:
            try:
                with open(dependencies_installed_flag, "w") as f:
                    f.write("Dependencies installed on " + str(datetime.now()))
                print("✓ Recorded dependencies installation status")
            except:
                print("Failed to create dependencies flag file")
    else:
        # 所有包都已安装，创建标记文件避免下次重复检查
        try:
            with open(dependencies_installed_flag, "w") as f:
                f.write("Dependencies checked on " + str(datetime.now()))
            print("✓ All dependencies are already installed")
        except:
            pass

def get_ext_dir(subpath=None, mkdir=False):
    """获取扩展目录路径"""
    dir = os.path.dirname(os.path.abspath(__file__))
    
    if subpath and ("temp" in subpath.lower() or "tmp" in subpath.lower()):
        global MODULE_TEMP_DIR
        sub_temp = os.path.join(MODULE_TEMP_DIR, subpath)
        if mkdir and not os.path.exists(sub_temp):
            os.makedirs(sub_temp, exist_ok=True)
        return sub_temp
    
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    
    return dir

def download_model_from_hf(repo_id, model_name, save_path):
    """从 HuggingFace 下载模型"""
    try:
        from huggingface_hub import hf_hub_download
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"Downloading {model_name} from {repo_id}...")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_name,
            local_dir=os.path.dirname(save_path),
            local_dir_use_symlinks=False
        )
        print(f"✓ Downloaded to: {downloaded_path}")
        return True
    except Exception as e:
        print(f"Failed to download {model_name}: {e}")
        return False

def get_insightface_root():
    """获取 insightface 根目录 - 保持原有路径 ComfyUI/models/insightface"""
    comfy_base = get_comfyui_base_dir()
    if comfy_base:
        insightface_root = os.path.join(comfy_base, "models", "insightface")
        if not os.path.exists(insightface_root):
            os.makedirs(insightface_root, exist_ok=True)
        return insightface_root
    
    # 回退到节点目录
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    fallback_path = os.path.join(cur_dir, "..", "checkpoints", "auxiliary")
    if not os.path.exists(fallback_path):
        os.makedirs(fallback_path, exist_ok=True)
    return fallback_path
    
def download_insightface_models():
    """下载 InsightFace buffalo_l 模型"""
    insightface_root = get_insightface_root()
    buffalo_dir = os.path.join(insightface_root, "models", "buffalo_l")
    
    # 检查 buffalo_l 是否完整
    required_files = ["det_10g.onnx", "w600k_r50.onnx", "2d106det.onnx"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(os.path.join(buffalo_dir, file)):
            missing_files.append(file)
    
    if not missing_files:
        print(f"✓ InsightFace buffalo_l models found at: {buffalo_dir}")
        return True
    
    print(f"\n⚠️ InsightFace buffalo_l models missing: {missing_files}")
    print(f"Target location: {buffalo_dir}")
    
    # 尝试自动下载 buffalo_l.zip
    print("Attempting to download InsightFace models...")
    
    try:
        import requests
        import zipfile
        
        zip_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
        zip_path = os.path.join(insightface_root, "buffalo_l.zip")
        
        print(f"Downloading from: {zip_url}")
        
        # 下载 zip 文件
        response = requests.get(zip_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"Downloaded: {percent:.1f}%", end='\r')
        
        print(f"\nDownloaded to: {zip_path}")
        
        # 解压文件
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(buffalo_dir)
        
        # 清理 zip 文件
        os.remove(zip_path)
        
        print(f"✓ InsightFace buffalo_l models downloaded and extracted to: {buffalo_dir}")
        return True
        
    except Exception as e:
        print(f"Failed to auto-download InsightFace models: {e}")
        print("\n" + "="*80)
        print("⚠️  Please manually download InsightFace buffalo_l models:")
        print("   Download from: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip")
        print(f"   Extract the zip file and place the contents in:")
        print(f"   {buffalo_dir}")
        print("\n   Required files:")
        for file in required_files:
            print(f"     - {file}")
        print("="*80 + "\n")
        return False
    
def setup_models():
    """设置并下载所有必需的模型"""
    latentsync_models_dir = get_latentsync_models_dir()
    
    if not latentsync_models_dir:
        raise RuntimeError("Cannot find ComfyUI models directory")
    
    # LatentSync 相关模型放到 LatentSyncModels 目录
    models_to_download = [
        {
            "name": "LatentSync UNet",
            "repo_id": "ByteDance/LatentSync-1.6",
            "filename": "latentsync_unet.pt",
            "subpath": "unet",
            "url": "https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/latentsync_unet.pt"
        },
        {
            "name": "Whisper tiny",
            "repo_id": "ByteDance/LatentSync-1.6",
            "filename": "tiny.pt",
            "subpath": "whisper",
            "url": "https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/whisper/tiny.pt"
        },
        {
            "name": "S3FD face detection",
            "repo_id": "vinthony/SadTalker",
            "filename": "s3fd-619a316812.pth",
            "subpath": "face_detection",
            "url": "https://huggingface.co/vinthony/SadTalker/resolve/main/hub/checkpoints/s3fd-619a316812.pth"
        }
    ]
    
    for model in models_to_download:
        model_path = os.path.join(latentsync_models_dir, model["subpath"], model["filename"])
        
        if os.path.exists(model_path):
            print(f"✓ {model['name']} found at: {model_path}")
            continue
        
        print(f"Downloading {model['name']}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id=model["repo_id"],
                filename=model["filename"],
                local_dir=os.path.join(latentsync_models_dir, model["subpath"]),
                local_dir_use_symlinks=False
            )
            print(f"✓ Downloaded {model['name']}")
        except Exception as e:
            print(f"Failed to download {model['name']}: {e}")
            print(f"Please manually download from: {model['url']}")
            print(f"and place at: {model_path}")
    
    # 下载 InsightFace buffalo_l 模型（保持原路径）
    download_insightface_models()

def get_latentsync_config_path(cur_dir):
    """获取 LatentSync 配置文件路径"""
    config_512 = os.path.join(cur_dir, "configs", "unet", "stage2_512.yaml")
    if os.path.exists(config_512):
        print("Using LatentSync 1.6 config (512x512)")
        return config_512
    
    config_256 = os.path.join(cur_dir, "configs", "unet", "stage2.yaml")
    if os.path.exists(config_256):
        print("Using LatentSync 1.5 config (256x256)")
        return config_256
    
    print("Config files not found, defaulting to LatentSync 1.6 config")
    return config_512

class LatentSyncNode:
    def __init__(self):
        global MODULE_TEMP_DIR
        if not os.path.exists(MODULE_TEMP_DIR):
            os.makedirs(MODULE_TEMP_DIR, exist_ok=True)
        
        check_and_install_dependencies()
        setup_models()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE",),
                    "audio": ("AUDIO", ),
                    "seed": ("INT", {"default": 1247}),
                    "lips_expression": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1}),
                    "inference_steps": ("INT", {"default": 20, "min": 1, "max": 999, "step": 1}),
                 },}

    CATEGORY = "LatentSyncNode"
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio") 
    FUNCTION = "inference"

    def save_video_with_imageio(self, video_path, frames, fps=25):
        """使用 imageio 保存视频"""
        import imageio
        
        # 确保 frames 是 numpy 数组
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
        
        # 确保数据类型正确
        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8)
        
        writer = imageio.get_writer(video_path, fps=fps, codec='libx264', macro_block_size=1)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"Video saved with imageio to {video_path}")

    def save_audio_with_soundfile(self, audio_path, waveform, sample_rate):
        """使用 soundfile 保存音频"""
        import soundfile as sf
        
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        
        if len(waveform.shape) == 2:
            if waveform.shape[0] == 1:
                waveform = waveform.squeeze(0)
            elif waveform.shape[0] < waveform.shape[1]:
                waveform = waveform.T
        
        sf.write(audio_path, waveform, sample_rate)
        print(f"Audio saved with soundfile to {audio_path}")

    def read_video_with_imageio(self, video_path):
        """使用 imageio 读取视频"""
        import imageio
        
        reader = imageio.get_reader(video_path)
        frames = []
        for frame in reader:
            frames.append(frame)
        reader.close()
        return np.array(frames)

    def inference(self, images, audio, seed, lips_expression=1.5, inference_steps=20):
        global MODULE_TEMP_DIR
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        BATCH_SIZE = 4
        use_mixed_precision = False
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            gpu_mem_gb = gpu_mem / (1024 ** 3)
    
            if gpu_mem_gb > 20:
                BATCH_SIZE = 32
                enable_tf32 = True
                use_mixed_precision = True
            elif gpu_mem_gb > 8:
                BATCH_SIZE = 16
                enable_tf32 = False
                use_mixed_precision = True
            else:
                BATCH_SIZE = 8
                enable_tf32 = False
                use_mixed_precision = False
    
            torch.backends.cudnn.benchmark = True
            if enable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
    
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.8)
    
        run_id = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        temp_dir = os.path.join(MODULE_TEMP_DIR, f"run_{run_id}")
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_video_path = None
        output_video_path = None
        audio_path = None
    
        try:
            temp_video_path = os.path.join(temp_dir, f"temp_{run_id}.mp4")
            output_video_path = os.path.join(temp_dir, f"latentsync_{run_id}_out.mp4")
            audio_path = os.path.join(temp_dir, f"latentsync_{run_id}_audio.wav")
            
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 处理输入帧
            if isinstance(images, list):
                frames = torch.stack(images).to(device)
            else:
                frames = images.to(device)
     
            frames_cpu = frames.cpu()  
            del frames  
            torch.cuda.empty_cache()  
    
            frames = (frames_cpu * 255).to(torch.uint8)
    
            # 处理音频
            waveform = audio["waveform"].to(device)
            sample_rate = audio["sample_rate"]
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)
    
            if sample_rate != 16000:
                new_sample_rate = 16000
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=new_sample_rate
                ).to(device)
                waveform_16k = resampler(waveform)
                waveform, sample_rate = waveform_16k, new_sample_rate
    
            resampled_audio = {
                "waveform": waveform.unsqueeze(0),
                "sample_rate": sample_rate
            }
            
            # 保存音频
            self.save_audio_with_soundfile(audio_path, waveform, sample_rate)
    
            # 保存视频
            self.save_video_with_imageio(temp_video_path, frames, fps=25)
    
            # 获取模型路径
            latentsync_models_dir = get_latentsync_models_dir()
            if not latentsync_models_dir:
                raise RuntimeError("Cannot find LatentSync models directory")
    
            inference_script_path = os.path.join(cur_dir, "scripts", "inference.py")
            config_path = get_latentsync_config_path(cur_dir)
            scheduler_config_path = os.path.join(cur_dir, "configs")
            ckpt_path = os.path.join(latentsync_models_dir, "unet", "latentsync_unet.pt")
            whisper_ckpt_path = os.path.join(latentsync_models_dir, "whisper", "tiny.pt")
    
            # 检查模型文件
            if not os.path.exists(ckpt_path):
                raise RuntimeError(f"UNet model not found at: {ckpt_path}")
            if not os.path.exists(whisper_ckpt_path):
                raise RuntimeError(f"Whisper model not found at: {whisper_ckpt_path}")
    
            config = OmegaConf.load(config_path)
    
            mask_image_path = os.path.join(cur_dir, "latentsync", "utils", "mask.png")
            if not os.path.exists(mask_image_path):
                alt_mask_path = os.path.join(cur_dir, "utils", "mask.png")
                if os.path.exists(alt_mask_path):
                    mask_image_path = alt_mask_path
    
            if hasattr(config, "data") and hasattr(config.data, "mask_image_path"):
                config.data.mask_image_path = mask_image_path
    
            args = argparse.Namespace(
                unet_config_path=config_path,
                inference_ckpt_path=ckpt_path,
                video_path=temp_video_path,
                audio_path=audio_path,
                video_out_path=output_video_path,
                seed=seed,
                inference_steps=inference_steps,
                guidance_scale=lips_expression,
                scheduler_config_path=scheduler_config_path,
                whisper_ckpt_path=whisper_ckpt_path,
                device=device,
                batch_size=BATCH_SIZE,
                use_mixed_precision=use_mixed_precision,
                temp_dir=temp_dir,
                mask_image_path=mask_image_path,
                extension_dir=cur_dir,
            )
    
            package_root = os.path.dirname(cur_dir)
            if package_root not in sys.path:
                sys.path.insert(0, package_root)
            if cur_dir not in sys.path:
                sys.path.insert(0, cur_dir)
    
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
            inference_module = import_inference_script(inference_script_path)
            
            inference_temp = os.path.join(temp_dir, "temp")
            os.makedirs(inference_temp, exist_ok=True)
            
            inference_module.main(config, args)
    
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
            if not os.path.exists(output_video_path):
                raise FileNotFoundError(f"Output video not found at: {output_video_path}")
            
            # 读取输出视频
            processed_frames = self.read_video_with_imageio(output_video_path)
            processed_frames = torch.from_numpy(processed_frames).float() / 255.0
            
            # 确保返回的 tensor 在 CPU 上（ComfyUI 期望 CPU tensor）
            if processed_frames.device.type != 'cpu':
                processed_frames = processed_frames.cpu()
            
            # 确保音频也在 CPU 上
            if resampled_audio["waveform"].device.type != 'cpu':
                resampled_audio["waveform"] = resampled_audio["waveform"].cpu()
    
            return (processed_frames, resampled_audio)
    
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
        finally:
            for path in [temp_video_path, output_video_path, audio_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        print(f"Removed temporary file: {path}")
                    except Exception as e:
                        print(f"Failed to remove {path}: {str(e)}")
    
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    print(f"Removed run temporary directory: {temp_dir}")
                except Exception as e:
                    print(f"Failed to remove temp run directory: {str(e)}")
    
            cleanup_comfyui_temp_directories()
    
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class VideoLengthAdjuster:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "mode": (["normal", "pingpong", "loop_to_audio"], {"default": "normal"}),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0}),
                "silent_padding_sec": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 3.0, "step": 0.1}),
            }
        }

    CATEGORY = "LatentSyncNode"
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "adjust"

    def adjust(self, images, audio, mode, fps=25.0, silent_padding_sec=0.5):
        waveform = audio["waveform"].squeeze(0)
        sample_rate = int(audio["sample_rate"])
        original_frames = [images[i] for i in range(images.shape[0])] if isinstance(images, torch.Tensor) else images.copy()

        if mode == "normal":
            audio_duration = waveform.shape[1] / sample_rate
            silence_samples = math.ceil(silent_padding_sec * sample_rate)
            silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)
            padded_audio = torch.cat([waveform, silence], dim=1)
            padded_audio_duration = (waveform.shape[1] + silence_samples) / sample_rate
            required_frames = int(padded_audio_duration * fps)
            
            if len(original_frames) > required_frames:
                adjusted_frames = original_frames[:required_frames]
            else:
                adjusted_frames = original_frames
                required_samples = int(len(original_frames) / fps * sample_rate)
                padded_audio = padded_audio[:, :required_samples]
            
            return (
                torch.stack(adjusted_frames),
                {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
            )

        elif mode == "pingpong":
            video_duration = len(original_frames) / fps
            audio_duration = waveform.shape[1] / sample_rate
            if audio_duration <= video_duration:
                required_samples = int(video_duration * sample_rate)
                silence = torch.zeros((waveform.shape[0], required_samples - waveform.shape[1]), dtype=waveform.dtype)
                adjusted_audio = torch.cat([waveform, silence], dim=1)

                return (
                    torch.stack(original_frames),
                    {"waveform": adjusted_audio.unsqueeze(0), "sample_rate": sample_rate}
                )

            else:
                silence_samples = math.ceil(silent_padding_sec * sample_rate)
                silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)
                padded_audio = torch.cat([waveform, silence], dim=1)
                total_duration = (waveform.shape[1] + silence_samples) / sample_rate
                target_frames = math.ceil(total_duration * fps)
                reversed_frames = original_frames[::-1][1:-1]
                frames = original_frames + reversed_frames
                while len(frames) < target_frames:
                    frames += frames[:target_frames - len(frames)]
                return (
                    torch.stack(frames[:target_frames]),
                    {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
                )

        elif mode == "loop_to_audio":
            silence_samples = math.ceil(silent_padding_sec * sample_rate)
            silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)
            padded_audio = torch.cat([waveform, silence], dim=1)
            total_duration = (waveform.shape[1] + silence_samples) / sample_rate
            target_frames = math.ceil(total_duration * fps)

            frames = original_frames.copy()
            while len(frames) < target_frames:
                frames += original_frames[:target_frames - len(frames)]
            
            return (
                torch.stack(frames[:target_frames]),
                {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
            )


NODE_CLASS_MAPPINGS = {
    "LatentSyncNode": LatentSyncNode,
    "VideoLengthAdjuster": VideoLengthAdjuster,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentSyncNode": "LatentSync1.6 Node",
    "VideoLengthAdjuster": "Video Length Adjuster",
}