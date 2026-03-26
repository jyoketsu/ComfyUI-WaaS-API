import os
import base64
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import traceback
import json


class BaseImageGenerator:
    """图像生成器基类，包含所有共同的功能"""

    # 类级变量
    api_host = "http://waas.k8s.dev.inner"
    _models_cache = None  # 类级别缓存
    _models_map = {}  # pname -> pid 映射

    def __init__(self):
        """初始化日志系统和API密钥存储"""
        self.log_messages = []  # 全局日志消息存储
        # 获取节点所在目录
        self.node_dir = os.path.dirname(os.path.abspath(__file__))
        self.key_file = os.path.join(self.node_dir, "gemini_api_key.txt")
        # API地址配置
        self.api_base_url_template = f"{self.api_host}/api/generate/image/content"

        # 检查依赖库版本
        try:
            # 检查PIL/Pillow版本
            try:
                import PIL

                self.log(f"当前PIL/Pillow版本: {PIL.__version__}")
            except Exception as e:
                self.log(f"无法检查PIL/Pillow版本: {str(e)}")

            # 检查requests库
            try:
                import requests

                self.log(f"当前requests版本: {requests.__version__}")
            except Exception as e:
                self.log(f"无法检查requests版本: {str(e)}")
        except Exception as e:
            self.log(f"无法检查版本信息: {e}")

    def log(self, message):
        """全局日志函数：记录到日志列表"""
        if hasattr(self, "log_messages"):
            self.log_messages.append(message)
        return message

    def validate_image_size(self, img_byte_arr, max_size_mb=4):
        """
        验证图像大小是否超过限制

        Args:
            img_byte_arr: BytesIO对象，包含图像数据
            max_size_mb: 最大大小限制（MB），默认4MB

        Returns:
            (bool, str): (通过验证, 消息)
        """
        # 获取字节大小
        img_bytes = img_byte_arr.getvalue()
        size_bytes = len(img_bytes)
        size_mb = size_bytes / (1024 * 1024)
        max_size_bytes = max_size_mb * 1024 * 1024

        if size_bytes > max_size_bytes:
            message = f"图像大小 ({size_mb:.2f}MB) 超过限制 ({max_size_mb}MB)"
            return False, message

        return True, f"图像大小验证通过 ({size_mb:.2f}MB)"

    @classmethod
    def get_available_models(cls):
        """从API动态获取可用模型列表，返回格式：[{\"pname\": \"...\", \"pid\": \"...\"}, ...]"""
        default_models = []

        # 如果缓存存在，返回缓存
        if cls._models_cache:
            return cls._models_cache

        try:
            # 尝试从新API获取模型列表
            api_url = f"{cls.api_host}/api/generate/image/types"
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # 处理新的API响应格式：{"code": 0, "data": [...], "ok": true}
                if isinstance(data, dict) and data.get("ok"):
                    models = data.get("data", [])
                    if isinstance(models, list) and models:
                        # 构建映射
                        cls._models_map = {
                            m.get("pname", ""): m.get("pid", "") for m in models
                        }
                        cls._models_cache = models
                        return models
                # 兼容旧格式（直接返回数组）
                elif isinstance(data, list) and data:
                    cls._models_map = {
                        m.get("pname", ""): m.get("pid", "") for m in data
                    }
                    cls._models_cache = data
                    return data
        except Exception as e:
            cls.log(f"获取模型列表失败: {e}")

        # 返回默认列表
        cls._models_cache = default_models
        cls._models_map = {m.get("pname", ""): m.get("pid", "") for m in default_models}
        return default_models

    @classmethod
    def get_model_choices(cls):
        """获取UI显示的模型名称列表 (pname)"""
        models = cls.get_available_models()
        return [m.get("pname", "") for m in models if m.get("pname")]

    @classmethod
    def get_model_pid(cls, pname):
        """根据显示名称 (pname) 获取模型ID (pid)"""
        # 先确保映射已加载
        if not cls._models_map:
            cls.get_available_models()

        # 查找映射
        pid = cls._models_map.get(pname, "")
        if pid:
            return pid

        # 如果未找到，返回原始名称（向后兼容）
        return pname

    def get_api_key(self, user_input_key):
        """获取API密钥，优先使用用户输入的密钥"""
        # 如果用户输入了有效的密钥，使用并保存
        if user_input_key and len(user_input_key) > 10:
            self.log("使用用户输入的API密钥")
            # 保存到文件中
            try:
                with open(self.key_file, "w") as f:
                    f.write(user_input_key)
                self.log("已保存API密钥到节点目录")
            except Exception as e:
                self.log(f"保存API密钥失败: {e}")
            return user_input_key

        # 如果用户没有输入，尝试从文件读取
        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, "r") as f:
                    saved_key = f.read().strip()
                if saved_key and len(saved_key) > 10:
                    self.log("使用已保存的API密钥")
                    return saved_key
            except Exception as e:
                self.log(f"读取保存的API密钥失败: {e}")

        # 如果都没有，返回空字符串
        self.log("警告: 未提供有效的API密钥")
        return ""

    def generate_empty_image(self, width=512, height=512):
        """生成标准格式的空白RGB图像张量 - 使用默认尺寸"""
        # 根据比例设置默认尺寸
        empty_image = np.ones((height, width, 3), dtype=np.float32) * 0.2
        tensor = torch.from_numpy(empty_image).unsqueeze(0)  # [1, H, W, 3]

        self.log(f"创建ComfyUI兼容的空白图像: 形状={tensor.shape}, 类型={tensor.dtype}")
        return tensor

    def validate_and_fix_tensor(self, tensor, name="图像"):
        """验证并修复张量格式，确保完全兼容ComfyUI"""
        try:
            # 基本形状检查
            if tensor is None:
                self.log(f"警告: {name} 是None")
                return None

            self.log(
                f"验证 {name}: 形状={tensor.shape}, 类型={tensor.dtype}, 设备={tensor.device}"
            )

            # 确保形状正确: [B, C, H, W]
            if len(tensor.shape) != 4:
                self.log(f"错误: {name} 形状不正确: {tensor.shape}")
                return None

            if tensor.shape[1] != 3:
                self.log(f"错误: {name} 通道数不是3: {tensor.shape[1]}")
                return None

            # 确保类型为float32
            if tensor.dtype != torch.float32:
                self.log(f"修正 {name} 类型: {tensor.dtype} -> torch.float32")
                tensor = tensor.to(dtype=torch.float32)

            # 确保内存连续
            if not tensor.is_contiguous():
                self.log(f"修正 {name} 内存布局: 使其连续")
                tensor = tensor.contiguous()

            # 确保值范围在0-1之间
            min_val = tensor.min().item()
            max_val = tensor.max().item()

            if min_val < 0 or max_val > 1:
                self.log(f"修正 {name} 值范围: [{min_val}, {max_val}] -> [0, 1]")
                tensor = torch.clamp(tensor, 0.0, 1.0)

            return tensor
        except Exception as e:
            self.log(f"验证张量时出错: {e}")
            traceback.print_exc()
            return None

    def get_aspect_ratio(self, aspect_ratio):
        """根据比例字符串返回对应的比例值"""
        if "Free" in aspect_ratio:
            ratio = "1:1"  # 默认方形
        elif "Landscape" in aspect_ratio:
            ratio = "16:9"
        elif "Portrait" in aspect_ratio:
            ratio = "9:16"
        else:  # Square
            ratio = "1:1"
        return ratio

    def build_api_payload(self, parts, aspect_ratio, image_size):
        """构造API请求体"""
        # ratio = self.get_aspect_ratio(aspect_ratio)

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {"imageSize": image_size, "aspectRatio": aspect_ratio},
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ],
        }
        return payload

    def call_api(self, api_url, headers, payload):
        """调用API并返回响应"""
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=120,
        )

        # 检查HTTP状态码
        if response.status_code != 200:
            error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            raise Exception(f"API请求失败: {error_msg}")

        # 检查响应是否为空
        if not response.text:
            raise Exception("API返回空响应")

        try:
            res = response.json()
        except ValueError as e:
            raise Exception(f"JSON解析失败: {str(e)}\n响应内容: {response.text[:200]}")

        if res.get("ok"):
            return response
        else:
            raise Exception(res.get("msg"))

    def process_response(self, response_data):
        """处理API响应，提取图像和文本"""
        # 检查响应格式（Gemini API 格式）
        if "candidates" not in response_data or len(response_data["candidates"]) == 0:
            self.log("API响应格式错误：没有candidates字段")
            return None, None, None

        # 获取响应内容
        candidate = response_data["candidates"][0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        if not parts:
            self.log("API响应中没有parts内容")
            return None, None, None

        # 检查是否包含图像数据
        image_found = False
        response_text = ""
        img_tensor = None

        # 遍历 parts 寻找图像数据
        for part in parts:
            # 检查是否有 inlineData 字段
            if "inlineData" in part:
                try:
                    inline_data = part["inlineData"]
                    mime_type = inline_data.get("mimeType", "")
                    base64_data = inline_data.get("data", "")

                    if base64_data and "image" in mime_type:
                        self.log(f"发现内联图像数据，MIME类型: {mime_type}")

                        # 解码base64图像数据
                        image_data = base64.b64decode(base64_data)
                        buffer = BytesIO(image_data)
                        pil_image = Image.open(buffer)

                        self.log(
                            f"成功解析图像: {pil_image.width}x{pil_image.height}, 格式: {pil_image.format}"
                        )

                        # 确保是RGB模式
                        if pil_image.mode != "RGB":
                            pil_image = pil_image.convert("RGB")

                        # 转换为ComfyUI格式
                        img_array = np.array(pil_image).astype(np.float32) / 255.0
                        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

                        self.log(f"图像转换为张量成功, 形状: {img_tensor.shape}")
                        image_found = True

                        # 检查是否有 thoughtSignature
                        thought_signature = part.get("thoughtSignature", "")
                        if thought_signature:
                            response_text += f"Thought Signature: {thought_signature}\n"
                except Exception as e:
                    self.log(f"处理内联图像数据失败: {str(e)}")
                    traceback.print_exc()

            # 检查是否有文本内容
            elif "text" in part:
                response_text += part["text"] + "\n"

        # 没有找到图像数据
        if not image_found:
            self.log("API响应中未找到图像数据")
            if not response_text:
                response_text = "API未返回任何图像或文本"

        model_version = response_data.get("modelVersion", "unknown")
        return img_tensor, response_text, model_version

    def get_error_response(self, error_message):
        raise Exception(error_message)
        """返回错误响应"""
        # full_text = (
        #     "## 处理日志\n"
        #     + "\n".join(self.log_messages)
        #     + "\n\n## 错误\n"
        #     + error_message
        # )
        # return (self.generate_empty_image(512, 512), full_text)

    def get_success_response(self, img_tensor, response_text, model_version=""):
        """返回成功响应"""
        full_text = (
            "## 处理日志\n"
            + "\n".join(self.log_messages)
            + "\n\n## API返回\n"
            + response_text
        )
        if model_version:
            full_text += f"\n模型版本: {model_version}"
        return (img_tensor, full_text)
