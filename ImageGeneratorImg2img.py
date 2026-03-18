import os
import base64
import io
import json
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import traceback


class ImageGeneratorImg2img:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (
                    [
                        "PRO-低价渠道版",
                        "PRO-官方稳定版",
                        "V2-低价渠道版",
                        "V2-官方稳定版",
                        "gemini-3-pro-image-preview",
                        "gemini-3.1-flash-image-preview",
                    ],
                    {"default": "gemini-3-pro-image-preview"},
                ),
                "aspect_ratio": (
                    [
                        "Free (自由比例)",
                        "Landscape (横屏)",
                        "Portrait (竖屏)",
                        "Square (方形)",
                    ],
                    {"default": "Free (自由比例)"},
                ),
                "image_size": (["1K", "2K"], {"default": "1K"}),
                "image1": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "API Respond")
    FUNCTION = "generate_image"
    CATEGORY = "云扉AiGate"

    def __init__(self):
        self.api_host = "https://api.chaojizhiti.com"
        """初始化日志系统和API密钥存储"""
        self.log_messages = []  # 全局日志消息存储
        # 获取节点所在目录
        self.node_dir = os.path.dirname(os.path.abspath(__file__))
        self.key_file = os.path.join(self.node_dir, "gemini_api_key.txt")
        # API地址配置
        self.api_base_url_template = (
            f"{self.api_host}/v1beta/models/{{model}}:generateContent"
        )

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

    def generate_image(
        self,
        prompt,
        api_key,
        model,
        aspect_ratio,
        image_size,
        image1,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        image7=None,
        image8=None,
        image9=None,
        image10=None,
    ):
        """生成图像 - 支持多张参考图片"""
        response_text = ""

        # 重置日志消息
        self.log_messages = []

        try:
            # 获取API密钥
            actual_api_key = self.get_api_key(api_key)

            if not actual_api_key:
                error_message = (
                    "错误: 未提供有效的API密钥。请在节点中输入API密钥或确保已保存密钥。"
                )
                self.log(error_message)
                full_text = (
                    "## 错误\n"
                    + error_message
                    + "\n\n## 使用说明\n1. 在节点中输入您的API密钥\n2. 密钥将自动保存到节点目录，下次可以不必输入"
                )
                return (
                    self.generate_empty_image(512, 512),
                    full_text,
                )  # 使用默认尺寸的空白图像

            # 设置API请求头（不需要Authorization，key通过query参数传递）
            headers = {
                "Content-Type": "application/json",
            }

            # 构造 Gemini API 格式的请求体
            parts = []
            reference_images_count = 0

            # 收集所有图像（image1 到 image10）
            all_images = [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10]

            # 处理参考图像(单张或多张) - 转换为内联数据格式
            for idx, images in enumerate(all_images, 1):
                if images is not None:
                    try:
                        # 确定图像数量
                        batch_size = images.shape[0]
                        self.log(f"检测到 image{idx} 有 {batch_size} 张图像")

                        # 逐一处理每张图像
                        for i in range(batch_size):
                            # 获取单张图像
                            input_image = images[i].cpu().numpy()

                            # 转换为PIL图像
                            input_image = (input_image * 255).astype(np.uint8)
                            pil_image = Image.fromarray(input_image)

                            self.log(
                                f"参考图像 image{idx}[{i + 1}] 处理成功，尺寸: {pil_image.width}x{pil_image.height}"
                            )

                            # 转换为base64
                            img_byte_arr = BytesIO()
                            pil_image.save(img_byte_arr, format="PNG")
                            img_byte_arr.seek(0)
                            image_base64 = base64.b64encode(img_byte_arr.read()).decode(
                                "utf-8"
                            )

                            # 添加图像到 parts（Gemini 格式）
                            parts.append(
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": image_base64,
                                    }
                                }
                            )
                            reference_images_count += 1

                        self.log(f"成功添加 image{idx} 的图像到请求中")
                    except Exception as img_error:
                        self.log(f"image{idx} 处理错误: {str(img_error)}")

            # 添加文本提示
            parts.append({"text": prompt})

            # 确定图像比例
            if "Free" in aspect_ratio:
                ratio = "1:1"  # 默认方形
            elif "Landscape" in aspect_ratio:
                ratio = "16:9"
            elif "Portrait" in aspect_ratio:
                ratio = "9:16"
            else:  # Square
                ratio = "1:1"

            # 构造请求体
            payload = {
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"],
                    "imageConfig": {"imageSize": image_size, "aspectRatio": ratio},
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

            # 根据选择的模型构造API地址
            self.api_base_url = self.api_base_url_template.format(model=model)
            self.log(f"API地址: {self.api_base_url}")

            # 构造带 API key 的完整 URL
            api_url_with_key = f"{self.api_base_url}?key={actual_api_key}"

            # 调用API
            response = requests.post(
                api_url_with_key,
                headers=headers,
                json=payload,
                timeout=120,
            )

            # 检查响应状态
            if response.status_code != 200:
                error_msg = f"API请求失败，状态码: {response.status_code}, 响应: {response.text}"
                print(error_msg)
                full_text = (
                    "## 处理日志\n"
                    + "\n".join(self.log_messages)
                    + "\n\n## 错误\n"
                    + error_msg
                )
                return (self.generate_empty_image(512, 512), full_text)

            # 解析响应
            response_data = response.json()

            # 响应处理
            print("API响应接收成功，正在处理...")

            # 检查响应格式（Gemini API 格式）
            if (
                "candidates" not in response_data
                or len(response_data["candidates"]) == 0
            ):
                self.log("API响应格式错误：没有candidates字段")
                full_text = (
                    "## 处理日志\n"
                    + "\n".join(self.log_messages)
                    + "\n\n## 错误\nAPI返回了空响应或格式错误"
                )
                return (self.generate_empty_image(512, 512), full_text)

            # 获取响应内容
            candidate = response_data["candidates"][0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            if not parts:
                self.log("API响应中没有parts内容")
                full_text = (
                    "## 处理日志\n"
                    + "\n".join(self.log_messages)
                    + "\n\n## 错误\nAPI响应中没有生成的内容"
                )
                return (self.generate_empty_image(512, 512), full_text)

            # 检查是否包含图像数据
            image_found = False
            response_text = ""

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
                                response_text += (
                                    f"Thought Signature: {thought_signature}\n"
                                )

                            # 合并日志和API返回文本
                            full_text = (
                                "## 处理日志\n"
                                + "\n".join(self.log_messages)
                                + "\n\n## API返回\n"
                                + response_text
                                + f"\n模型版本: {response_data.get('modelVersion', 'unknown')}"
                            )
                            return (img_tensor, full_text)
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

            # 合并日志和API返回文本
            full_text = (
                "## 处理日志\n"
                + "\n".join(self.log_messages)
                + "\n\n## API返回\n"
                + response_text
            )
            return (self.generate_empty_image(512, 512), full_text)

        except Exception as e:
            error_message = f"处理过程中出错: {str(e)}"
            self.log(f"Gemini图像生成错误: {str(e)}")

            # 合并日志和错误信息
            full_text = (
                "## 处理日志\n"
                + "\n".join(self.log_messages)
                + "\n\n## 错误\n"
                + error_message
            )
            return (self.generate_empty_image(512, 512), full_text)


