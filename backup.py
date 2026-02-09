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


class GeminiImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (
                    [
                        "gemini-3-pro-image-preview",
                        "gemini-2.5-flash-image-preview"
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
                "temperature": (
                    "FLOAT",
                    {"default": 1, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
            },
            "optional": {
                "seed": ("INT", {"default": 66666666, "min": 0, "max": 2147483647}),
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "API Respond")
    FUNCTION = "generate_image"
    CATEGORY = "WaaS-AiGate"

    def __init__(self):
        """初始化日志系统和API密钥存储"""
        self.log_messages = []  # 全局日志消息存储
        # 获取节点所在目录
        self.node_dir = os.path.dirname(os.path.abspath(__file__))
        self.key_file = os.path.join(self.node_dir, "gemini_api_key.txt")
        # 固定API地址
        self.api_base_url = "https://api.chaojizhiti.com"

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
            traceback.self.log_exc()
            return None

    def generate_image(
        self,
        prompt,
        api_key,
        model,
        aspect_ratio,
        temperature,
        seed=66666666,
        images=None,
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

            # 设置API请求头
            headers = {
                "Authorization": f"Bearer {actual_api_key}",
                "Content-Type": "application/json",
            }

            # 处理种子值
            if seed == 0:
                import random

                seed = random.randint(1, 2**31 - 1)
                self.log(f"生成随机种子值: {seed}")
            else:
                self.log(f"使用指定的种子值: {seed}")

            # 直接从选择确定方向和提示词
            if "Free" in aspect_ratio:
                # 不指定任何方向或比例限制
                simple_prompt = f"Create a detailed image of: {prompt}."
            elif "Landscape" in aspect_ratio:
                orientation = (
                    "wide rectangular image where width is greater than height"
                )
                simple_prompt = f"Generate the image as a {orientation}.Create a detailed image of: {prompt}."
            elif "Portrait" in aspect_ratio:
                orientation = (
                    "tall rectangular image where height is greater than width"
                )
                simple_prompt = f"Generate the image as a {orientation}.Create a detailed image of: {prompt}."
            else:  # Square
                orientation = "square image where width equals height"
                simple_prompt = f"Generate the image as a {orientation}.Create a detailed image of: {prompt}."

            # 记录温度设置
            self.log(f"使用温度值: {temperature}，种子值: {seed}")

            # 构造OpenAI兼容的消息格式
            message_content = []
            reference_images_count = 0

            # 处理参考图像(单张或多张) - 转换为base64格式
            if images is not None:
                try:
                    # 确定图像数量
                    batch_size = images.shape[0]
                    self.log(f"检测到 {batch_size} 张参考图像")

                    # 逐一处理每张图像
                    for i in range(batch_size):
                        # 获取单张图像
                        input_image = images[i].cpu().numpy()

                        # 转换为PIL图像
                        input_image = (input_image * 255).astype(np.uint8)
                        pil_image = Image.fromarray(input_image)

                        self.log(
                            f"参考图像 {i + 1} 处理成功，尺寸: {pil_image.width}x{pil_image.height}"
                        )

                        # 转换为base64
                        img_byte_arr = BytesIO()
                        pil_image.save(img_byte_arr, format="PNG")
                        img_byte_arr.seek(0)
                        image_base64 = base64.b64encode(img_byte_arr.read()).decode(
                            "utf-8"
                        )

                        # 添加图像到消息内容
                        message_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            }
                        )
                        reference_images_count += 1

                    self.log(f"成功添加 {reference_images_count} 张参考图像到请求中")
                except Exception as img_error:
                    self.log(f"参考图像处理错误: {str(img_error)}")

            # 如果有参考图像，添加说明
            if reference_images_count > 0:
                if reference_images_count == 1:
                    simple_prompt += " Use this reference image as guidance."
                else:
                    simple_prompt += f" Use these {reference_images_count} reference images as guidance."

            # 添加文本提示
            message_content.append({"type": "text", "text": simple_prompt})

            # 构造请求体
            payload = {
                "model": model.replace("models/", ""),  # 移除 "models/" 前缀
                "messages": [{"role": "user", "content": message_content}],
                "temperature": temperature,
                "seed": seed,
            }

            # 打印请求信息
            self.log(
                f"请求API生成图像，种子值: {seed}, 包含参考图像数: {reference_images_count}"
            )
            self.log(f"API地址: {self.api_base_url}")

            # 调用API
            response = requests.post(
                f"{self.api_base_url}",
                headers=headers,
                json=payload,
                timeout=120,
            )

            # 检查响应状态
            if response.status_code != 200:
                error_msg = f"API请求失败，状态码: {response.status_code}, 响应: {response.text}"
                self.log(error_msg)
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
            self.log("API响应接收成功，正在处理...")

            # 检查响应格式
            if "choices" not in response_data or len(response_data["choices"]) == 0:
                self.log("API响应格式错误：没有choices字段")
                full_text = (
                    "## 处理日志\n"
                    + "\n".join(self.log_messages)
                    + "\n\n## 错误\nAPI返回了空响应或格式错误"
                )
                return (self.generate_empty_image(512, 512), full_text)

            # 获取响应内容
            choice = response_data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content", "")

            # 记录响应文本
            if isinstance(content, str):
                response_text = content
                self.log(
                    f"API返回文本: {content[:200]}..."
                    if len(content) > 200
                    else content
                )
            elif isinstance(content, list):
                # 处理多模态内容
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            response_text += item.get("text", "")
                        elif item.get("type") == "image_url":
                            response_text += "[图像]"

            # 检查是否包含图像数据
            image_found = False

            # 尝试从不同位置提取图像
            # 方式1: 检查content中的image_url
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        image_url = item.get("image_url", {}).get("url", "")
                        if image_url:
                            self.log("发现图像URL，正在处理...")
                            try:
                                # 处理base64编码的图像
                                if image_url.startswith("data:image"):
                                    # 提取base64数据
                                    base64_data = (
                                        image_url.split(",")[1]
                                        if "," in image_url
                                        else image_url
                                    )
                                    image_data = base64.b64decode(base64_data)

                                    # 转换为PIL图像
                                    buffer = BytesIO(image_data)
                                    pil_image = Image.open(buffer)
                                    self.log(
                                        f"成功解析图像: {pil_image.width}x{pil_image.height}, 格式: {pil_image.format}"
                                    )

                                    # 确保是RGB模式
                                    if pil_image.mode != "RGB":
                                        pil_image = pil_image.convert("RGB")

                                    # 转换为ComfyUI格式
                                    img_array = (
                                        np.array(pil_image).astype(np.float32) / 255.0
                                    )
                                    img_tensor = torch.from_numpy(img_array).unsqueeze(
                                        0
                                    )

                                    self.log(
                                        f"图像转换为张量成功, 形状: {img_tensor.shape}"
                                    )
                                    image_found = True

                                    # 合并日志和API返回文本
                                    full_text = (
                                        "## 处理日志\n"
                                        + "\n".join(self.log_messages)
                                        + "\n\n## API返回\n"
                                        + response_text
                                    )
                                    return (img_tensor, full_text)
                                else:
                                    # 从URL下载图像
                                    self.log(f"从URL下载图像: {image_url}")
                                    img_response = requests.get(image_url, timeout=30)
                                    if img_response.status_code == 200:
                                        buffer = BytesIO(img_response.content)
                                        pil_image = Image.open(buffer)
                                        self.log(
                                            f"成功下载图像: {pil_image.width}x{pil_image.height}"
                                        )

                                        # 确保是RGB模式
                                        if pil_image.mode != "RGB":
                                            pil_image = pil_image.convert("RGB")

                                        # 转换为ComfyUI格式
                                        img_array = (
                                            np.array(pil_image).astype(np.float32)
                                            / 255.0
                                        )
                                        img_tensor = torch.from_numpy(
                                            img_array
                                        ).unsqueeze(0)

                                        image_found = True
                                        full_text = (
                                            "## 处理日志\n"
                                            + "\n".join(self.log_messages)
                                            + "\n\n## API返回\n"
                                            + response_text
                                        )
                                        return (img_tensor, full_text)
                            except Exception as e:
                                self.log(f"处理图像URL失败: {str(e)}")
                                traceback.self.log_exc()

            # 方式2: 检查是否有data字段（类似DALL-E格式）
            if "data" in response_data:
                for item in response_data["data"]:
                    if "b64_json" in item:
                        try:
                            self.log("发现base64编码的图像数据")
                            image_data = base64.b64decode(item["b64_json"])
                            buffer = BytesIO(image_data)
                            pil_image = Image.open(buffer)

                            if pil_image.mode != "RGB":
                                pil_image = pil_image.convert("RGB")

                            img_array = np.array(pil_image).astype(np.float32) / 255.0
                            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                            image_found = True

                            full_text = (
                                "## 处理日志\n"
                                + "\n".join(self.log_messages)
                                + "\n\n## API返回\n"
                                + response_text
                            )
                            return (img_tensor, full_text)
                        except Exception as e:
                            self.log(f"处理b64_json图像失败: {str(e)}")

                    elif "url" in item:
                        try:
                            self.log(f"从data字段下载图像URL: {item['url']}")
                            img_response = requests.get(item["url"], timeout=30)
                            if img_response.status_code == 200:
                                buffer = BytesIO(img_response.content)
                                pil_image = Image.open(buffer)

                                if pil_image.mode != "RGB":
                                    pil_image = pil_image.convert("RGB")

                                img_array = (
                                    np.array(pil_image).astype(np.float32) / 255.0
                                )
                                img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                                image_found = True

                                full_text = (
                                    "## 处理日志\n"
                                    + "\n".join(self.log_messages)
                                    + "\n\n## API返回\n"
                                    + response_text
                                )
                                return (img_tensor, full_text)
                        except Exception as e:
                            self.log(f"从URL下载图像失败: {str(e)}")

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


# 注册节点
NODE_CLASS_MAPPINGS = {"WaaS-AiGate": GeminiImageGenerator}

NODE_DISPLAY_NAME_MAPPINGS = {"WaaS-AiGate": "WaaS AiGate image"}
