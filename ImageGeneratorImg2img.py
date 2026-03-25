import base64
import json
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from .BaseImageGenerator import BaseImageGenerator


class ImageGeneratorImg2img(BaseImageGenerator):
    @classmethod
    def INPUT_TYPES(cls):
        model_choices = BaseImageGenerator.get_model_choices()
        default_model = model_choices[0] if model_choices else ""
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "settings": ("STRUCT",),
                "model": (
                    model_choices,
                    {"default": default_model},
                ),
                "aspect_ratio": (
                    [
                        "1:1",
                        "16:9",
                        "9:16",
                        "4:3",
                        "3:4",
                        "3:2",
                        "2:3",
                        "5:4",
                        "4:5",
                        "21:9",
                    ],
                    {"default": "1:1"},
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

    def generate_image(
        self,
        prompt,
        settings,
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
        # 重置日志消息
        self.log_messages = []

        try:
            # 验证prompt不为空
            if not prompt or not prompt.strip():
                error_message = "错误: Prompt不能为空，请输入描述文字"
                self.log(error_message)
                return self.get_error_response(error_message)
            # 从settings中提取API密钥
            api_key_from_settings = (
                settings.get("apiKey", "") if isinstance(settings, dict) else ""
            )
            actual_api_key = self.get_api_key(api_key_from_settings)

            if not actual_api_key:
                error_message = (
                    "错误: 未提供有效的API密钥。请在节点中输入API密钥或确保已保存密钥。"
                )
                self.log(error_message)
                return self.get_error_response(error_message)

            # 设置API请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": actual_api_key,
            }

            # 构造请求体
            parts = []

            # 收集所有图像（image1 到 image10）
            all_images = [
                image1,
                image2,
                image3,
                image4,
                image5,
                image6,
                image7,
                image8,
                image9,
                image10,
            ]

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

                            # 验证图像大小（不超过4MB）
                            is_valid, size_message = self.validate_image_size(
                                img_byte_arr, max_size_mb=10
                            )

                            if not is_valid:
                                raise Exception(size_message)

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

                        self.log(f"成功添加 image{idx} 的图像到请求中")
                    except Exception as img_error:
                        self.log(f"image{idx} 处理错误: {str(img_error)}")
                        raise Exception(img_error)

            # 添加文本提示
            parts.append({"text": prompt})

            # 构造API payload
            payload = self.build_api_payload(parts, aspect_ratio, image_size)

            # 将模型显示名称转换为模型ID
            model_id = self.get_model_pid(model)

            # 根据选择的模型构造API地址
            self.api_base_url = self.api_base_url_template.format(model=model_id)
            self.log(f"使用模型: {model} (ID: {model_id})")
            self.log(f"API地址: {self.api_base_url}")

            # 构造API URL（不需要在URL中添加key参数）
            api_url = f"{self.api_base_url}?code={model_id}"

            # 调试输出：打印请求信息
            # print(f"\n=== API请求调试信息 ===")
            # print(f"URL: {api_url}")
            # print(f"Headers: {json.dumps(headers, indent=2)}")
            # print(f"Payload: {json.dumps(payload, indent=2)}")
            # print(f"========================\n")

            # 调用API
            response = self.call_api(api_url, headers, payload)

            # 检查响应状态
            if response.status_code != 200:
                error_msg = f"API请求失败，状态码: {response.status_code}, 响应: {response.text}"
                self.log(error_msg)
                return self.get_error_response(error_msg)

            # 解析响应
            response_data = response.json()

            # 响应处理
            self.log("API响应接收成功，正在处理...")

            # 处理响应
            img_tensor, response_text, model_version = self.process_response(
                response_data.get("data", {})
            )

            # 检查响应格式
            if response_text is None:
                return self.get_error_response("API返回了空响应或格式错误")

            # 如果没有找到图像，返回空图像和日志
            if img_tensor is None:
                if not response_text:
                    response_text = "API未返回任何图像或文本"
                return self.get_success_response(
                    self.generate_empty_image(512, 512), response_text, model_version
                )

            # 返回成功响应
            return self.get_success_response(img_tensor, response_text, model_version)

        except Exception as e:
            error_message = f"处理过程中出错: {str(e)}"
            self.log(f"图像生成错误: {str(e)}")
            raise Exception(error_message)
            # return self.get_error_response(error_message)
