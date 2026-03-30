import json
from .BaseImageGenerator import BaseImageGenerator


class ImageGeneratorTxt2img(BaseImageGenerator):
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
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "☁️ 云扉AiGate"

    def generate_image(
        self,
        prompt,
        settings,
        model,
        aspect_ratio,
        image_size,
    ):
        """生成图像 - 纯文本到图像"""

        try:
            # 验证prompt不为空
            if not prompt or not prompt.strip():
                error_message = "错误: Prompt不能为空，请输入描述文字"
                print(error_message)
                raise Exception(error_message)
            # 从settings中提取API密钥
            actual_api_key = (
                settings.get("apiKey", "") if isinstance(settings, dict) else ""
            )

            if not actual_api_key:
                error_message = (
                    "错误: 未提供有效的API密钥。请在节点中输入API密钥或确保已保存密钥。"
                )
                print(error_message)
                raise Exception(error_message)

            # 设置API请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": actual_api_key,
            }

            # 获取ComfyUI的job id并添加到headers
            job_id = self.get_job_id()
            if job_id:
                headers["Job-ID"] = str(job_id)
                print(f"Job-ID: {job_id}")

            # 构造请求体（仅包含文本提示）
            parts = [{"text": prompt}]

            # 构造API payload
            payload = self.build_api_payload(parts, aspect_ratio, image_size)

            # 将模型显示名称转换为模型ID
            model_id = self.get_model_pid(model)

            # 根据选择的模型构造API地址
            self.api_base_url = self.api_base_url_template.format(model=model_id)
            print(f"使用模型: {model} (ID: {model_id})")
            print(f"API地址: {self.api_base_url}")

            # 构造API URL（不需要在URL中添加key参数）
            api_url = f"{self.api_base_url}?code={model_id}"

            # 调试输出：打印请求信息
            # print(f"\n=== API请求调试信息 ===")
            # print(f"URL: {api_url}")
            # print(f"Headers: {json.dumps(headers, indent=2)}")
            # print(f"Payload: {json.dumps(payload, indent=2)}")
            # print(f"========================\n")

            # 调用API
            print("正在调用API...")
            response = self.call_api(api_url, headers, payload)

            # 检查响应状态
            if response.status_code != 200:
                error_msg = f"API请求失败，状态码: {response.status_code}, 响应: {response.text}"
                print(error_msg)
                raise Exception(error_msg)

            # 解析响应
            response_data = response.json()

            # 响应处理
            print("API响应接收成功，正在处理...")

            # 处理响应
            img_tensor, response_text, model_version = self.process_response(
                response_data.get("data", {})
            )

            # 检查响应格式
            if response_text is None:
                raise Exception("API返回了空响应或格式错误")

            # 如果没有找到图像，返回空图像和日志
            if img_tensor is None:
                if not response_text:
                    response_text = "API未返回任何图像或文本"
                return (self.generate_empty_image(512, 512),)

            # 返回成功响应
            return (img_tensor,)

        except Exception as e:
            error_message = f"处理过程中出错: {str(e)}"
            print(f"图像生成错误: {str(e)}")
            raise Exception(error_message)
