from .ImageGeneratorImg2img import ImageGeneratorImg2img

NODE_CLASS_MAPPINGS = {"aigate_img2img": ImageGeneratorImg2img}

NODE_DISPLAY_NAME_MAPPINGS = {"aigate_img2img": "全能图片-图生图"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
