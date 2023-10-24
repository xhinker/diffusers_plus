import os
from PIL import PngImagePlugin
from PIL import Image
from diffusers.utils import load_image
import json

class FileOps:
    @classmethod
    def get_new_image_path(cls, image_path:str, additional_info:list = [])->str:
        target_image_folder = os.path.dirname(image_path)
        image_file_name_ext = os.path.basename(image_path)
        base_name, ext      = os.path.splitext(image_file_name_ext)
        additional_info_str = "_".join(additional_info)
        target_image_name   = f"{base_name}_{additional_info_str}.png"
        target_image_path   = os.path.join(target_image_folder,target_image_name) 
        return target_image_path


    @classmethod
    def save_image_with_info(cls,image:Image, image_save_path:str,json_obj_dict:dict) -> str:
        metadata = PngImagePlugin.PngInfo()

        for obj_key in json_obj_dict:
            meta_str = json.dumps(json_obj_dict[obj_key])
            metadata.add_text("obj_key",meta_str)
        
        image.save(image_save_path, "PNG", pnginfo=metadata)

        return image_save_path
    
    @classmethod
    def read_info_from_image(cls,image_path:str) -> dict:
        image = Image.open(image_path)
        metadata = image.info

        text_info_dict = {}
        for key,value in metadata.items():
            text_info_dict[key] = value

        return text_info_dict
