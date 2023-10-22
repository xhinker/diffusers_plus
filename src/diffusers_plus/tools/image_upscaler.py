# not working well enough
import logging
logger = logging.getLogger(__name__)

from diffusers.utils import load_image
from PIL import Image

def get_width_height(width, height):
    width = (width//8)*8
    height = (height//8)*8
    return width,height 

def resize_img(img_path,upscale_times):
    img             = load_image(img_path)
    if upscale_times <=0:
        return img
    width,height    = img.size
    width           = width * upscale_times
    height          = height * upscale_times
    width,height    = get_width_height(int(width),int(height))
    img             = img.resize(
        (width,height)
        ,resample = Image.LANCZOS if upscale_times > 1 else Image.AREA
    )
    return img
