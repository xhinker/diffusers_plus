import torch
from ..models.annotator.openpose import OpenposeDetector
from diffusers.utils import load_image
import numpy as np
from PIL import Image, ImageOps
from diffusers.utils.pil_utils import pt_to_pil,numpy_to_pil
from transformers import CLIPSegProcessor,CLIPSegForImageSegmentation
from controlnet_aux import CannyDetector, PidiNetDetector, HEDdetector


def get_pose_img(
    image
    , detect_hand = True
    , detect_face = True
):
    openpose = OpenposeDetector()
    source_img = load_image(image)
    source_img_array = np.array(source_img)
    pose = openpose(
        source_img_array
        , hand = detect_hand
        , face = detect_face
    )
    return Image.fromarray(pose)

def get_soft_edge(
    image:Image
):
    processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
    #processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
    return processor(image, safe=True)

def get_canny(
    image:Image
    , low_threshold = 25
    , high_threshold = 100
) -> Image:
    '''
    Get image canny. 
    Usage sample:
        ```py
        import sys
        sys.path.append('../src/')
        
        from diffusers_plus.tools.controlnet_tools import get_canny
        from diffusers.utils import load_image

        img_path = "https://huggingface.co/datasets/ayushtues/blipdiffusion_images/resolve/main/flower.jpg"
        image = load_image(img_path)
        get_canny(image, low_threshold=25, high_threshold=100)
        ```
    '''
    #width, height = image.size
    canny = CannyDetector()
    return canny(
        input_image         = image
        , low_threshold     = low_threshold
        , high_threshold    = high_threshold
        , output_type       = "pil"
    )
    
def get_mask_pil(
    target_prompt
    , target_image
    , bw_thresh = 100
):
    processor = CLIPSegProcessor.from_pretrained(
        "CIDAS/clipseg-rd64-refined"
    )
    model = CLIPSegForImageSegmentation.from_pretrained(
        "CIDAS/clipseg-rd64-refined"
    )
        
    prompts = [target_prompt]
    inputs = processor(
        text             = prompts
        , images         = [target_image] * len(prompts)
        , padding        = True
        , return_tensors = "pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)

    preds = outputs.logits
    mask_data = torch.sigmoid(preds)
    
    mask_data_numpy = mask_data.detach().unsqueeze(-1).numpy()
    mask_pil = numpy_to_pil(mask_data_numpy)[0].resize(target_image.size)
    
    bw_fn = lambda x : 255 if x>bw_thresh else 0 
    bw_mask_pil = mask_pil.convert("L").point(bw_fn, mode="1")
    return mask_pil, bw_mask_pil

def get_masked_image(source_image:Image, bw_mask:Image):
    image_2 = Image.new("RGBA", source_image.size, (0,0,0,0))
    return Image.composite(source_image,image_2, bw_mask)

def remove_background(source_image:Image, threashold:int = 100):
    '''
    Remove the background base on the bw_mask
    '''
    prompt = "background"
    _, bw_mask = get_mask_pil(
        target_prompt   = prompt
        , target_image  = source_image
        , bw_thresh    = threashold
    )

    output_image = Image.new("RGBA", source_image.size, (255,255,255,255))
    inverse_bw_mask_pil = ImageOps.invert(bw_mask)
    r = Image.composite(source_image ,output_image, inverse_bw_mask_pil)
    return r

def remove_background_rembg(source_image:Image):
    from rembg import remove
    white_bg = Image.new("RGBA", source_image.size, (255,255,255))
    image_wo_bg = remove(source_image)
    output_image = Image.alpha_composite(white_bg, image_wo_bg)
    return output_image


def get_object_w_prompt(source_img:Image, prompt:str, bw_thresh = 100):
    # get mask
    _, bw_mask = get_mask_pil(
        target_prompt = prompt
        , target_image = source_img
        , bw_thresh = bw_thresh
    )
    
    # get image
    object_img = get_masked_image(
        source_image = source_img
        , bw_mask = bw_mask
    )
    
    return object_img


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
    
