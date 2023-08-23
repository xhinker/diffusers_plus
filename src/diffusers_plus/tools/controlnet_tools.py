import torch
from ..models.annotator.openpose import OpenposeDetector
from diffusers.utils import load_image
import numpy as np
from PIL import Image
from diffusers.utils.pil_utils import pt_to_pil,numpy_to_pil
from transformers import CLIPSegProcessor,CLIPSegForImageSegmentation

openpose = OpenposeDetector()

processor = CLIPSegProcessor.from_pretrained(
    "CIDAS/clipseg-rd64-refined"
)
model = CLIPSegForImageSegmentation.from_pretrained(
    "CIDAS/clipseg-rd64-refined"
)

def get_pose_img(
    image
    , detect_hand = True
    , detect_face = True
):
    source_img = load_image(image)
    source_img_array = np.array(source_img)
    pose = openpose(
        source_img_array
        , hand = detect_hand
        , face = detect_face
    )
    return Image.fromarray(pose)

def get_mask_pil(
    target_prompt
    , target_image
    , bw_thresh = 100
):
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
    
