import torch
from diffusers import (
    StableDiffusionPipeline
    , ControlNetModel
    , StableDiffusionControlNetImg2ImgPipeline 
    , EulerDiscreteScheduler
)
from ..tools.sd_embeddings import get_weighted_text_embeddings_v15

def load_sd15_tile_cn_pipe_from_file(model_path:str):
    controlnet = ControlNetModel.from_pretrained(
        'takuma104/control_v11'
        , subfolder='control_v11f1e_sd15_tile'
        , torch_dtype=torch.float16
    )
    
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
        model_path
        , torch_dtype           = torch.float16
        , load_safety_checker   = False
        , controlnet            = controlnet
    )
    
    return pipe


def sd15_controlnet(
    pipe:StableDiffusionControlNetImg2ImgPipeline
    , original_image
    , control_image
    , prompt:str
    , neg_prompt:str
    , seed = 1
    , steps = 20
    , cfg = 7
    , strength = 0.5
    , scheduler = EulerDiscreteScheduler
    , controlnet_conditioning_scale = 0.5
):
    # if type(control_images) is not list:
    #     control_images = [control_images]
        
    pipe.to("cuda")
    (
        prompt_embeds
        , prompt_neg_embeds
    ) = get_weighted_text_embeddings_v15(
        pipe
        , prompt = prompt
        , neg_prompt = neg_prompt
    )
    
    pipe.enable_vae_tiling()
    pipe.scheduler = scheduler.from_config(pipe.scheduler.config)
    image = pipe(
        prompt_embeds                   = prompt_embeds 
        , negative_prompt_embeds        = prompt_neg_embeds 
        , image                         = original_image
        , control_image                 = control_image
        , guidance_scale                = cfg
        , num_inference_steps           = steps
        , strength                      = strength
        , generator                     = torch.Generator("cuda").manual_seed(seed)
        , controlnet_conditioning_scale = controlnet_conditioning_scale
    ).images[0]
    
    pipe.to("cpu")
    torch.cuda.empty_cache()
    return image
        
