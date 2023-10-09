import torch
from diffusers import (
    StableDiffusionXLPipeline
    , StableDiffusionXLImg2ImgPipeline
)
from diffusers import (
    EulerDiscreteScheduler
    , EulerAncestralDiscreteScheduler
    , DPMSolverMultistepScheduler
    , ControlNetModel
    , StableDiffusionXLControlNetPipeline
    , StableDiffusionXLInpaintPipeline
)
from ..tools.sd_embeddings import get_weighted_text_embeddings_sdxl
from ..tools.image_upscale import resize_img

def load_sdxl_pipe_from_file(model_path:str):
    '''
    Load sdxl text2img pipe from safetensor files directly
    '''
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path
        , torch_dtype           = torch.float16
        , use_safetensors       = True
        , load_safety_checker   = False
    )
    pipe.watermark = None
    return pipe

def load_sdxl_img2img_pipe_from_file(model_path:str):
    '''
    load sdxl img2img pipe from safetensor files directly
    '''
    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
        model_path
        , torch_dtype           = torch.float16
        , use_safetensors       = True
        , load_safety_checker   = False
    )
    pipe.watermark = None
    return pipe

def load_sdxl_inpaint_pipe_from_file(model_path:str = "stabilityai/stable-diffusion-xl-base-1.0"):
    sdxl_inpaint = StableDiffusionXLInpaintPipeline.from_single_file(
        model_path
        , torch_dtype           = torch.float16
        , variant               = "fp16"
        , use_safetensors       = True
        , load_safety_checker   = False
    )
    sdxl_inpaint.watermark = None
    return sdxl_inpaint

def load_sdxl_openpose_cn_pipe_from_pretrained(
        base_model_id = "thibaud/controlnet-openpose-sdxl-1.0"
        , model_id:str = "RunDiffusion/RunDiffusion-XL-Beta"
    ):
    sdxl_pose_controlnet = ControlNetModel.from_pretrained(
        base_model_id
        , torch_dtype=torch.float16
    )

    # load sdxl controlnet pipeline
    sdxl_cn_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        model_id
        , torch_dtype           = torch.float16
        , use_safetensors       = True
        , load_safety_checker   = False
        , add_watermarker       = False
        , controlnet            = sdxl_pose_controlnet
    )
    sdxl_cn_pipe.watermark = None
    return sdxl_cn_pipe

def sdxl_text2img(
    pipe
    , prompt:str
    , neg_prompt:str
    , seed = 1
    , steps = 40
    , cfg = 10
    , width = 832
    , height = 1216
    , scheduler = EulerDiscreteScheduler
):
    
    pipe.to("cuda")
    (
        prompt_embeds
        , prompt_neg_embeds
        , pooled_prompt_embeds
        , negative_pooled_prompt_embeds
    ) = get_weighted_text_embeddings_sdxl(
        pipe
        , prompt = prompt
        , neg_prompt = neg_prompt
    )
    
    pipe.scheduler = scheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler.config.algorithm_type = "sde-dpmsolver++"
    # pipe.scheduler.config.use_karras_sigmas = True
    
    raw_image = pipe(
        prompt_embeds                   = prompt_embeds 
        , negative_prompt_embeds        = prompt_neg_embeds 
        , pooled_prompt_embeds          = pooled_prompt_embeds
        , negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
        , width                         = width
        , height                        = height      
        , generator                     = torch.Generator("cuda").manual_seed(seed)
        , num_inference_steps           = steps
        , guidance_scale                = cfg
    ).images[0]

    pipe.to("cpu")
    torch.cuda.empty_cache()
    return raw_image

def sdxl_img2img(
    pipe
    , input_image
    , prompt:str
    , neg_prompt:str
    , resize_times = 2.0
    , seed = 1
    , steps = 40
    , cfg = 10
    , strength = 0.5
    , scheduler = EulerDiscreteScheduler
):
    pipe.to("cuda")
    
    if resize_times > 1.0:
        resized_img = resize_img(input_image,resize_times)
    else:
        resized_img = input_image
    
    (
        prompt_embeds
        , prompt_neg_embeds
        , pooled_prompt_embeds
        , negative_pooled_prompt_embeds
    ) = get_weighted_text_embeddings_sdxl(
        pipe
        , prompt = prompt
        , neg_prompt = neg_prompt
    )

    pipe.enable_vae_tiling()
    
    pipe.scheduler = scheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler.config.algorithm_type = "sde-dpmsolver++"
    # pipe.scheduler.config.use_karras_sigmas = True
    
    image = pipe(
        prompt_embeds                   = prompt_embeds 
        , negative_prompt_embeds        = prompt_neg_embeds 
        , pooled_prompt_embeds          = pooled_prompt_embeds
        , negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
        , image                         = [resized_img]
        , strength                      = strength
        , guidance_scale                = cfg
        , num_inference_steps           = steps
        , generator = torch.Generator("cuda").manual_seed(seed)
    ).images[0]

    pipe.to("cpu")
    torch.cuda.empty_cache()
    return image

def sdxl_inpaint(
    pipe:StableDiffusionXLInpaintPipeline
    , input_image
    , mask_image
    , prompt:str
    , neg_prompt:str
    , strength = 1.0
    , seed = 1
    , steps = 40
    , cfg = 7.5
    , scheduler = EulerDiscreteScheduler
):
    pipe.to("cuda")
    (
        prompt_embeds
        , prompt_neg_embeds
        , pooled_prompt_embeds
        , negative_pooled_prompt_embeds
    ) = get_weighted_text_embeddings_sdxl(
        pipe
        , prompt = prompt
        , neg_prompt = neg_prompt
    )
    
    pipe.scheduler = scheduler.from_config(pipe.scheduler.config)
    
    sdxl_inpaint_output = pipe(
        prompt_embeds                   = prompt_embeds 
        , negative_prompt_embeds        = prompt_neg_embeds 
        , pooled_prompt_embeds          = pooled_prompt_embeds
        , negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
        , image                         = input_image
        , mask_image                    = mask_image
        , num_inference_steps           = steps
        , guidance_scale                = cfg
        , strength                      = strength
        , width                         = input_image.size[0]
        , height                        = input_image.size[1]
        , generator                     = torch.Generator("cuda:0").manual_seed(seed) 
    ).images[0]
    pipe.to("cpu")
    torch.cuda.empty_cache()
    
    return sdxl_inpaint_output
    
def sdxl_controlnet(
    pipe
    , control_images
    , prompt:str
    , neg_prompt:str
    , seed = 1
    , steps = 40
    , cfg = 10
    , strength = 0.5
    , scheduler = EulerDiscreteScheduler
    , controlnet_conditioning_scale = 0.5
):
    if type(control_images) is not list:
        control_images = [control_images]
        
    pipe.to("cuda")
    (
        prompt_embeds
        , prompt_neg_embeds
        , pooled_prompt_embeds
        , negative_pooled_prompt_embeds
    ) = get_weighted_text_embeddings_sdxl(
        pipe
        , prompt = prompt
        , neg_prompt = neg_prompt
    )
    
    pipe.enable_vae_tiling()
    pipe.scheduler = scheduler.from_config(pipe.scheduler.config)
    image = pipe(
        prompt_embeds                   = prompt_embeds 
        , negative_prompt_embeds        = prompt_neg_embeds 
        , pooled_prompt_embeds          = pooled_prompt_embeds
        , negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
        , image                         = control_images
        , guidance_scale                = cfg
        , num_inference_steps           = steps
        , generator                     = torch.Generator("cuda").manual_seed(seed)
        #, latents                      = raw_image_us
        , controlnet_conditioning_scale = controlnet_conditioning_scale
    ).images[0]
    
    pipe.to("cpu")
    torch.cuda.empty_cache()
    return image

def sdxl_pipeline_loading_test():
    print("good2")