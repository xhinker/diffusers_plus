from diffusers import (
    StableDiffusionXLPipeline
)
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

def convert_sdxl_safetensors_to_diffusers(
    source_file_path:str
    , target_directory_path:str
):
    pipe = download_from_original_stable_diffusion_ckpt(
        checkpoint_path_or_dict     = source_file_path
        , from_safetensors          = True
        , device                    = "cuda:0"
        , model_type                = "SDXL"
        , pipeline_class            = StableDiffusionXLPipeline
    )
    pipe.save_pretrained(target_directory_path)