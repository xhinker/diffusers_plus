import os
import torch
from ..models.swin2sr.models.network_swin2sr import Swin2SR as net
from ..models.swin2sr.utils import util_calculate_psnr_ssim as util
from .sdxl_pipeline_call import (
    load_sdxl_pipe_from_file
    , sdxl_img2img
    , load_sdxl_img2img_pipe_from_file
    , load_sdxl_peft_lora
)
from .sd15_pipeline_call import (
    load_sd15_tile_cn_pipe_from_file
    , sd15_controlnet
)
from PIL import Image
from ..models.clip_interrogator_az.clip_interrogator_az import Config, Interrogator
from diffusers.utils import load_image
from ..tools.sd_embeddings import (
    get_weighted_text_embeddings_v15
    , get_weighted_text_embeddings_sdxl
)
from diffusers import (
    ControlNetModel
    , StableDiffusionControlNetPipeline
)
from ..tools.image_upscaler_helper import (
    resize_img
)

from ..tools.common_tools import FileOps
import random

from diffusers import (
    EulerDiscreteScheduler
    , EulerAncestralDiscreteScheduler
)

import regex as re

import logging
logging.basicConfig(format='[%(filename)s - Code line: %(lineno)d - %(levelname)s]: %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class AZ_SD_SR:
    '''
    1. Use clip-interrogator to extract prompt from an image
    2. Or use SD1.5 Tile control net to upscale the image. 

    CN tile is good at add detail for low quality image under 1024x1024, 
    when image size is too larger, the tile can only cover a small part of the image,
    hence lead to weird image. 

    Use SDXL img2img for even higher upscale.

    ci 1 require transformers==4.26.1
    ci 2 require
    '''
    def __init__(
            self
            , clip_model_id = "ViT-L-14/openai"
            , caption_model_name = "blip2-flan-t5-xl"
            , sd15_base_model_file_path = ""
            , sd15_lora_list = []
            , sdxl_base_model_file_path = ""
            , sdxl_lora_list = []
        ) -> None:
        # load clip-interrogator 
        # pip install git+https://github.com/openai/CLIP.git
        # pip install -U open-clip-torch
        self.config_obj = Config(
            clip_model_path = clip_model_id
            , caption_model_name = caption_model_name
            , caption_max_length = 128
        )  # or ViT-H-14/laion2b_s32b_b79k
        # set ci handler
        self.ci = None
        # set cn_pipe handler
        self.sd15_cn_pipe = None
        # set sd15 model file path 
        self.sd15_base_model_file_path = sd15_base_model_file_path
        self.sd15_lora_list = sd15_lora_list
        self.sd15_model_name,_ = os.path.splitext(os.path.basename(sd15_base_model_file_path))

        # set sdxl img2img pipe
        self.sdxl_img2img_pipe = None
        # set sdxl model file path 
        self.sdxl_base_model_file_path = sdxl_base_model_file_path
        self.sdxl_lora_list = sdxl_lora_list
        self.sdxl_model_name,_ = os.path.splitext(os.path.basename(sdxl_base_model_file_path))

    def __clear_cuda_cache(self):
        self.ci.caption_model.to("cpu")
        self.ci.clip_model.to("cpu")
        torch.cuda.empty_cache()

    def __load_model_to_cuda(self):
        self.ci.caption_model.to("cuda:0")
        self.ci.clip_model.to("cuda:0")

    def __remove_emojis(self, input_text:str)->str:
        emoj = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002500-\U00002BEF"  # chinese char
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
                        "]+", re.UNICODE)
        return re.sub(emoj, '', input_text)

    def gen_prompt(self, image:Image) -> str:
        if self.ci is None:
            self.ci = Interrogator(
                self.config_obj
            )
            #self.__clear_cuda_cache()

        self.__load_model_to_cuda()
        prompt = self.ci.interrogate(image)

        prompt_wo_emojis = self.__remove_emojis(prompt)

        neg_prompt = self.ci.interrogate_negative(image)

        self.__clear_cuda_cache()
        return prompt_wo_emojis, neg_prompt

    def sd15_tile_upscale(
        self
        , image_path:str
        , upscale:float = 2.0
        , user_prompt:str = ""
        , user_neg_prompt:str = ""
        , steps = 40
        , cfg = 7
        , strength = 0.75
        , controlnet_weight = 0.75
        , save_image:bool = False
    ):
        '''
        Usage sample:
            from diffusers_plus.pipelines.image_upscaler_pipeline import AZ_SD15_SR
            azsdsr = AZ_SD15_SR(
                sd_base_model_file_path = "/path/to/model.safetensors"
            )
            image_path = "<path>"
            r = azsdsr.sd15_tile_upscale(
                image_path          = image_path
                , user_prompt       = ""
                , user_neg_prompt   = ""
                , steps             = 40
                , upscale           = 4
                , strength          = 0.85
                , controlnet_weight = 0.75
                , cfg               = 7.5
            )
            r
        '''
        if self.sd15_cn_pipe is None:
            self.sd15_cn_pipe = load_sd15_tile_cn_pipe_from_file(
                model_path = self.sd15_base_model_file_path
            )

        # resize input image
        resized_img = resize_img(img_path = image_path, upscale_times = upscale)

        # get prompt
        gen_prompt, gen_neg_prompt = self.gen_prompt(image = resized_img)

        pre_defined_prompt = "masterpiece, ultra-detailed, high resolution, best quality,8k, sharp focus, raw photo, flawless, perfect"
        prompt = f"{pre_defined_prompt},{user_prompt},{gen_prompt}"

        pre_defined_neg_prompt = "mosaic,noise spot,blur,low quality,mutated,low res,low resolution,ugly,bad taste, pexels,bad anatomy"
        neg_prompt = f"{user_neg_prompt}, {pre_defined_neg_prompt}, {gen_neg_prompt}"

        print(f"prompt:{prompt}")
        print(f"neg_prompt:{neg_prompt}")

        # run pipe
        r = sd15_controlnet(
            pipe                            = self.sd15_cn_pipe
            , original_image                = resized_img
            , control_image                 = resized_img
            , prompt                        = prompt
            , neg_prompt                    = neg_prompt
            , steps                         = steps
            , cfg                           = cfg
            , strength                      = strength
            , controlnet_conditioning_scale = controlnet_weight
        )

        # save image
        image_save_path = ""
        if save_image:
            sd_meta = {
                "prompt":prompt
                , "neg_prompt":neg_prompt
                , "steps": steps
                , "cfg": cfg
                , "strength": strength
                , "controlnet_conditioning_scale": controlnet_weight
                , "upscale": upscale
            }
            new_image_path = FileOps.get_new_image_path(
                image_path
                , additional_info=[
                    "sd15_tile"
                    , "steps_"+str(sd_meta["steps"])
                    , "cfg_"+str(sd_meta["cfg"])
                    , "strength_"+str(sd_meta["strength"])
                    , "cn_"+str(sd_meta["controlnet_conditioning_scale"])
                    , "upscale_"+str(sd_meta["upscale"])
                ]
            )
            image_save_path = FileOps.save_image_with_info(
                image = r
                , image_save_path = new_image_path
                , json_obj_dict = {"sd_meta": sd_meta}
            )

        return r, image_save_path
    
    def sdxl_img_upscale(
        self
        , image_path:str
        , upscale:float = 2.0
        , user_prompt:str = ""
        , user_neg_prompt:str = ""
        , steps = 40
        , cfg = 4
        , strength = 0.4
        , save_image:bool = False
        , seed:int = None
        , scheduler = EulerAncestralDiscreteScheduler
        , lora_weights = ()             # a tuple (['name1','name2'],[0.5,0.6])
    ):
        # load sdxl_img2img_pipe is not exists
        if self.sdxl_img2img_pipe is None:
            self.sdxl_img2img_pipe = load_sdxl_img2img_pipe_from_file(
                model_path = self.sdxl_base_model_file_path
            )
            load_sdxl_peft_lora(
                pipe = self.sdxl_img2img_pipe
                , lora_info_list = self.sdxl_lora_list
            )
            logger.info("load up SDXL image2image pipe and LoRAs")
            
        
        # resize input image
        #resized_img = resize_img(img_path = image_path, upscale_times = upscale)
        image = load_image(image_path).convert("RGB")

        # get prompt
        gen_prompt, gen_neg_prompt = self.gen_prompt(image = image)
        
        pre_defined_prompt = "masterpiece, ultra-detailed, high resolution, best quality,8k, sharp focus, raw photo, flawless, perfect"
        prompt = f"{pre_defined_prompt},{user_prompt},{gen_prompt}"

        pre_defined_neg_prompt = "mosaic,noise spot,blur,low quality,mutated,low res,low resolution,ugly,bad taste, pexels,bad anatomy"
        neg_prompt = f"{user_neg_prompt}, {pre_defined_neg_prompt}, {gen_neg_prompt}"

        print(f"prompt:{prompt}")
        print(f"neg_prompt:{neg_prompt}")

        if seed is None:
            seed = random.randint(0,10000000000000)

        # set lora
        if len(lora_weights)>0:
            self.sdxl_img2img_pipe.set_adapters(
                lora_weights[0]
                , adapter_weights = lora_weights[1]
            )

        lora_scale = {
            "scale":1.0
        }
        r = sdxl_img2img(
            pipe                     = self.sdxl_img2img_pipe
            , input_image            = image
            , prompt                 = prompt
            , neg_prompt             = neg_prompt
            , resize_times           = upscale
            , seed                   = seed
            , steps                  = steps
            , cfg                    = cfg
            , strength               = strength
            , scheduler              = scheduler
            , cross_attention_kwargs = lora_scale
        )

        # save image
        image_save_path = ""
        if save_image:
            sd_meta = {
                "model_name": self.sdxl_model_name
                ,"prompt":prompt
                , "neg_prompt":neg_prompt
                , "steps": steps
                , "cfg": cfg
                , "strength": strength
                , "upscale": upscale
                , "seed":seed
                , "scheduler": scheduler.__name__
            }
            new_image_path = FileOps.get_new_image_path(
                image_path
                , additional_info=[
                    "sdxl_img2img"
                    , "steps_"+str(sd_meta["steps"])
                    , "cfg_"+str(sd_meta["cfg"])
                    , "strength_"+str(sd_meta["strength"])
                    , "upscale_"+str(sd_meta["upscale"])
                ]
            )
            image_save_path = FileOps.save_image_with_info(
                image = r
                , image_save_path = new_image_path
                , json_obj_dict = {"sd_meta": sd_meta}
            )
        
        return r, image_save_path
    
    def test_func(self, msg):
        print(msg)




class Swin2sr_AZ:
    '''
    Not working well, 
    '''
    def __init__(
        self
        , model_path:str    = None
        , task:str          = "real_sr"     # classical_sr, lightweight_sr, real_sr, gray_dn, color_dn, jpeg_car, color_jpeg_car
        , upscale:int       = 2             # 'scale factor: 1, 2, 3, 4, 8'
        , tile:int          = None          # int Tile size, None for no tile during testing, "tile size should be a multiple of window_size 
        , tile_overlap:int  = 32            # Overlapping of different tiles
        , large_model:bool  = False         # for realsr only
        , device:str        = "cuda:0"      # execution cuda device
    ) -> None:
        '''
        Load model and prepare parameters
        '''
        self.model_path     = model_path
        self.task           = task
        self.scale          = upscale
        self.large_model    = large_model
        self.jpeg           = 40                            # scale factor: 10, 20, 30, 40
        self.device         = device
        self.tile           = tile
        self.tile_overlap   = tile_overlap

        if not os.path.exists(model_path):
            try:
                self.__download_model_file()
            except Exception as e:
                logger.error(f"download model file error, please double check the model path and model name | {e}")

        self.model = self.__define_model()
        self.model.eval()               # set training to false
        self.model = self.model.to(device)

        logger.info("swin2sr model is initialized")
        
    def __define_model(self):
        # 001 classical image sr
        if self.task == 'classical_sr':
            model = net(upscale=self.scale, in_chans=3, img_size=128, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
            param_key_g = 'params'

        # 002 lightweight image sr
        # use 'pixelshuffledirect' to save parameters
        elif self.task in ['lightweight_sr']:
            model = net(upscale=self.scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
            param_key_g = 'params'
            
        elif self.task == 'compressed_sr':
            model = net(upscale=self.scale, in_chans=3, img_size=128, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffle_aux', resi_connection='1conv')
            param_key_g = 'params'                

        # 003 real-world image sr
        elif self.task == 'real_sr':
            if not self.large_model:
                # use 'nearest+conv' to avoid block artifacts
                model = net(upscale=self.scale, in_chans=3, img_size=64, window_size=8,
                            img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                            mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
            else:
                # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
                model = net(upscale=self.scale, in_chans=3, img_size=64, window_size=8,
                            img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                            num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                            mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
            param_key_g = 'params_ema'

        # 006 grayscale JPEG compression artifact reduction
        # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
        elif self.task == 'jpeg_car':
            model = net(upscale=1, in_chans=1, img_size=126, window_size=7,
                        img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='', resi_connection='1conv')
            param_key_g = 'params'

        # 006 color JPEG compression artifact reduction
        # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
        elif self.task == 'color_jpeg_car':
            model = net(upscale=1, in_chans=3, img_size=126, window_size=7,
                        img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='', resi_connection='1conv')
            param_key_g = 'params'

        pretrained_model = torch.load(self.model_path)
        model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

        return model

    def __download_model_file(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        url = f'https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/{os.path.basename(self.model_path)}'
        r = requests.get(url, allow_redirects=True)
        logger.info(f"downloading model {self.model_path} ...")
        with open(self.model_path, 'wb') as f:
            f.write(r.content)
        logger.inf0(f"download model {self.model_path} done.")

    def __get_image_pair(self, path):
        (imgname, imgext) = os.path.splitext(os.path.basename(path))

        # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs)
        if self.task in ['classical_sr', 'lightweight_sr']:
            img_gt = None
            img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.            
              
        elif self.task in ['compressed_sr']:
            img_gt = None
            img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.            
     
        # 003 real-world image sr (load lq image only)
        elif self.task in ['real_sr', 'lightweight_sr_infer']:
            img_gt = None
            img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

        # 006 grayscale JPEG compression artifact reduction (load gt image and generate lq image on-the-fly)
        elif self.task in ['jpeg_car']:
            img_gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img_gt.ndim != 2:
                img_gt = util.bgr2ycbcr(img_gt, y_only=True)
            result, encimg = cv2.imencode('.jpg', img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg])
            img_lq = cv2.imdecode(encimg, 0)
            img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.
            img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.

        # 006 JPEG compression artifact reduction (load gt image and generate lq image on-the-fly)
        elif self.task in ['color_jpeg_car']:
            img_gt = cv2.imread(path)
            result, encimg = cv2.imencode('.jpg', img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg])
            img_lq = cv2.imdecode(encimg, 1)
            img_gt = img_gt.astype(np.float32)/ 255.
            img_lq = img_lq.astype(np.float32)/ 255.

        return imgname, img_lq, img_gt

    def __read_image(self, image_path:str):
        imgname, img_lq, img_gt = self.__get_image_pair(path = image_path)  # image to HWC-BGR, float32
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)  # CHW-RGB to NCHW-RGB
        return img_lq, imgname

    def __setup(self, image_path:str = None):
        if not image_path:
            logger.error(f"no image_path is provided")
            return False
        
        image_dir = os.path.dirname(image_path)
        #os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # 001 classical image sr/ 002 lightweight image sr
        if self.task in ['classical_sr', 'lightweight_sr', 'compressed_sr']:
            save_dir = f'{image_dir}/swin2sr_{self.task}_x{self.scale}'
            border = self.scale
            window_size = 8

        # 003 real-world image sr
        elif self.task in ['real_sr']:
            save_dir = f'{image_dir}/swin2sr_{self.task}_x{self.scale}'
            if self.large_model:
                save_dir += '_large'
            border = 0
            window_size = 8

        # 006 JPEG compression artifact reduction
        elif self.task in ['jpeg_car', 'color_jpeg_car']:
            save_dir = f'{image_dir}/swin2sr_{self.task}_x{self.scale}'
            border = 0
            window_size = 7

        return "", save_dir, border, window_size

    def __tile_inference(self, img_lq, model, window_size):
        if self.tile is None:
            # test the image as a whole
            with torch.no_grad():
                output = model(img_lq)
        else:
            # test the image tile by tile
            b, c, h, w = img_lq.size()
            tile = min(self.tile, h, w)
            assert tile % window_size == 0, "tile size should be a multiple of window_size"
            tile_overlap = self.tile_overlap
            sf = self.scale

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
            w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
            E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
            W = torch.zeros_like(E)

            with torch.no_grad():
                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                        out_patch = model(in_patch)
                        out_patch_mask = torch.ones_like(out_patch)

                        E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                        W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
                output = E.div_(W)

        return output

    def upscale_image(self, input_image_path:str):
        '''
        Upscale image using real_sr upscaler
        '''
        _, save_dir, border, window_size = self.__setup(image_path=input_image_path)
        os.makedirs(save_dir, exist_ok=True)

        # get image and image name
        img_lq, imgname = self.__read_image(image_path=input_image_path)
        
        # prepare data
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]

        # inference
        output = self.__tile_inference(img_lq = img_lq, model = self.model,window_size=window_size)
        if self.task == 'compressed_sr':
            output = output[0][..., :h_old * self.scale, :w_old * self.scale]
        else:
            output = output[..., :h_old * self.scale, :w_old * self.scale]

        # save output image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR

        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        cv2.imwrite(f'{save_dir}/{imgname}_Swin2SR.png', output)

        logger.info("upscale done")
        

# class SwinIR_AZ:
#     def __init__(
#         self
#         ,model_path
#         ,scale = 4
#         ,training_patch_size = 128
#         ,large_model = False
#         ,output_path = '.'
#         ,device = "cuda:0"
#     ) -> None:
#         # prepare and load
#         self.model_path             = model_path
#         self.device                 = device
#         self.scale                  = scale
#         self.training_patch_size    = training_patch_size
#         self.large_model            = large_model
#         self.output_path            = output_path
        
#     def define_model(
#         self
#         ,task                   = 'real_sr'
#         ,scale                  = 4
#         ,training_patch_size    = 128
#         ,large_model            = False
#     ):
#         '''
#         Args: 
#             model_path: (path to the pth model file)
#             task: (classical_sr, lightweight_sr, real_sr:default, gray_dn, color_dn, jpeg_car, color_jpeg_car)
#             scale: (scale factor: 1, 2, 3, 4:default, 8) 
#             training_patch_size: (patch size used in training SwinIR, default to 128)
#         '''
#         # 003 real-world image sr
#         if task == 'real_sr':
#             if not large_model:
#                 # use 'nearest+conv' to avoid block artifacts
#                 model = net(upscale=scale, in_chans=3, img_size=64, window_size=8,
#                             img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
#                             mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
#             else:
#                 # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
#                 model = net(upscale=scale, in_chans=3, img_size=64, window_size=8,
#                             img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
#                             num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
#                             mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
#             param_key_g = 'params_ema'

#         pretrained_model = torch.load(self.model_path)
#         model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

#         return model

#     def get_image_pair(
#         self
#         ,img_path
#         ,task      = 'real_sr'
#     ):
#         (imgname, imgext) = os.path.splitext(os.path.basename(img_path))
#         # 003 real-world image sr (load lq image only)
#         if task in ['real_sr']:
#             img_gt = None
#             img_lq = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

#         return imgname, img_lq, img_gt

#     def load_pth_model(self):
#         '''
#         Download and load models from https://github.com/JingyunLiang/SwinIR/releases
#         '''
#         # setup model
#         if os.path.exists(self.model_path):
#             logger.info(f'loading model from {self.model_path}')
#         else:
#             os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
#             url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(self.model_path))
#             r = requests.get(url, allow_redirects=True)
#             logger.info(f'downloading model {self.model_path}')
#             open(self.model_path, 'wb').write(r.content)

#         model = self.define_model(
#             scale                = self.scale
#             ,training_patch_size = self.training_patch_size
#             ,large_model         = self.large_model
#         )
#         model.eval()
#         model = model.to(self.device)
#         return model

#     def upscale_img(self,img_path,task = 'real_sr'):
#         # read image
#         imgname, img_lq, img_gt = self.get_image_pair(img_path=img_path)  # image to HWC-BGR, float32
#         img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
#         img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)  # CHW-RGB to NCHW-RGB

#         model = self.load_pth_model()

#         window_size = 8
#         # inference
#         with torch.no_grad():
#             # pad input image to be a multiple of window_size
#             _, _, h_old, w_old = img_lq.size()
#             h_pad = (h_old // window_size + 1) * window_size - h_old
#             w_pad = (w_old // window_size + 1) * window_size - w_old
#             img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
#             img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
#             output = model(img_lq)
#             output = output[..., :h_old * self.scale, :w_old * self.scale]

#         # save image
#         output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
#         if output.ndim == 3:
#             output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
#         output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

#         self.output_path = os.path.dirname(img_path)
#         cv2.imwrite(f'{self.output_path}/{imgname}_SwinIR.png', output)