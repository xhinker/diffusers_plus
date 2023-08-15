import os
import torch
import cv2
import numpy as np
import requests
from SwinIR.models.network_swinir import SwinIR as net

import logging
logger = logging.getLogger(__name__)


class SwinIR_AZ:
    def __init__(
        self
        ,model_path
        ,scale = 4
        ,training_patch_size = 128
        ,large_model = False
        ,output_path = '.'
        ,device = "cuda:0"
    ) -> None:
        # prepare and load
        self.model_path             = model_path
        self.device                 = device
        self.scale                  = scale
        self.training_patch_size    = training_patch_size
        self.large_model            = large_model
        self.output_path            = output_path
        
    def define_model(
        self
        ,task                   = 'real_sr'
        ,scale                  = 4
        ,training_patch_size    = 128
        ,large_model            = False
    ):
        '''
        Args: 
            model_path: (path to the pth model file)
            task: (classical_sr, lightweight_sr, real_sr:default, gray_dn, color_dn, jpeg_car, color_jpeg_car)
            scale: (scale factor: 1, 2, 3, 4:default, 8) 
            training_patch_size: (patch size used in training SwinIR, default to 128)
        '''
        # 003 real-world image sr
        if task == 'real_sr':
            if not large_model:
                # use 'nearest+conv' to avoid block artifacts
                model = net(upscale=scale, in_chans=3, img_size=64, window_size=8,
                            img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                            mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
            else:
                # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
                model = net(upscale=scale, in_chans=3, img_size=64, window_size=8,
                            img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                            num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                            mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
            param_key_g = 'params_ema'

        pretrained_model = torch.load(self.model_path)
        model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

        return model

    def get_image_pair(
        self
        ,img_path
        ,task      = 'real_sr'
    ):
        (imgname, imgext) = os.path.splitext(os.path.basename(img_path))
        # 003 real-world image sr (load lq image only)
        if task in ['real_sr']:
            img_gt = None
            img_lq = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

        return imgname, img_lq, img_gt

    def load_pth_model(self):
        '''
        Download and load models from https://github.com/JingyunLiang/SwinIR/releases
        '''
        # setup model
        if os.path.exists(self.model_path):
            logger.info(f'loading model from {self.model_path}')
        else:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(self.model_path))
            r = requests.get(url, allow_redirects=True)
            logger.info(f'downloading model {self.model_path}')
            open(self.model_path, 'wb').write(r.content)

        model = self.define_model(
            scale                = self.scale
            ,training_patch_size = self.training_patch_size
            ,large_model         = self.large_model
        )
        model.eval()
        model = model.to(self.device)
        return model

    def upscale_img(self,img_path,task = 'real_sr'):
        # read image
        imgname, img_lq, img_gt = self.get_image_pair(img_path=img_path)  # image to HWC-BGR, float32
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)  # CHW-RGB to NCHW-RGB

        model = self.load_pth_model()

        window_size = 8
        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = model(img_lq)
            output = output[..., :h_old * self.scale, :w_old * self.scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

        self.output_path = os.path.dirname(img_path)
        cv2.imwrite(f'{self.output_path}/{imgname}_SwinIR.png', output)