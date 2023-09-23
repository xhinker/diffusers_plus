from typing import List, Callable, Dict, Any
import torch
from torch import Generator,FloatTensor
from diffusers import (
    StableDiffusionPipeline
    , EulerDiscreteScheduler
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from ..tools.prompt_parser import parse_scheduled_prompts
from ..tools.sd_embeddings import get_weighted_text_embeddings_v15

class StableDiffusionPipeline_EXT(StableDiffusionPipeline):
    @torch.no_grad()
    def super_call(
        self
        , prompt: str = None
        , height: int | None = 512
        , width: int | None = 512
        , num_inference_steps: int = 50
        , guidance_scale: float = 7.5
        , negative_prompt: str | List[str] | None = None
        , eta: float = 0
        , generator: Generator | List[Generator] | None = None
        , latents: FloatTensor | None = None
        , prompt_embeds: FloatTensor | None = None
        , output_type: str | None = "pil"
        , callback: Callable[[int, int, FloatTensor], None] | None = None
        , callback_steps: int = 1
        , cross_attention_kwargs: Dict[str, Any] | None = None
    ):
        '''
        This function use the enhance text encoder support weighted prompt and unlimited length prompt.
        Also support scheduled prompt.  
        
        This function support one image generation at a time. 
        '''
        
        # set default scheduler to EulerDiscreteScheduler
        if self.scheduler._class_name == "PNDMScheduler":
            self.scheduler = EulerDiscreteScheduler.from_config(
                self.scheduler.config
            )
        num_images_per_prompt = 1
            
        # 1. prepare
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # 2. get the prompt list by parsing 
        prompt_list = parse_scheduled_prompts(prompt, steps = num_inference_steps)
        
        # 3. If no scheduled prompt is found, use the normal single prompt logic
        embedding_list = []
        if len(prompt_list) == 1:
            prompt_embed, neg_prompt_embed = get_weighted_text_embeddings_v15(
                pipe = self
                , prompt = prompt
                , neg_prompt = negative_prompt
            )
            prompt_embeds = self._encode_prompt(
                prompt                          = prompt 
                , negative_prompt               = negative_prompt
                , device                        = self.device
                , num_images_per_prompt         = num_images_per_prompt
                , do_classifier_free_guidance   = do_classifier_free_guidance
                , prompt_embeds                 = prompt_embed
                , negative_prompt_embeds        = neg_prompt_embed
            )
        else:
            for prompt in prompt_list:
                prompt_embed, neg_prompt_embed = get_weighted_text_embeddings_v15(
                    pipe = self
                    , prompt = prompt
                    , neg_prompt = negative_prompt
                )
                prompt_embeds = self._encode_prompt(
                    prompt                          = prompt 
                    , negative_prompt               = negative_prompt
                    , device                        = self.device
                    , num_images_per_prompt         = num_images_per_prompt
                    , do_classifier_free_guidance   = do_classifier_free_guidance
                    , prompt_embeds                 = prompt_embed
                    , negative_prompt_embeds        = neg_prompt_embed
                )
                embedding_list.append(prompt_embeds)
        
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
            
        latents = self.prepare_latents(
            num_images_per_prompt
            , num_channels_latents
            , height
            , width
            , prompt_embeds.dtype
            , device
            , generator
            , latents
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # AZ code to enable Prompt Scheduling, will only function when 
                # when there is a prompt_embeds_l provided. 
                prompt_embeds_l_len = len(embedding_list)
                if prompt_embeds_l_len > 0:
                    # ensure no None prompt will be used
                    pe_index = (i)%prompt_embeds_l_len                      
                    prompt_embeds = embedding_list[pe_index]

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                        
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        
        # Offload all models
        self.maybe_free_model_hooks()

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)
    
    @torch.no_grad()
    def scheduler_call(
        self
        , prompt: str | List[str] = None
        , height: int | None = 512
        , width: int | None = 512
        , num_inference_steps: int = 50
        , guidance_scale: float = 7.5
        , negative_prompt: str | List[str] | None = None
        , eta: float = 0
        , generator: Generator | List[Generator] | None = None
        , latents: FloatTensor | None = None
        , prompt_embeds: FloatTensor | None = None
        , output_type: str | None = "pil"
        , callback: Callable[[int, int, FloatTensor], None] | None = None
        , callback_steps: int = 1
        , cross_attention_kwargs: Dict[str, Any] | None = None
    ):
        '''
        This function use the default text encoder from the Diffusers pipeline. 
        This function support one image generation at a time only. 
        '''
        # set default scheduler to EulerDiscreteScheduler
        if self.scheduler._class_name == "PNDMScheduler":
            self.scheduler = EulerDiscreteScheduler.from_config(
                self.scheduler.config
            )
        num_images_per_prompt = 1
            
        # 1. prepare
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # 2. get the prompt list by parsing 
        prompt_list = parse_scheduled_prompts(prompt, steps = num_inference_steps)
        
        # 3. If no scheduled prompt is found, use the normal single prompt logic
        embedding_list = []
        if len(prompt_list) == 1:
            prompt_embeds = self._encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt
            )
        else:
            for prompt in prompt_list:
                prompt_embeds = self._encode_prompt(
                    prompt,
                    device,
                    num_images_per_prompt,
                    do_classifier_free_guidance,
                    negative_prompt
                )
                embedding_list.append(prompt_embeds)
        
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
            
        latents = self.prepare_latents(
            num_images_per_prompt
            , num_channels_latents
            , height
            , width
            , prompt_embeds.dtype
            , device
            , generator
            , latents
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # AZ code to enable Prompt Scheduling, will only function when 
                # when there is a prompt_embeds_l provided. 
                prompt_embeds_l_len = len(embedding_list)
                if prompt_embeds_l_len > 0:
                    # ensure no None prompt will be used
                    pe_index = (i)%prompt_embeds_l_len                      
                    prompt_embeds = embedding_list[pe_index]

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                        
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        
        # Offload all models
        self.maybe_free_model_hooks()

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)