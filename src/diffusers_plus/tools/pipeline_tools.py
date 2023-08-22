from safetensors.torch import load_file
import os
import torch
from diffusers import StableDiffusionPipeline

def load_lora_to_model(
    pipeline
    ,lora_path
    ,lora_weight_delta = 0.5
    ,device = 'cpu'
):
    '''
    For Stable Diffusion V1.5 based LoRA only, not working for SDXL
    lora_weight_delta = new_weight - current_weight
    '''
    state_dict = load_file(lora_path, device=device)
    LORA_PREFIX_UNET = 'lora_unet'
    LORA_PREFIX_TEXT_ENCODER = 'lora_te'

    alpha = lora_weight_delta
    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # as we have set the alpha beforehand, so just skip
        if '.alpha' in key or key in visited:
            continue
            
        if 'text' in key:
            layer_infos = key.split('.')[0].split(LORA_PREFIX_TEXT_ENCODER+'_')[-1].split('_')
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split('.')[0].split(LORA_PREFIX_UNET+'_')[-1].split('_')
            curr_layer = pipeline.unet

        # find the target layer
        # loop through the layers to find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                # no exception means the layer is found
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                # layer found but length is 0, 
                # break the loop and curr_layer keep point to the current layer
                elif len(layer_infos) == 0:
                    break
            except Exception:
                # no such layer exist, pop next name and try again
                if len(temp_name) > 0:
                    temp_name += '_'+layer_infos.pop(0)
                else:
                    # temp_name is empty
                    temp_name = layer_infos.pop(0)
        
        # org_forward(x) + lora_up(lora_down(x)) * multiplier
        # ensure the sequence of lora_up(A) then lora_down(B)
        pair_keys = []
        if 'lora_down' in key:
            pair_keys.append(key.replace('lora_down', 'lora_up'))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace('lora_up', 'lora_down'))
        
        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            # squeeze(3) and squeeze(2) remove dimensions of size 1 from the tensor to make the tensor more compact
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)
            
        # update visited list, ensure no duplicated weight is processed. 
        for item in pair_keys:
            visited.append(item)

    # record the loaded lora and weight in self.lora_dict
    # lora_file_name = os.path.basename(lora_path)
    # lora_name,ext = os.path.splitext(lora_file_name)

    # if lora_name in self.lora_dict:
    #     # if the lora is already loaded, the new weight need to be an sum 
    #     # of previous weight plus weight delta, the delta could be less than 0
    #     # e.g. the previous lora weight is 0.3, the new delta is 0.1. 
    #     # then, the new weight should be 0.3 + 0.1
    #     self.lora_dict[lora_name] = self.lora_dict[lora_name] + lora_weight_delta
    # else:
    #     # the lora is never been loaded, so the delta is the new weight. 
    #     self.lora_dict[lora_name] = lora_weight_delta

    # return pipeline
    
def load_lora_to_sdxl_model(
    pipeline
    ,lora_path
    ,lora_weight_delta = 0.5
    ,device = 'cpu'
):
    '''
    For Stable Diffusion V1.5 based LoRA only, not working for SDXL
    lora_weight_delta = new_weight - current_weight
    '''
    state_dict = load_file(lora_path, device=device)
    LORA_PREFIX_UNET = 'lora_unet'
    LORA_PREFIX_TEXT_ENCODER_1 = 'lora_te1'
    LORA_PREFIX_TEXT_ENCODER_2 = 'lora_te2'

    alpha = lora_weight_delta
    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # as we have set the alpha beforehand, so just skip
        if '.alpha' in key or key in visited:
            continue
            
        if 'te1_text' in key:
            layer_infos = key.split('.')[0].split(LORA_PREFIX_TEXT_ENCODER_1+'_')[-1].split('_')
            curr_layer = pipeline.text_encoder
        elif 'te2_text' in key: 
            layer_infos = key.split('.')[0].split(LORA_PREFIX_TEXT_ENCODER_2+'_')[-1].split('_')
            curr_layer = pipeline.text_encoder_2
        else:
            layer_infos = key.split('.')[0].split(LORA_PREFIX_UNET+'_')[-1].split('_')
            curr_layer = pipeline.unet

        # find the target layer
        # loop through the layers to find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                # no exception means the layer is found
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                # layer found but length is 0, 
                # break the loop and curr_layer keep point to the current layer
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(layer_infos) == 0:
                    curr_layer = None
                    break
                # no such layer exist, pop next name and try again
                if len(temp_name) > 0:
                    temp_name += '_'+layer_infos.pop(0)
                else:
                    # temp_name is empty
                    temp_name = layer_infos.pop(0)
        
        if curr_layer is None:
            continue
        
        # org_forward(x) + lora_up(lora_down(x)) * multiplier
        # ensure the sequence of lora_up(A) then lora_down(B)
        pair_keys = []
        if 'lora_down' in key:
            pair_keys.append(key.replace('lora_down', 'lora_up'))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace('lora_up', 'lora_down'))
        
        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            # squeeze(3) and squeeze(2) remove dimensions of size 1 from the tensor to make the tensor more compact
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)
            
        # update visited list, ensure no duplicated weight is processed. 
        for item in pair_keys:
            visited.append(item)
    

from diffusers import StableDiffusionXLPipeline
def load_lora_sdxl(
    pipe: StableDiffusionXLPipeline
    , lora_path: str
    , lora_weight = 0.5
    , reset_lora: bool = False
):
    if reset_lora:
        pipe.unload_lora_weights()
        
    #pipe._lora_scale = lora_weight

    state_dict, network_alphas = pipe.lora_state_dict(
        lora_path
        , unet_config = pipe.unet.config
    )

    for key in network_alphas:
        network_alphas[key] = network_alphas[key] * lora_weight

    pipe.load_lora_into_unet(
        state_dict
        , network_alphas = network_alphas
        , unet = pipe.unet
    )

    text_encoder_state_dict = {k: v for k, v in state_dict.items() if "text_encoder." in k}
    if len(text_encoder_state_dict) > 0:
        pipe.load_lora_into_text_encoder(
            state_dict          = text_encoder_state_dict
            , network_alphas    = network_alphas
            , text_encoder      = pipe.text_encoder
            , prefix            = "text_encoder"
            , lora_scale        = lora_weight
        )
    
    text_encoder_2_state_dict = {k: v for k, v in state_dict.items() if "text_encoder_2." in k}
    if len(text_encoder_state_dict) > 0:
        pipe.load_lora_into_text_encoder(
            state_dict          = text_encoder_2_state_dict
            , network_alphas    = network_alphas
            , text_encoder      = pipe.text_encoder_2
            , prefix            = "text_encoder_2"
            , lora_scale        = lora_weight
        )

def load_textual_inversion(
    learned_embeds_path
    , token
    , text_encoder
    , tokenizer
    , weight = 0.5
    , device = "cpu"
):
    '''
    Use this function to load textual inversion model in model initilization stage 
    or image generation stage. 
    Works for Stable Diffusion v1.5 based model
    '''
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location=device)
    if "string_to_token" in loaded_learned_embeds:
        string_to_token = loaded_learned_embeds['string_to_token']
        string_to_param = loaded_learned_embeds['string_to_param']
        
        # separate token and the embeds
        trained_token = list(string_to_token.keys())[0]
        embeds = string_to_param[trained_token]
        embeds = embeds[0] * weight
    elif "emb_params" in loaded_learned_embeds:
        embeds = loaded_learned_embeds["emb_params"][0] * weight
    else:
        keys = list(loaded_learned_embeds.keys())
        embeds =  loaded_learned_embeds[keys[0]] * weight

    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)

    # add the token in tokenizer
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    if num_added_tokens == 0:
        raise ValueError(
            f"""The tokenizer already contains the token {token}. 
            Please pass a different `token` that is not already in the tokenizer."""
        )
    
    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))
    
    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds
    return (tokenizer,text_encoder)


def load_textual_inversion_sdxl(
    pipe:StableDiffusionXLPipeline
    , ti_path
    , token         = "az_diffusers_plus_ti"
    , weight        = 1.0
    , device        = "cpu"
):
    '''
    Use this function to load textual inversion model in model initilization stage 
    or image generation stage. 
    Works for Stable Diffusion XL based model
    '''
    tokenizer_1,tokenizer_2 = pipe.tokenizer, pipe.tokenizer_2
    text_encoder_1, text_encoder_2 = pipe.text_encoder, pipe.text_encoder_2
    
    from safetensors.torch import load_file
    loaded_learned_embeds = load_file(ti_path, device = device)
    # if "string_to_token" in loaded_learned_embeds:
    #     string_to_token = loaded_learned_embeds['string_to_token']
    #     string_to_param = loaded_learned_embeds['string_to_param']
        
    #     # separate token and the embeds
    #     trained_token = list(string_to_token.keys())[0]
    #     embeds = string_to_param[trained_token]
    #     embeds = embeds[0] * weight
    # elif "emb_params" in loaded_learned_embeds:
    #     embeds = loaded_learned_embeds["emb_params"][0] * weight
    # else:
    # keys = list(loaded_learned_embeds.keys())
    embeds_1 =  loaded_learned_embeds["clip_l"]
    embeds_2 =  loaded_learned_embeds["clip_g"]

    # cast to dtype of text_encoder
    dtype_1 = text_encoder_1.get_input_embeddings().weight.dtype
    embeds_1.to(dtype_1)
    dtype_2 = text_encoder_2.get_input_embeddings().weight.dtype
    embeds_2.to(dtype_2)

    # add the token in tokenizer
    token = token if token is not None else "xxx"
    
    num_added_tokens = tokenizer_1.add_tokens(token)
    if num_added_tokens == 0:
        print("existing token")
    #     raise ValueError(
    #         f"""The tokenizer already contains the token {token}. 
    #         Please pass a different `token` that is not already in the tokenizer."""
    #     )
    
    num_added_tokens = tokenizer_2.add_tokens(token)
    if num_added_tokens == 0:
        print("existing token")
    #     raise ValueError(
    #         f"""The tokenizer already contains the token {token}. 
    #         Please pass a different `token` that is not already in the tokenizer."""
    #     )
    
    # resize the token embeddings
    text_encoder_1.resize_token_embeddings(len(tokenizer_1))
    text_encoder_2.resize_token_embeddings(len(tokenizer_2))
    
    # get the id for the token and assign the embeds
    token_id_1 = tokenizer_1.convert_tokens_to_ids(token)
    #eos_1 = tokenizer_1.bos_token_id
    #eos_1_embed = text_encoder_1.get_input_embeddings().weight.data[eos_1]
    #updated_weight_1 = eos_1_embed + (embeds_1[0] - eos_1_embed) * weight
    text_encoder_1.get_input_embeddings().weight.data[token_id_1] = embeds_1[0] * weight
    
    token_id_2 = tokenizer_2.convert_tokens_to_ids(token)
    #eos_2 = tokenizer_2.bos_token_id
    #eos_2_embed = text_encoder_2.get_input_embeddings().weight.data[eos_2]
    #updated_weight_2 = eos_2_embed + (embeds_2[0] - eos_2_embed) * weight
    text_encoder_2.get_input_embeddings().weight.data[token_id_2] = embeds_2[0] * weight
    
    return (tokenizer_1, tokenizer_2,text_encoder_1,text_encoder_2)