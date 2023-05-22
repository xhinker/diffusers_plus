from safetensors.torch import load_file
import os
import torch

def load_lora(
    pipeline
    ,lora_path
    ,lora_weight_delta = 0.5
    ,device = 'cpu'
):
    '''
    lora_weight_delta = new_weight - current_weight
    '''
    state_dict = load_file(lora_path, device=device)
    LORA_PREFIX_UNET = 'lora_unet'
    LORA_PREFIX_TEXT_ENCODER = 'lora_te'

    alpha = lora_weight_delta
    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"
        
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
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += '_'+layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)
        
        # org_forward(x) + lora_up(lora_down(x)) * multiplier
        pair_keys = []
        if 'lora_down' in key:
            pair_keys.append(key.replace('lora_down', 'lora_up'))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace('lora_up', 'lora_down'))
        
        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)
            
        # update visited list
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