# The call sequence
# 1. parse_scheduled_prompts -> 2.get_weighted_text_embeddings -> 3. get_prompts_with_weights
# -> 4. pad_tokens_and_weights -> 5. get_unweighted_text_embeddings 
# -> 

import logging
import torch
from typing import Callable,List, Optional, Union
from diffusers import (
    StableDiffusionPipeline
)

from .prompt_parser import (
    get_learned_conditioning_prompt_schedules
    , parse_prompt_attention
)

logger = logging.getLogger(__name__)

class AZ_SD_Prompt:
    def __init__(self) -> None:
        pass
    
    def parse_scheduled_prompts(self,text,steps=30):
        '''
        This function will handle scheduled and alternative prompt
        '''
        text = text.strip()
        parse_result = None
        try:
            parse_result = get_learned_conditioning_prompt_schedules([text],steps=steps)[0]
            logger.info(
                f"parse_result from get_learned_conditioning_prompt_schedules function:\n {str(parse_result)}"
            )
        except Exception as e:
            logger.error(f"Parse scheduled prompt error:\n {e}")

        if len(parse_result) == 1:
            # no scheduling
            return parse_result
        
        prompts_list = []
        for i in range(steps):
            current_prompt_step, current_prompt_content = parse_result[0][0],parse_result[0][1]
            step = i + 1
            if step < current_prompt_step:
                prompts_list.append(current_prompt_content)
                continue
            
            if step == current_prompt_step:
                prompts_list.append(current_prompt_content)
                parse_result.pop(0)
            
        return prompts_list
    
    def get_weighted_text_embeddings(
        self
        , pipe: StableDiffusionPipeline
        , prompt: Union[str, List[str]]
        , uncond_prompt: Optional[Union[str, List[str]]] = None
        , max_embeddings_multiples: Optional[int] = 3
        , no_boseos_middle: Optional[bool] = False
        , skip_parsing: Optional[bool] = False
        , skip_weighting: Optional[bool] = False
    ):
        r"""
        Prompts can be assigned with local weights using brackets. For example,
        prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
        and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.

        Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.

        Args:
            pipe (`StableDiffusionPipeline`):
                Pipe to provide access to the tokenizer and the text encoder.
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            uncond_prompt (`str` or `List[str]`):
                The unconditional prompt or prompts for guide the image generation. If unconditional prompt
                is provided, the embeddings of prompt and uncond_prompt are concatenated.
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
            no_boseos_middle (`bool`, *optional*, defaults to `False`):
                If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
                ending token in each of the chunk in the middle.
            skip_parsing (`bool`, *optional*, defaults to `False`):
                Skip the parsing of brackets.
            skip_weighting (`bool`, *optional*, defaults to `False`):
                Skip the weighting. When the parsing is skipped, it is forced True.
        """
        # pipe.tokenizer.model_max_length == 77
        max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
        if isinstance(prompt, str):
            prompt = [prompt]

        if not skip_parsing:
            prompt_tokens, prompt_weights = self.get_prompts_with_weights(pipe, prompt, max_length - 2)
            if uncond_prompt is not None:
                if isinstance(uncond_prompt, str):
                    uncond_prompt = [uncond_prompt]
                uncond_tokens, uncond_weights = self.get_prompts_with_weights(pipe, uncond_prompt, max_length - 2)
        else:
            prompt_tokens = [
                token[1:-1] for token in pipe.tokenizer(prompt, max_length=max_length, truncation=True).input_ids
            ]
            prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
            if uncond_prompt is not None:
                if isinstance(uncond_prompt, str):
                    uncond_prompt = [uncond_prompt]
                uncond_tokens = [
                    token[1:-1]
                    for token in pipe.tokenizer(uncond_prompt, max_length=max_length, truncation=True).input_ids
                ]
                uncond_weights = [[1.0] * len(token) for token in uncond_tokens]

        # round up the longest length of tokens to a multiple of (model_max_length - 2)
        max_length = max([len(token) for token in prompt_tokens])
        if uncond_prompt is not None:
            max_length = max(max_length, max([len(token) for token in uncond_tokens]))

        max_embeddings_multiples = min(
            max_embeddings_multiples,
            (max_length - 1) // (pipe.tokenizer.model_max_length - 2) + 1,
        )
        max_embeddings_multiples = max(1, max_embeddings_multiples)
        max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

        # pad the length of tokens and weights
        bos = pipe.tokenizer.bos_token_id
        eos = pipe.tokenizer.eos_token_id
        prompt_tokens, prompt_weights = self.pad_tokens_and_weights(
            prompt_tokens,
            prompt_weights,
            max_length,
            bos,
            eos,
            no_boseos_middle=no_boseos_middle,
            chunk_length=pipe.tokenizer.model_max_length,
        )
        logger.warn(f"pipe device:{pipe.device}")
        prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=pipe.device)
        if uncond_prompt is not None:
            uncond_tokens, uncond_weights = self.pad_tokens_and_weights(
                uncond_tokens,
                uncond_weights,
                max_length,
                bos,
                eos,
                no_boseos_middle=no_boseos_middle,
                chunk_length=pipe.tokenizer.model_max_length,
            )
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=pipe.device)

        # get the embeddings
        text_embeddings = self.get_unweighted_text_embeddings(
            pipe,
            prompt_tokens,
            pipe.tokenizer.model_max_length,
            no_boseos_middle=no_boseos_middle,
        )
        prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=pipe.device)
        logger.warn(f"prompt_weights device:{prompt_weights.device}")
        logger.warn(f"text_embeddings device:{text_embeddings.device}")

        if uncond_prompt is not None:
            uncond_embeddings = self.get_unweighted_text_embeddings(
                pipe,
                uncond_tokens,
                pipe.tokenizer.model_max_length,
                no_boseos_middle=no_boseos_middle,
            )
            uncond_weights = torch.tensor(uncond_weights, dtype=uncond_embeddings.dtype, device=pipe.device)

        # assign weights to the prompts and normalize in the sense of mean
        # TODO: should we normalize by chunk or in a whole (current implementation)?
        if (not skip_parsing) and (not skip_weighting):
            previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
            text_embeddings *= prompt_weights.unsqueeze(-1)
            current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
            text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
            if uncond_prompt is not None:
                previous_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
                uncond_embeddings *= uncond_weights.unsqueeze(-1)
                current_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
                uncond_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

        if uncond_prompt is not None:
            return text_embeddings, uncond_embeddings
        return text_embeddings, None
    
    def get_prompts_with_weights(
        self 
        , pipe: StableDiffusionPipeline
        , prompt: List[str]
        , max_length: int):
        r"""
        Tokenize a list of prompts and return its tokens with weights of each token.

        No padding, starting or ending token is included.
        """
        tokens = []
        weights = []
        truncated = False
        for text in prompt:
            texts_and_weights = parse_prompt_attention(text)
            text_token = []
            text_weight = []
            for word, weight in texts_and_weights:
                # skip separation tokens
                # if (word.strip() == ",") or (word.strip() == "."):
                #     continue

                # tokenize and discard the starting and the ending token
                token = pipe.tokenizer(word).input_ids[1:-1]
                text_token += token
                # copy the weight by length of token
                text_weight += [weight] * len(token)
                # stop if the text is too long (longer than truncation limit)
                if len(text_token) > max_length:
                    truncated = True
                    break
            # truncate
            if len(text_token) > max_length:
                truncated = True
                text_token = text_token[:max_length]
                text_weight = text_weight[:max_length]
            tokens.append(text_token)
            weights.append(text_weight)
        if truncated:
            logger.warning("Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
        return tokens, weights

    def pad_tokens_and_weights(
        self
        , tokens
        , weights
        , max_length
        , bos
        , eos
        , no_boseos_middle=True
        , chunk_length=77
    ):
        r"""
        Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
        """
        max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
        weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
        for i in range(len(tokens)):
            tokens[i] = [bos] + tokens[i] + [eos] * (max_length - 1 - len(tokens[i]))
            if no_boseos_middle:
                weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
            else:
                w = []
                if len(weights[i]) == 0:
                    w = [1.0] * weights_length
                else:
                    for j in range(max_embeddings_multiples):
                        w.append(1.0)  # weight for starting token in this chunk
                        w += weights[i][j * (chunk_length - 2) : min(len(weights[i]), (j + 1) * (chunk_length - 2))]
                        w.append(1.0)  # weight for ending token in this chunk
                    w += [1.0] * (weights_length - len(w))
                weights[i] = w[:]

        return tokens, weights
    
    def get_unweighted_text_embeddings(
        self
        , pipe: StableDiffusionPipeline
        , text_input: torch.Tensor
        , chunk_length: int
        , no_boseos_middle: Optional[bool] = True
    ):
        """
        When the length of tokens is a multiple of the capacity of the text encoder,
        it should be split into chunks and sent to the text encoder individually.
        """
        max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
        if max_embeddings_multiples > 1:
            text_embeddings = []
            for i in range(max_embeddings_multiples):
                # extract the i-th chunk
                text_input_chunk = text_input[:, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2].clone()

                # cover the head and the tail by the starting and the ending tokens
                text_input_chunk[:, 0] = text_input[0, 0]
                text_input_chunk[:, -1] = text_input[0, -1]
                text_embedding = pipe.text_encoder(text_input_chunk)[0]

                if no_boseos_middle:
                    if i == 0:
                        # discard the ending token
                        text_embedding = text_embedding[:, :-1]
                    elif i == max_embeddings_multiples - 1:
                        # discard the starting token
                        text_embedding = text_embedding[:, 1:]
                    else:
                        # discard both starting and ending tokens
                        text_embedding = text_embedding[:, 1:-1]

                text_embeddings.append(text_embedding)
            text_embeddings = torch.concat(text_embeddings, axis=1)
        else:
            text_embeddings = pipe.text_encoder(text_input)[0]
        return text_embeddings

    def func_test(input:str):
        print(input)