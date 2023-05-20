# The call sequence
# 

import re
import logging
from .prompt_parser import get_learned_conditioning_prompt_schedules

logger = logging.getLogger(__name__)

re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)


class AZ_SD_Prompt:
    def __init__(self) -> None:
        pass
    
    def parse_scheduled_prompts(text,steps=30):
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

    def func_test(input:str):
        print(input)