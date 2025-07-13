import base64
import os
import anthropic
import json
from typing import Union, List, Dict
import pandas as pd
import tqdm
from PIL import Image
from io import BytesIO
import pickle
import re
import logging
logger = logging.getLogger('livexiv')


MAX_ALLOWED_SIZE = 2 * 1024 * 1024

def init_anthropic_client():        
    anthropic_key = "ENTER YOUR KEY HERE"
    client = anthropic.Anthropic(
        api_key=anthropic_key,
    )
    return client

anthropic_client = init_anthropic_client()


def resize_to_closest_ar(image_path: str):    
    predefined_sizes = {
        (1092, 1092): "1:1",
        (951, 1268): "3:4",
        (896, 1344): "2:3",
        (819, 1456): "9:16",
        (784, 1568): "1:2"
    }
    img = Image.open(image_path)                
    width, height = img.size
    format = img.format    
    input_ratio =  height / width
        
    closest_size = min(predefined_sizes.keys(), 
                       key=lambda size: abs(size[0]/size[1] - input_ratio))
    new_height, new_width = closest_size
    img = img.resize((new_width, new_height), Image.LANCZOS)
    return img, format
    
    
def _resize_image_by_factor(image_path: str, factor: float) -> Image:
        img = Image.open(image_path)                
        width, height = img.size
        format = img.format
        new_width = int(width * factor)
        new_height = int(height * factor)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        return img, format
    

def _image_to_base64(img: Image, format: str):
    buffered = BytesIO()
    img.save(buffered, format=format)  # Use the original image format
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode("utf-8")


def encode_image(image_path: Union[str, Image.Image]):   
    if isinstance(image_path, str):
        suffix = os.path.splitext(image_path)[1][1:]  # ignore '.'
        img_type = f'image/{suffix}'    
        with open(image_path, "rb") as image_file:
            img_data = base64.b64encode(image_file.read()).decode("utf-8")
    else:
        format_ = image_path.format
        img_data = _image_to_base64(image_path, )
        if format_ == 'PNG':
            img_type = f'image/png'    
        elif format_ == 'JPEG':
            img_type = f'image/jpg'    
        else:
            raise ValueError(f'format {format_} is not supported')
    
    factor = 1    
    while len(img_data) >= MAX_ALLOWED_SIZE:
        factor *= 0.9
        img, format = _resize_image_by_factor(image_path, factor)
        img_data = _image_to_base64(img, format)
    
    # img, format = resize_to_closest_ar(image_path)
    # img_data = _image_to_base64(img, format)
    return img_data, img_type


def get_general_description_prompt(text_desc: str) -> str:
    str_ = f"""This is a figure from a scientific paper with the following caption: {text_desc}.
Please describe the image in as much details as possible. For all the details you are confident about include everything you see, and be as specific as possible, such as existing numbers, describing objects, attributes ..."""

    return str_


def get_questions_prompt(gpt_desc: str, n_questions: int = 10) -> str:
    str_1 = f"""Compositional reasoning defines the understanding of attributes, relations and word order significance. A good vision-language model should be able to accurately answer composition reasoning questions about an image. Your task is to fool a vision-language model by generating challenging compositional reasoning questions about the figure. Given the image and the description you generated: {gpt_desc}, generate {n_questions} diverse and challenging compositional reasoning questions which a vision-language model would incorrectly answer."""
    # str_1 = f"""Scientific image understanding defines the understanding of attributes, relations and word order significance. A good vision-language model should be able to accurately answer scientific understanding questions about an image. Your task is to fool a vision-language model by generating challenging scientific image understanding questions about the figure. Given the description you generated: {gpt_desc}, and the given scientific figure generate {n_questions} challenging questions which a vision-language model would incorrectly answer."""
    str_2 = """For each question include the following: - A compositional reasoning question - A correct answer - 3 hard negative options. Each negative option should differ only subtly from the correct answer but still be clearly incorrect given the image, and the question. The goal is for a vision-language model to choose the negative option over the positive option when you asked to answer the question in binary multiple choice format. Only include questions you are confident in your answer and make sure there is indeed only a single correct answer and the others are false answers. Format your response as a string in the format [{"Q":<question>, "a":<correct answer>, "n1":<negative option 1>, "n2":<negative option 2>, ...}]."""
    # str_2 = """For each question include the following: - A scientific image understanding question - A correct answer - 5 hard negative options. Each negative option should differ only subtly from the correct answer but still be clearly incorrect given the figure, caption and the question. The goal is for a vision-language model to choose the negative option over the positive option when you asked to answer the question in binary multiple choice format. Only include questions you are confident in your answer for and make sure there is indeed only a single correct answer and the others are false answers. Format your response as a string in the format [{"Q":<question>, "a":<correct answer>, "n1":<negative option 1>, "n2":<negative option 2>, ...}]."""
    return str_1 + str_2    


def get_table_questions_prompt(table_text: str, n_questions: int = 10, q_type='reasoning') -> str:
    str_1 = f"""Document and table understanding defines the understanding of values, metrics and perform arithmetic operations over numerical values and commonsense reasoning. A good language model should be able to accurately answer {q_type} questions from a given table. Your task is to fool a language model by generating challenging table {q_type} questions about the table. Given the table:\n{table_text}\nGenerate {n_questions} diverse and challenging {q_type} questions on the table questions which a language model would incorrectly answer."""
    str_2 = """For each question include the following: - A question - A correct answer - 3 hard negative options. Each negative option should differ only subtly from the correct answer but still be clearly incorrect given the figure, caption and the question. The goal is for a language model to choose the negative option over the positive option when you asked to answer the question in binary multiple choice format. Only include questions you are confident in your answer and make sure there is indeed only a single correct answer and the others are false answers. Format your response as a string in the format [{"Q":<question>, "a":<correct answer>, "n1":<negative option 1>, "n2":<negative option 2>, ...}]."""
    return str_1 + str_2   

def _get_consistency_prompt_str(q_:List[str] , opts_: List[List], correct_:List[str]) -> str:
    prompt_str = "Think step by step before answering.\n"
    for i in range(len(q_)):
        prompt_str += f"""For the given image and question: {q_[i]}\nwrite only the words yes or no if think the option {correct_[i]} is indeed the correct answer out of {opts_[i]} for this question?\n\n"""
    return prompt_str


def agreement_check(df: pd.DataFrame, img_base_dir: str) -> str:
    check_cache = True
    responses = {}
    letters_to_opts = {'A':0, 'B':1, 'C':2, 'D':3}
    groups = df.groupby('image_path')
    if check_cache:
        cache_filepath = os.path.join(img_base_dir, 'debug_claude.pkl')
        if os.path.isfile(cache_filepath):
            with open(cache_filepath, 'rb') as debug_f:
                responses = pickle.load(debug_f)
    for name, grp in tqdm.tqdm(groups, total=len(groups), desc='Checking agreement with Claude'):        
        q = grp['Q'].to_list()
        idx_list = grp.index.to_list()
        opts = grp['options'].to_list()
        correct_idx = [letters_to_opts[g] for g in grp['gt'].to_list()]
        correct = [o[ci] for o, ci in zip(opts, correct_idx)]
        # img_data, img_type = encode_image(os.path.join(img_base_dir, grp['image_path'].to_list()[0]))
        img_path = os.path.join(img_base_dir, grp['image_path'].to_list()[0])
        if name in responses.keys():
            if responses[name][-1]:
                # logger.info('response found in cache file - skipping')
                continue
        try:            
            prompt = _get_consistency_prompt_str(q, opts, correct)
            raw_content = api_call(prompt, img_path)
            
        except Exception as e:
            logger.warning(e)
            raw_content = ''
                
        responses[name] = [idx_list, q, raw_content]
    
    with open(os.path.join(img_base_dir, 'debug_claude.pkl'), 'wb') as debug_f:
        pickle.dump(responses, debug_f, pickle.HIGHEST_PROTOCOL)
                
    df['Claude-agreement'] = False
    for img_name, vals in responses.items():
        matches = re.findall(r'\b(yes|no)\b', vals[-1], re.IGNORECASE)
        if matches:
            for ind, decision in zip(vals[0], matches):
                df.loc[ind, 'Claude-agreement'] = True if 'yes' in decision.lower() else False
    return df[df['Claude-agreement']]


def api_call(prompt: str, image_path: str, params: Dict = {'temperature': 0.0}, 
             struct_out: bool = False, do_resize: bool = False, keep_res: bool = False) -> str:
    base64_image, img_type = encode_image(image_path)
    params['max_tokens'] = 1024
    messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img_type,
                            "data": base64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]}]
    
    if struct_out:
        messages.append(
            {
                "role": "assistant",
                "content": "Here is the JSON requested:\n["
            }
        )
        
    response = anthropic_client.messages.create(
                            model="claude-3-5-sonnet-20240620",
                            messages=messages,
                            **params,                            
                            )
           
    result = response.content[0].text 
    if struct_out:
        result = "[" + result[:result.rfind("]") + 1]
        
    return result


def main():
    img_data, img_type = encode_image('enter_image')
    with open('json_question_file', 'r') as f:
        data = json.load(f)
        
    for data_point in data:
        question = data_point['Q']
        answer = data_point['a']
        
        message = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img_type,
                            "data": img_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"Look at the image and try to answer the question: {question}. Do you think the answer {answer} is correct? if so say yes, if not, suggest your correct answer shortly?"
                    }
                ],
            }
        ],
        )
        print(f'question: {question} - GPT answer: {answer}')
        print('Claude:')
        print(message.content[0].text)

    
if __name__ == "__main__":
    main()