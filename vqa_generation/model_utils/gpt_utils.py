import base64
import os
from openai import OpenAI
import json
from io import BytesIO
from PIL import Image
import numpy as np
import re
import pickle
import pandas as pd
import tqdm
from typing import List, Dict
import logging
logger = logging.getLogger('livexiv')


def init_openai_client():    
    openai_key = 'ENTER YOUR KEY HERE'
    client = OpenAI(
        api_key=openai_key,
    )
    return client

openai_client = init_openai_client()


def nearest_multiple_crop(value, crop_sz):
    return (value // crop_sz) * crop_sz        


def _resize_image_by_factor(image_path: str, factor: float) -> Image:
        CROP_SZ = 512
        img = Image.open(image_path)                
        width, height = img.size
        ratio = width / height
        format = img.format
        if width > height:
            new_width = width if width <= CROP_SZ else nearest_multiple_crop(width, CROP_SZ)
            if new_width != width:
                new_height = new_width / ratio
                new_height = new_height if new_height <= CROP_SZ else nearest_multiple_crop(new_height, CROP_SZ)
            else:
                new_height = height
        else:
            new_height = height if height <= CROP_SZ else nearest_multiple_crop(height, CROP_SZ)
            if new_height != height:
                new_width = new_height * ratio
                new_width = new_width if new_width <= CROP_SZ else nearest_multiple_crop(new_width, CROP_SZ)
            else:
                new_width = width
        new_width = int(np.round(new_width))
        new_height = int(np.round(new_height))  
        
        # print(f'Original dims: {width}X{height}') 
        # print(f'resized dims: {new_width}X{new_height}')              
        
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        return img, format
    

def _image_to_base64(img: Image, format: str):
    buffered = BytesIO()
    img.save(buffered, format=format)  # Use the original image format
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode("utf-8")


def encode_image(image_path: str):
    img = Image.open(image_path)
    fmt = img.format     
    img_data = _image_to_base64(img, fmt)    
    return img_data
    

def get_general_description_prompt(text_desc: str = '') -> str:
    if text_desc:
        str_ = f"""This is a figure from a scientific paper with the following caption: {text_desc}.
    Please describe the image in as much details as possible. For all the details you are confident about include everything you see, and be as specific as possible, such as existing numbers, describing objects, attributes ..."""
    else:
            str_ = f"""This is a figure from a scientific paper.
    Please describe the image in as much details as possible. For all the details you are confident about include everything you see, and be as specific as possible, such as existing numbers, describing objects, attributes ..."""

    return str_


def generate_masking_prompt_from_text(text_ : str, detailed_desc: str = '') -> str:
    if detailed_desc:
        str_ = f"""Think step by step before answering, Given the following image caption:\n{text_} Mask at least 2 or 3 parts of the caption that can only be revealed by looking at the image, for example locations, names, colors, etc... Avoid masking information that cannot be directly obtain from the image and from the following detailed description {detailed_desc}.\nReturn the masked captions, replace the masked words with <mask>, make sure the masked words """
    else:
        str_ = f"""Think step by step before answering, Given the following image caption:\n{text_}\nMask at least 2 or 3 parts of the caption that can only be revealed by looking at the image, for example locations, names, colors, etc... Avoid masking information that cannot be obtained from the image. \nReturn the masked caption, replace the masked words with <mask> and provide the masked words as a python list.
        example for output: Figure 2. the <mask> is playing <mask>.\n\n ['child', 'basketball']"""
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


def generate_descriptions(classes):
    desc_data = {cls: [] for cls in classes}
    for cls in classes:
        prompts = []
        prompts.append("In one sentence, How would you visually recognize the " + cls + " in a figure?")
        prompts.append("In one sentence, Describe the appearance of " + cls + "in an image?")
        prompts.append("In one sentence, What visual cues help in identifying the " + cls + " in an image?")
        prompts.append("In one sentence, What visual patterns are commonly observed in the " + cls + "?")
        prompts.append("In one sentence, How can one visually recognize the " + cls + " in an image?")

        res_ = {}
        for curr_prompt in prompts:
            all_result = []

            response = openai_client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "You are a helpful and coincise AI assistant."},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"{curr_prompt}"}                    
                    ]}
                ],
                temperature=.99,
                max_tokens=70,
                n=10,
            )

            for r in range(len(response.choices)):
                result = response.choices[r].message.content
                all_result.append(result.replace("\n\n", ""))

            res_[curr_prompt] = all_result
        
        desc_data[cls] = res_
    with open('descriptions.json', 'w') as f:
        json.dump(desc_data, f, indent=4)


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
        cache_filepath = os.path.join(img_base_dir, 'debug_gpt.pkl')
        if os.path.isfile(cache_filepath):
            with open(cache_filepath, 'rb') as debug_f:
                responses = pickle.load(debug_f)
    for name, grp in tqdm.tqdm(groups, total=len(groups), desc='Checking agreement with GPT'):        
        q = grp['Q'].to_list()
        idx_list = grp.index.to_list()
        opts = grp['options'].to_list()
        correct_idx = [letters_to_opts[g] for g in grp['gt'].to_list()]
        correct = [o[ci] for o, ci in zip(opts, correct_idx)]
        # img_data = encode_image(os.path.join(img_base_dir, grp['image_path'].to_list()[0]))
        img_path = os.path.join(img_base_dir, grp['image_path'].to_list()[0])
        if name in responses.keys() and responses[name][-1]:
            continue
        try:            
            prompt = _get_consistency_prompt_str(q, opts, correct)
            raw_content = api_call(prompt, img_path)

        
        except Exception as e:
            logger.warning(e)
            raw_content = ''
                
        responses[name] = [idx_list, q, raw_content]
    
    with open(os.path.join(img_base_dir, 'debug_gpt.pkl'), 'wb') as debug_f:
        pickle.dump(responses, debug_f, pickle.HIGHEST_PROTOCOL)
                
    df['GPT-agreement'] = False
    for img_name, vals in responses.items():
        matches = re.findall(r'\b(yes|no)\b', vals[-1], re.IGNORECASE)
        if matches:
            for ind, decision in zip(vals[0], matches):
                df.loc[ind, 'GPT-agreement'] = True if 'yes' in decision.lower() else False
    return df[df['GPT-agreement']]


def api_call(prompt: str, image_path: str, params: Dict = {'temperature': 0.0}, 
             struct_out: bool = False, do_resize: bool = False, keep_res: bool = False) -> str:
    base64_image = encode_image(image_path)
    if struct_out:
        params['response_format'] = {
                                    "type": "json_schema",
                                    "json_schema": {
                                        "name": "vqa",
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "questions": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "question": {"type": "string"},
                                                            "a": {"type": "string"},
                                                            "n1": {"type": "string"},
                                                            "n2": {"type": "string"},
                                                            "n3": {"type": "string"},
                                                        },
                                                        "required": ["question", "a", "n1", "n2", "n3"],
                                                        "additionalProperties": False
                                                    }
                                                },                    
                                            },
                                            "required": ["questions"],
                                            "additionalProperties": False
                                        },
                                        "strict": True
                                    }
                                }                            
        
    response = openai_client.chat.completions.create(
                            model="gpt-4o-2024-08-06",
                            messages=[
                                {"role": "system", "content": "You are a helpful AI visual assistant who can analyze images."},
                                {"role": "user", "content": [
                                    {"type": "text", "text": f"{prompt}"},
                                    {"type": "image_url", "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"}
                                    }
                                ]}
                            ],
                            **params
                        )
    # if struct_out and response.choices[0].message.refusal is None:        
    #     result = response.choices[0].message.content
    # else:
    result = response.choices[0].message.content
        
    return result


    
