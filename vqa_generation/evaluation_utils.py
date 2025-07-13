import json
import glob
from typing import List, Dict
from pathlib import Path
import tqdm
import random
import os
import re
import base64
from io import BytesIO
from PIL import Image
random.seed(12345)


def collect_all_questions(src_dir: str, q_type: str, gen_model:str) -> List[str]:
    if q_type == 'figure':
        suffix = 'questions.json'
        img_suffix = ''
    elif q_type == 'table':
        suffix = 'questions_table.json'
        img_suffix = '_table'
        
    recover_text_to_json(src_dir)    
    all_questions_files = sorted(glob.glob(src_dir + f'/*{suffix}'))    
    all_questions = []
    for qfile in tqdm.tqdm(all_questions_files, total=len(all_questions_files), desc='Collecting questions from files'):
        p = Path(qfile)
        try:
            with open(qfile, 'r') as qf:
                cur_data = json.load(qf)
            if isinstance(cur_data, dict):
                cur_data = cur_data['questions']
                
            cur_data = [{**c, 'arxiv_ref': p.stem[:p.stem.rfind('_')], 'gen_model': gen_model, 'image_path': os.path.basename(qfile).replace(f'_{suffix}', f'{img_suffix}.png')} for c in cur_data]
            all_questions.extend(cur_data)
        except Exception as e:
            print(e)
            continue
    
    return all_questions



def recover_text_to_json(src_dir: str) -> None:
    dd = {}
    files = glob.glob(src_dir + '/*questions*.txt')
    for fpath in files:
        with open(fpath, 'r') as f:
            data = f.readlines()
        try:
            # print(data)
            dd[Path(fpath).stem] = []
            start_line = [i for i in range(len(data)) if data[i].startswith('[')]        
            end_line = [i for i in range(len(data)) if data[i].startswith(']')]
            if all([start_line, end_line]):
                assert len(start_line) == len(end_line)
                l_ = []
                for s,e in zip(start_line, end_line):                                
                    d_ = json.loads("".join(data[s:e+1]))
                    l_.extend(d_)
                
                with open(fpath.replace('.txt', '.json'), 'w') as f:
                    json.dump(l_, f, indent=4)
                                
        except Exception as e:
            print(fpath, e)            
        
        
def extract_gt(gt_val: str, shuffled_options: List) -> str:
    ind2let = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    ind = shuffled_options.index(gt_val)
    return ind2let[ind]
    
    
def prepare_questions_for_eval(questions_list: List[Dict], save_path: str = '') -> List[Dict]:
    all_questions_for_eval = []
    for q in tqdm.tqdm(questions_list, total=len(questions_list), desc='Preparing questions'):
        try:
            negative_options = [v for k, v in q.items() if k.startswith('n')]
            random.shuffle(negative_options)
            negative_options = negative_options[:3]
            options = [q['a']] + negative_options                    
            random.shuffle(options)  
            question = q['question'] if 'question' in q else q['Q']
            t = {'arxiv_ref': q['arxiv_ref'], 'Q': question, 'options': options, 'image_path': q['image_path'], 'gt': extract_gt(q['a'], options), 'gen_model': q['gen_model']}
            all_questions_for_eval.append(t)
        except Exception as e:
            print(e)
            print(q)
            continue
    
    random.shuffle(all_questions_for_eval)
    if save_path:
        with open(save_path, 'w') as savef:
            json.dump(all_questions_for_eval, savef, indent=4)
            
    return all_questions_for_eval
        

def construct_questions(artifacts_dir: str, questions_filname: str, q_type: str, gen_model: str) -> List[Dict]:
    ret_questions_list = []
    if q_type == 'figure':        
        if questions_filname:
            q_filename = questions_filname
        else:
            q_filename = os.path.join(artifacts_dir, 'unfiltered_questions_for_eval_fig.json')
        if os.path.isfile(q_filename):
            with open(q_filename, 'r') as cache_f:
                all_questions_for_eval_fig = json.load(cache_f)
        else:
            all_questions = collect_all_questions(artifacts_dir, q_type, gen_model)
            all_questions_for_eval_fig = prepare_questions_for_eval(all_questions, save_path=q_filename)
        
        ret_questions_list.append(all_questions_for_eval_fig) 
        
    if q_type == 'table':        
        if questions_filname:
            q_filename = questions_filname
        else:
            q_filename = os.path.join(artifacts_dir, 'unfiltered_questions_for_eval_table.json')
        if os.path.isfile(q_filename):
            with open(q_filename, 'r') as cache_f:
                all_questions_for_eval_table = json.load(cache_f)
        else:
            all_questions = collect_all_questions(artifacts_dir, q_type, gen_model)
            all_questions_for_eval_table = prepare_questions_for_eval(all_questions, save_path=q_filename)
        
        ret_questions_list.append(all_questions_for_eval_table) 

    return ret_questions_list
    

def try_to_fix_preds(pred: str) -> str:
    pattern = r'[A-Z]'
    matches = re.findall(pattern, pred)
    if matches:
        return matches[0]
    else:
        return pred
    
    
def encode_image(img: Image) -> str:
    buffered = BytesIO()
    format_ = img.format
    img.save(buffered, format=format_)
    buffered.seek(0)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str, format_