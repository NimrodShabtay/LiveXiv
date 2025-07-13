import os
import tqdm
import json
from typing import Dict, List
import argparse
import pandas as pd
from classify_figures import classify_figure_type
from model_utils.claude_utils import agreement_check as agreement_check_claude
from model_utils.gpt_utils import agreement_check as agreement_check_gpt
from functools import partial

import logging
logger = logging.getLogger('livexiv')


def without_duplicates(lst):
    return len(lst) == len(set(lst))


def find_different_rows(*dataframes):
    if len(dataframes) < 2:
        raise ValueError("At least two DataFrames are required for comparison")
    
    # Merge all DataFrames
    merged_df = dataframes[0].copy()
    for i, df in enumerate(dataframes[1:], start=1):
        merged_df = merged_df.merge(df,on='Unnamed: 0', how='left', suffixes=('', f'_{i}'))
    
    # Identify columns to compare
    base_columns = ['acc-fixed']
    # Create a mask for rows with differences
    res = []
    for col in base_columns:
        compare_cols = [col] + [f"{col}_{i}" for i in range(1, len(dataframes))]
        
        for i, row in merged_df.iterrows():
            if all(row[cc] == row['acc-fixed'] for cc in compare_cols) and row['acc-fixed']:
                res.append(row)

        return res
    

def merge_and_remove_questions(all_questions_filename: str, 
                               list_of_rows_to_exclude: List, 
                               figure_type_to_exclude: List[str], 
                               save_path: str, 
                               artifacts_dir: str, 
                               do_agreement: bool, 
                               filter_model: str):
    with open(all_questions_filename, 'r') as unfiltered_f:
        unfiltered_questions = json.load(unfiltered_f)
    
    logger.info(f'Before any filtering: {len(unfiltered_questions)}')
        
    # Filter by blind check
    filtered_questions = pd.DataFrame.from_records(list_of_rows_to_exclude).Question
    unfiltered_questions_df = pd.DataFrame.from_records(unfiltered_questions)  
        
    filtered_dataset = unfiltered_questions_df[~unfiltered_questions_df.Q.isin(filtered_questions)]
    logger.info(f'After remove consistent blind easy questions: {len(filtered_dataset)}')
        
    # Filter out duplicate answers
    filtered_dataset = filtered_dataset.drop_duplicates(subset='Q')  # remove duplicate questions
    filtered_dataset = filtered_dataset[filtered_dataset['options'].apply(without_duplicates)]  # remove questions with duplicate options
    logger.info(f'After remove duplications: {len(filtered_dataset)}')
    
    #Filter by chart type    
    if not 'figure_type' in filtered_dataset.columns:
        filtered_dataset = classify_figure_type(artifacts_dir, filtered_dataset, device='cpu')        
        
    for figure_to_exclude in figure_type_to_exclude:        
        filtered_dataset = filtered_dataset[filtered_dataset['figure_type'] != figure_to_exclude]        
        logger.info(f'After remove "{figure_to_exclude}" figures related questions: {len(filtered_dataset)}')
    
    # Filter by other VLM
    if do_agreement:
        if filter_model == 'gpt':
            agreement_check_fn = agreement_check_gpt
            logger.info('Agreement using GPT')
        elif filter_model == 'claude':
            agreement_check_fn = agreement_check_claude
            logger.info('Agreement using Claude')
        else:
            raise ValueError(f'Filter model {filter_model} not recognized')
            
        filtered_dataset = agreement_check_fn(df=filtered_dataset, img_base_dir=artifacts_dir)
        logger.info(f'After remove disagreed questions: {len(filtered_dataset)}')
    
    logger.info(f'Filtered dataset size: {len(filtered_dataset)}')
    
    filtered_rows = []
    for idx, row in filtered_dataset.iterrows():
          filtered_rows.append({'arxiv_ref': row['arxiv_ref'], 
                                'Q': row['Q'], 'options': row['options'], 
                                'image_path': row['image_path'], 
                                'gt': row['gt'], 
                                'figure_type': row['figure_type'],
                                'gen_model': row['gen_model'],
                                'filter_model': filter_model})
        
    with open(save_path, 'w') as savef:
        json.dump(filtered_rows, savef, indent=4)
    logger.info(f'Filtered Dataset saved: {save_path}')


def group_by(list_of_dicts: List[Dict], group_key: str ='arxiv_ref') -> Dict:
    grouped_dict = {}
    for item in list_of_dicts:
        arxiv_ref = item.get(group_key)
        if arxiv_ref not in grouped_dict:
            grouped_dict[arxiv_ref] = []
        grouped_dict[arxiv_ref].append(item)
    return grouped_dict


def create_prompt(entry: Dict) -> str:
    options_str = "\n".join([f'{a}. {b}' for a, b in zip(['A', 'B', 'C', 'D'], entry['options'])])
    q = entry['Q']            
    prompt_str = f"{q}\n{options_str}\nAnswer only with the option's letter from the given choices directly."
    return prompt_str
  

def filter_questions_based_on_blind_preds(questions: List[Dict], blind_pred_file: str) -> List[Dict]:
        def _find_element_based_on_question(list_of_dicts: List[Dict], q: str) -> int:
            ret_val = -1
            for ld in list_of_dicts:
                if ld['Q'] == q:
                    ret_val = list_of_dicts.index(ld)    
                    break
                
            return ret_val
        
        
        questions_filter = questions.copy()
        blind_df = pd.read_csv(blind_pred_file)
        indices_to_delete = []
        for _, row in tqdm.tqdm(blind_df.iterrows(), total=len(blind_df), desc='Filtering questions based on blind results'):
            try:
                question = row['Question']
                index_to_remove = _find_element_based_on_question(questions, question)
                indices_to_delete.append(index_to_remove)
                
            except Exception as e:
                logger.warning(e)
                logger.warning(question)
                logger.warning(index_to_remove)
        
        for i in sorted(indices_to_delete, reverse=True):
            questions_filter.pop(i) 
            
        return questions_filter
    
        
def filter_based_on_llm(args):
    from llava_eval import eval_llava    

    blind_pred_file = args.filter_results_name
    
    if not os.path.isfile(blind_pred_file):   
        args.results_name = args.filter_results_name
        eval_llava(args)
            

def remove_questions(all_questions_filename: str, save_path: str, artifacts_dir: str, do_agreement: bool, filter_model: str):
    with open(all_questions_filename, 'r') as unfiltered_f:
        unfiltered_questions = json.load(unfiltered_f)
    
    logger.info(f'Before any filtering: {len(unfiltered_questions)}')
    
    filtered_dataset = pd.DataFrame.from_dict(unfiltered_questions)
    
    # Filter out duplicate answers
    filtered_dataset = filtered_dataset.drop_duplicates(subset='Q')  # remove duplicate questions
    filtered_dataset = filtered_dataset[filtered_dataset['options'].apply(without_duplicates)]  # remove questions with duplicate options
    logger.info(f'After remove duplications: {len(filtered_dataset)}')
    
    # Filter by other VLM
    if do_agreement:
        if filter_model == 'gpt':
            agreement_check_fn = agreement_check_gpt
            logger.info('Agreement using GPT')
        elif filter_model == 'claude':
            agreement_check_fn = agreement_check_claude
            logger.info('Agreement using Claude')
        else:
            raise ValueError(f'Filter model {filter_model} not recognized')
            
    filtered_dataset = agreement_check_fn(df=filtered_dataset, img_base_dir=artifacts_dir) 
    logger.info(f'After remove disagreed questions: {len(filtered_dataset)}')
    
    logger.info(f'Filtered dataset size: {len(filtered_dataset)}')
    
    filtered_rows = []
    for idx, row in filtered_dataset.iterrows():
          filtered_rows.append({'arxiv_ref': row['arxiv_ref'], 
                                'Q': row['Q'], 
                                'options': row['options'], 
                                'image_path': row['image_path'], 
                                'gt': row['gt'],
                                'gen_model': row['gen_model'],
                                'filter_model': filter_model})
    
    with open(save_path, 'w') as savef:
        json.dump(filtered_rows, savef, indent=4)
    logger.info(f'Filtered Dataset saved: {save_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="liuhaotian/llava-v1.6-34b", help="Model name in HF format")
    parser.add_argument("--artifacts_dir", type=str, default='./mm_live_bench/artifacts', help="Artifacts dir")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--filter_results_name", type=str, default='llava_1_6_34B_blind_results.csv', help="Results filename")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--remove_image_token", action='store_true')
    args = parser.parse_args()
    
    filter_based_on_llm(args) 