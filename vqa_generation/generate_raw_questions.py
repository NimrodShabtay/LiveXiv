import glob
import os
import json
import tqdm
import argparse
from pathlib import Path
import pandas as pd

from arxiv_utils import download_arxiv_papers
from document_utils import  load_description_from_image_path

from model_utils.gpt_utils import api_call as api_call_gpt
from model_utils.gpt_utils import get_questions_prompt as get_questions_prompt_gpt
from model_utils.gpt_utils import get_general_description_prompt as get_general_description_prompt_gpt
from model_utils.gpt_utils import get_table_questions_prompt as get_table_questions_prompt_gpt

from model_utils.claude_utils import api_call as api_call_claude
from model_utils.claude_utils import get_questions_prompt as get_questions_prompt_claude
from model_utils.claude_utils import get_general_description_prompt as get_general_description_prompt_claude
from model_utils.claude_utils import get_table_questions_prompt as get_table_questions_prompt_claude

from classify_figures import classify_figure_type
import logging
logger = logging.getLogger('livexiv')


def replace_delimiters(text, delimiters=[".", "_"], replacement="-"):    
    result = text
    for delimiter in delimiters:
        result = result.replace(delimiter, replacement)
    return result


def generate_raw_questions(args):
    os.makedirs(args.artifacts_dir, exist_ok=True)
    # ArXiv download
    existing_pdf_paths = glob.glob(args.artifacts_dir + '/*.pdf')
    if not args.generate.external_paper_list:
        if args.arxiv.do_download or not existing_pdf_paths:  # Download on empty folder or append to folder with papers
            pdf_paths = download_arxiv_papers(download_path=args.artifacts_dir, max_results=args.arxiv.max_results)
        else:
            pdf_paths = existing_pdf_paths
    else:
        import pickle
        with open(args.generate.external_paper_list, 'rb') as f:
            external_pdf_list = pickle.load(f)
            
        pdf_paths = external_pdf_list
                
    logger.info(f'{len(pdf_paths)} arxiv papers downloaded')
    api_cnt = 0
    
    generation_model = args.models.generation
    if generation_model == 'gpt':
        api_call_fn = api_call_gpt
        get_questions_prompt_fn = get_questions_prompt_gpt
        get_general_description_prompt_fn = get_general_description_prompt_gpt
        get_table_questions_prompt_fn = get_table_questions_prompt_gpt
    elif generation_model == 'claude':
        api_call_fn = api_call_claude
        get_questions_prompt_fn = get_questions_prompt_claude
        get_general_description_prompt_fn = get_general_description_prompt_claude
        get_table_questions_prompt_fn = get_table_questions_prompt_claude
    else:
        raise ValueError(f'Generation model is not recognized {generation_model}')
    
    logger.info(f'Chosen generation model: {generation_model}')
    if args.assests_type in ['all', 'figures']:        
        logger.info('VQA generation starts')
        figures_path = os.path.join(args.artifacts_dir, 'figures')
        os.makedirs(figures_path, exist_ok=True)
                            
        # Classify figure types using 0-shot classifier
        logger.info('Classifying Figures types for metadata')
        fig_type_list = classify_figure_type(artifacts_dir=figures_path, device='cpu')                            
        unordered_image_list = sorted(glob.glob(figures_path + '/*.png'))    
        
        images_by_paper = {}        
        for img_path in unordered_image_list:
            arxiv_prefix = os.path.basename(img_path).split('_')[0]
            for pdf in pdf_paths:                
                pdf_name = replace_delimiters(Path(pdf).stem)           
                if arxiv_prefix.lower() in pdf_name.lower():                      
                    if arxiv_prefix not in images_by_paper:            
                        images_by_paper[arxiv_prefix] = []
                    try:
                        img_cls_type_ = fig_type_list[img_path]
                    except:
                        img_cls_type_ = ''
                    images_by_paper[arxiv_prefix].append([img_path, img_cls_type_])
        
        logger.info('VQA questions generation')
        for paper, images in tqdm.tqdm(images_by_paper.items(), total=len(images_by_paper), desc='Iterating over papers'):    
            for img_path, img_cls_type in tqdm.tqdm(images, total=len(images), desc='generating questions for images', leave=False):                    
                if img_cls_type in args.filtering.fig_type_to_exclude:
                    continue
                try:
                    text_description = load_description_from_image_path(img_path)
                    if not text_description:            
                        continue
                    
                    desc_file = img_path.replace('.png', '_desc.txt')
                    desc_prompt = get_general_description_prompt_fn(text_description)
                    if os.path.isfile(desc_file):
                        with open(desc_file) as desc_f:
                            gpt_desc = desc_f.read()
                    else:
                        gpt_desc = api_call_fn(prompt=desc_prompt, image_path=img_path, keep_res=True)
            
                        api_cnt += 1
                        with open(desc_file, 'w') as desc_f:
                            desc_f.write(gpt_desc)
                            
                    questions_prompt = get_questions_prompt_fn(gpt_desc, n_questions=7)
                    q_file = img_path.replace('.png', '_questions')
                    if not any([os.path.isfile(f'{q_file}.json'), os.path.isfile(f'{q_file}.txt')]):
                        if args.generate.freestyle:
                            struct_out = False
                        else:
                            struct_out = True
                        api_cnt += 1
                        gpt_questions = api_call_fn(prompt=questions_prompt, image_path=img_path, keep_res=True, struct_out=struct_out, params={'temperature': 0.1})                        
                            
                        try:
                            gpt_questions_json = json.loads(gpt_questions)
                            q_file += '.json'
                            with open(q_file, 'w') as q_f:
                                json.dump(gpt_questions_json, q_f, indent=4)
                        except:
                            q_file += '.txt'
                            with open(q_file, 'w') as q_f:
                                q_f.write(gpt_questions)
                                                            
                except Exception as e:
                    logger.warning(e)

        logger.info('VQA generation ends')

    if args.assests_type in ['all', 'tables']:
        logger.info('TQA generation starts')
        tables_path = os.path.join(args.artifacts_dir, 'tables')
        os.makedirs(tables_path, exist_ok=True)
        
        unordered_table_image_list = sorted(glob.glob(tables_path + '/*_table.png'))
        tables_by_paper = {}
        for img_path in unordered_table_image_list:
            arxiv_prefix = os.path.basename(img_path).split('_')[0]            
            for pdf in pdf_paths:
                pdf_name = replace_delimiters(Path(pdf).stem)                
                if arxiv_prefix.lower() in pdf_name.lower(): 
                # if arxiv_prefix in Path(pdf).stem:       
                    if arxiv_prefix not in tables_by_paper:            
                        tables_by_paper[arxiv_prefix] = []                    
                    tables_by_paper[arxiv_prefix].append(img_path)

        logger.info('TQA questions generation')        
        for paper, images in tqdm.tqdm(tables_by_paper.items(), total=len(tables_by_paper), desc='Iterating over papers'):                            
            for img_path in tqdm.tqdm(images, total=len(images), desc='generating questions for tables', leave=False):                                    
                try: 
                    df = pd.read_csv(img_path.replace('_table.png', '.csv'), skiprows=1)
                    table_text = df.to_markdown(index=False)                
                    density = len(df) * len(df.columns)
                    if density < 60:
                        do_resize = True
                    else:
                        do_resize = False
                        
                    q_file = img_path.replace('_table.png', '_questions_table')
                    if not os.path.isfile(f'{q_file}.json'):                               
                        gpt_questions_json = []
                        q_file +='.txt'
                        # base64_image = encode_image(img_path, do_resize=do_resize)                                   
                        for q_type in ['data arithmetic', 'common sense reasoning']:
                            questions_prompt = get_table_questions_prompt_fn(table_text, n_questions=5, q_type=q_type)
                            if args.generate.freestyle:
                                struct_out=False
                            else:
                                struct_out=True                                                                              
                                
                            gpt_questions = api_call_fn(questions_prompt, img_path, do_resize=do_resize, struct_out=struct_out, params={'temperature': 0.2})
                            api_cnt += 1
                            try:
                                if generation_model == 'gpt':
                                    cur_json = json.loads(gpt_questions)['questions']
                                else:
                                    cur_json = json.loads(gpt_questions)
                                    
                                gpt_questions_json.extend(cur_json)
                            
                            except Exception as e:
                                logger.warning(e)
                                with open(q_file, 'a') as q_f:
                                    q_f.write(gpt_questions)
                                
                        if gpt_questions_json:       
                            q_file = q_file.replace('.txt', '.json')
                            with open(q_file, 'w') as q_f:
                                json.dump(gpt_questions_json, q_f, indent=4)    
                
                except Exception as APIe:
                    logger.warning(APIe)
                    continue                                                                                                        
            
        logger.info('TQA generation ends')

    logger.info(f'Total API calls: {api_cnt}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_results", type=int, default=10, help="Max results to scrape")
    parser.add_argument("--dpi", type=int, default=300, help="Figures DPI")
    parser.add_argument("--assests_type", choices=['all', 'figures', 'tables'], default='all')
    parser.add_argument("--artifacts_dir", type=str, default='./mm_live_bench/artifacts/', help="Artifacts dir")
    parser.add_argument("--do_download", action='store_true')
    parser.add_argument('--fig_type_to_exclude', nargs='+', help='Figure types to exclude from generation', default='')
    args = parser.parse_args()
    
    generate_raw_questions(args)