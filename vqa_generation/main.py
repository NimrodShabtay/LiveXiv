from generate_raw_questions import generate_raw_questions 
from filter_utils import *
from create_hf_dataset import create_dataset
from extract_data_from_papers import extract_papers_and_data_for_generation
from evaluation_utils import *
import torch
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import logging


@hydra.main(version_base=None, config_path="conf", config_name="config")
def live_xiv(args : DictConfig) -> None:    
    os.makedirs(args.artifacts_dir, exist_ok=True)
    with open(os.path.join(args.artifacts_dir, 'conf.yaml'), 'w') as conf_f:
        OmegaConf.save(config=args, f=conf_f)    
    
    # Setup logger
    logger = logging.getLogger('livexiv')
    logger.setLevel(logging.DEBUG)
    log_file = os.path.join(args.artifacts_dir, 'livexiv.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # You can set different levels for console (INFO here)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)       
    logger.addHandler(console_handler) 
        
    # Generation        
    # ----------
    if args.flow.fetch_data:
        extract_papers_and_data_for_generation(args)
    if args.flow.generate:        
        generate_raw_questions(args)        
    if args.assests_type in ['all', 'figures']:
        fig_artifacts_dir = os.path.join(args.artifacts_dir, 'figures')
        args.filter_results_fig_name = os.path.join(fig_artifacts_dir, args.filter_results_name.replace('.json', '_fig.json'))
        args.unfiltered_results_fig_name = os.path.join(fig_artifacts_dir, args.unfiltered_results_name.replace('.json', '_fig.json'))
        if args.flow.collect:
            construct_questions(fig_artifacts_dir, questions_filname=args.unfiltered_results_fig_name, q_type='figure', gen_model=args.models.generation)
    if args.assests_type in ['all', 'tables']:
        table_artifacts_dir = os.path.join(args.artifacts_dir, 'tables')
        args.filter_results_table_name = os.path.join(table_artifacts_dir, args.filter_results_name.replace('.json', '_table.json'))
        args.unfiltered_results_table_name = os.path.join(table_artifacts_dir, args.unfiltered_results_name.replace('.json', '_table.json'))
        if args.flow.collect:
            construct_questions(table_artifacts_dir, questions_filname=args.unfiltered_results_table_name, q_type='table', gen_model=args.models.generation)

    # Filtering    
    # ------------------------------------
    
    if args.flow.filtering:        
        # Blind check filtering (Figures Only)
        import argparse
        conf_dict = OmegaConf.to_container(args, resolve=True)
        args_namespace = argparse.Namespace(**conf_dict)
        args_namespace.remove_image_token = True        
        if args.assests_type in ['all', 'figures']:
            args_namespace.unfiltered_results_name = args.unfiltered_results_fig_name
            args_namespace.artifacts_dir = fig_artifacts_dir
            results_template = args_namespace.filter_results_fig_name.replace('.csv', '_{}.csv').replace('filter', 'blind_filter')
            n_iter = args.filtering.blind_n_iter
            # llava args:
            args_namespace.conv_mode = None
            args_namespace.temperature = 0.2
            args_namespace.top_p = None
            args_namespace.num_beams = 1
            args_namespace.max_new_tokens = 128
            args_namespace.model = args.filtering.model
            
            for i in range(n_iter):
                args_namespace.filter_results_name = results_template.format(i)
                filter_based_on_llm(args_namespace)
                torch.cuda.empty_cache()
                try:
                    torch.distributed.destroy_process_group()
                except:
                    continue
                
            # merge filter results
            csvs_filters = sorted(glob.glob(f'{fig_artifacts_dir}/blind_filter_*.csv'))
            dfs = [pd.read_csv(csv_f) for csv_f in csvs_filters]
            questions_to_filter_after_blind_check = find_different_rows(*dfs)
            args.filter_results_fig_name = args.filter_results_fig_name.replace(f'_{n_iter-1}.csv', '_fig.json').replace('blind_filter', 'filter')
            
            if args.filter_results_fig_name.endswith('.csv'):
                args.filter_results_fig_name = args.filter_results_fig_name.replace('.csv', '.json')
            merge_and_remove_questions(args.unfiltered_results_fig_name, 
                                       questions_to_filter_after_blind_check, 
                                       figure_type_to_exclude=args.filtering.fig_type_to_exclude, 
                                       save_path=args.filter_results_fig_name, 
                                       artifacts_dir=fig_artifacts_dir, 
                                       do_agreement=args.filtering.agreement, 
                                       filter_model=args.models.filter)            
            logger.info(f'Filterred results file VQA: {args.filter_results_fig_name}')
            
        if args.assests_type in ['all', 'tables']:
            if args.filter_results_table_name.endswith('.csv'):
                args.filter_results_table_name = args.filter_results_table_name.replace('.csv', '.json')
            remove_questions(args.unfiltered_results_table_name, 
                            save_path=args.filter_results_table_name, 
                            artifacts_dir=table_artifacts_dir, 
                            do_agreement=args.filtering.agreement, 
                            filter_model=args.models.filter)            
            logger.info(f'Filterred results file TableQA: {args.filter_results_table_name}')
        
    # Create HF datasets
    if args.flow.hf_dataset:      
        if args.filter_results_fig_name.endswith('.csv'):
                args.filter_results_fig_name = args.filter_results_fig_name.replace('.csv', '.json')  
        if args.assests_type in ['all', 'figures']:
            # save_path = os.path.join(fig_artifacts_dir, 'hf_dataset')
            # os.makedirs(save_path, exist_ok=True)
            # create_dataset(fig_artifacts_dir, args.filter_results_fig_name, save_path)
            
            save_path = os.path.join(fig_artifacts_dir, 'hf_dataset_img_path')
            os.makedirs(save_path, exist_ok=True)
            create_dataset(fig_artifacts_dir, args.filter_results_fig_name, save_path, add_image_path=True)
            
        if args.assests_type in ['all', 'tables']:
            if args.filter_results_table_name.endswith('.csv'):
                args.filter_results_table_name = args.filter_results_table_name.replace('.csv', '.json')
            save_path = os.path.join(table_artifacts_dir, 'hf_dataset')
            os.makedirs(save_path, exist_ok=True)
            create_dataset(table_artifacts_dir, args.filter_results_table_name, save_path)
            
            save_path = os.path.join(table_artifacts_dir, 'hf_dataset_img_path')
            os.makedirs(save_path, exist_ok=True)
            create_dataset(table_artifacts_dir, args.filter_results_table_name, save_path, add_image_path=True)


if __name__ == "__main__":
    live_xiv()