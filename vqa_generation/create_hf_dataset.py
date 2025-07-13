import datasets
from datasets import Dataset, concatenate_datasets, load_dataset
import os
import json
from PIL import Image
import tqdm
import numpy as np
import logging
from typing import List
import pyarrow.parquet as pq
from pathlib import Path
logger = logging.getLogger('livexiv')


def gen(qa_file, image_folder, add_image_path=False):
    if isinstance(qa_file, list):
        data = qa_file
    else:
        with open(qa_file, "r") as f:
            data = json.load(f)
        
        
    for q in tqdm.tqdm(data, total=len(data), desc='Generating HF Dataset'):        
        try:        
            ret_val = {}
            image_path = os.path.join(image_folder, q['image_path'])
            if add_image_path:
                ret_val['image_path'] = image_path
            ret_val["image"] = Image.open(image_path).convert("RGB")     
            ret_val['question'] = q['Q']
            ret_val['gt'] = q['gt']
            ret_val['option_a'] = q['options'][0]
            ret_val['option_b'] = q['options'][1]
            ret_val['option_c'] = q['options'][2]
            ret_val['option_d'] = q['options'][3]
            yield ret_val
        except Exception as e:
            logger.warning(e)


def create_dataset(image_folder: str, questions_file: str, save_path: str, samples_per_shard: int = 1000, add_image_path=False):
    os.makedirs(save_path, exist_ok=True)
    if add_image_path:
        print('add image path')
        features = datasets.Features(
            {            
                "question": datasets.Value("string"),            
                "image": datasets.Image(),
                'image_path': datasets.Value("string"),
                "gt": datasets.Value("string"),
                "option_a": datasets.Value("string"),
                "option_b": datasets.Value("string"),
                "option_c": datasets.Value("string"),
                "option_d": datasets.Value("string")
            }
        )
    else:
        features = datasets.Features(
            {            
                "question": datasets.Value("string"),            
                "image": datasets.Image(),
                # 'image_path': datasets.Value("string"),
                "gt": datasets.Value("string"),
                "option_a": datasets.Value("string"),
                "option_b": datasets.Value("string"),
                "option_c": datasets.Value("string"),
                "option_d": datasets.Value("string")
            }
        )
    
    with open(questions_file, 'r') as f:
        metadata = json.load(f)
        
    len_dataset = len(metadata)    
    num_shards = int(np.ceil(len_dataset / samples_per_shard))
    for i in range(num_shards):
        questions = metadata[i * samples_per_shard: (i + 1) * samples_per_shard]
        output_path_template = save_path + "/test-{index:05d}.parquet"
        data_test = datasets.Dataset.from_generator(
            gen,
            gen_kwargs={
                "qa_file": questions,
                "image_folder": image_folder,
                "add_image_path": add_image_path
            },
            features=features,
        )

        data = datasets.DatasetDict({"test": data_test})
        logger.info(data)
        data_test.to_parquet(output_path_template.format(index=i))

def merge_datasets(datasets_paths: List[str], save_path:str, samples_per_shard:int = 1000):
    os.makedirs(save_path, exist_ok=True)
    paths = [Path(data_path) for data_path in datasets_paths]    
        
    all_datasets = []
    for path in paths:
        for parquet_file in Path(path).glob("*.parquet"):
            try:
                pq.read_table(parquet_file)
            except Exception as e:
                print(f"Invalid file: {parquet_file}, Error: {e}")
        
        all_datasets.extend(
    [
        Dataset.from_parquet(str(parquet_file))
        for parquet_file in path.glob("*.parquet")
    ])
        
    merged_dataset = concatenate_datasets(all_datasets)
    print(f'Total: {len(merged_dataset)}')
    num_shards = int(np.ceil(len(merged_dataset) / samples_per_shard))
    for i in range(num_shards):
        output_path_template = save_path + "/test-{index:05d}.parquet"        
        shard = merged_dataset.shard(index=i, num_shards=num_shards, contiguous=True)
        shard.to_parquet(output_path_template.format(index=i))    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, help="'path to image folder", default='')
    parser.add_argument("--questions_file", type=str, help="'path to questions file", default='')
    parser.add_argument("--save_path", type=str, help="'path to save path", default='')
    parser.add_argument("--add_image_path", action='store_true', help="Include image path in the parquet files or not")
    parser.add_argument("--function", choices=['create_dataset', 'merge_datasets'], default='create_dataset')
    parser.add_argument("--dataset_path_1", type=str, help="'path to saved dataset", default='')
    parser.add_argument("--dataset_path_2", type=str, help="'path to saved dataset", default='')
    
    args = parser.parse_args()    
    if args.function == 'create_dataset':
        create_dataset(args.image_folder, args.questions_file, args.save_path, add_image_path=args.add_image_path)
    else:
        merge_datasets([args.dataset_path_1, args.dataset_path_2], args.save_path)
        ds = datasets.load_dataset(args.save_path)
        print(ds)
    print(f'HF dataset saved locally to {args.save_path}')
    