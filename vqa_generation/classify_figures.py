from transformers import AutoProcessor, AutoTokenizer, CLIPModel
from PIL import Image
import torch
import itertools
import tqdm
import argparse
import pandas as pd
import os
from typing import Union
import glob
import pickle


class FiguresCls:
    def __init__(self, sim_th: float = 0.15, device='cuda'):
        self.clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.sim_th = sim_th
        self.classes = ['block diagram', 'chart', 'qualitative']
        self.text_queries = self.create_templates()
        self.device = device
        
    def create_templates(self):
        self.templates = ['a photo of a', 'A figure of a']
        permutations = itertools.product(self.templates, self.classes)
        concatenated_permutations = [' '.join(item) for item in permutations]
        return concatenated_permutations

    def calc_sim(self, txt_ftrs, img_ftrs):
        image_features = img_ftrs / img_ftrs.norm(p=2, dim=-1, keepdim=True)
        text_features = txt_ftrs / txt_ftrs.norm(p=2, dim=-1, keepdim=True)
        res = torch.matmul(text_features, image_features.t())
        return res
    
    def run_single(self, image):        
        inputs = self.clip_tokenizer(self.text_queries, padding=True, return_tensors="pt").to(self.device)
        text_features = self.clip_model.get_text_features(**inputs)
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        image_features = self.clip_model.get_image_features(**inputs)
        similiraties = self.calc_sim(text_features, image_features)
        
        if similiraties.max() > self.sim_th:
            chosenid = similiraties.argmax()                
            pred_cls = self.classes[chosenid % len(self.classes)]
        else:
            pred_cls = 'Other'
        
        # print([round(s[0], 3) for s in similiraties.tolist()], pred_cls)
        return pred_cls

    def run_single_meta_prompt(self, image, device):
        import json
        with open('./descriptions.json', 'r') as desc_f:
            desc_data = json.load(desc_f)
            
        with torch.no_grad():
            zeroshot_weights = []
            for classname in self.classes:
                queries = []
                for v in desc_data[classname].values():
                    queries.extend(v)
                inputs = self.clip_tokenizer(queries, padding=True, return_tensors="pt")
                for k in inputs.keys():
                    inputs[k] = inputs[k].to(device)
                class_embeddings = self.clip_model.get_text_features(**inputs)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)  
        inputs = self.clip_processor(images=image, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].to(device)
        image_features = self.clip_model.get_image_features(**inputs)
        
        out = image_features.float().to(device) @ zeroshot_weights.float()
        out_norm = out / out.norm(dim=-1, keepdim=True)

        chosenid = torch.argmax(out_norm, dim=1)        
        pred_cls = self.classes[chosenid % len(self.classes)]
        
        # print([round(s, 3) for s in out_norm.tolist()[0]], pred_cls)
        return pred_cls
        
    def run_dataset(self, list_of_paths):
        predictions = {}
        for ii, img_path in tqdm.tqdm(enumerate(list_of_paths), total=len(list_of_paths)):
            cur_img = Image.open(img_path)
            cur_pred = self.run_single_meta_prompt(cur_img, 'cpu')
            predictions[img_path] = cur_pred
        return predictions
    
    
def classify_figure_type(artifacts_dir:str, df:pd.DataFrame=None, do_plot:bool=False, device: str = 'cuda') -> Union[pd.DataFrame, None]:
    res_filepath = f'{artifacts_dir}/type_list.pkl'                                
        
    figure_classifier = FiguresCls(device=device)    
    if df is not None:
        take_from_type_list = False
        if os.path.isfile(os.path.join(res_filepath)):
            with open(res_filepath, 'rb') as f:
                type_list = pickle.load(f)
                take_from_type_list = True
        for idx, row in tqdm.tqdm(df.iterrows(), total=len(df), desc='Classify Figures'):
            img_path = os.path.join(artifacts_dir, row['image_path'])
            img_ = Image.open(img_path)
            pred_cls = type_list[img_path] if take_from_type_list else figure_classifier.run_single_meta_prompt(img_, device)
            df.loc[idx, 'figure_type'] = pred_cls
        
        return df
    else:
        files_list = glob.glob(f'{artifacts_dir}/*.png')
        if do_plot:
            plot_dir = f'{args.artifacts_dir}/plots'
            os.makedirs(plot_dir, exist_ok=True)
            
        type_list = {}
        for i in tqdm.tqdm(range(len(files_list)), total=len(files_list)):
            sample_idx = i
            img_ = Image.open(files_list[sample_idx])
            pred_cls = figure_classifier.run_single_meta_prompt(img_, device)
            type_list[files_list[sample_idx]] = pred_cls
            
            if do_plot:
                plt.figure()
                plt.imshow(np.array(img_))
                plt.title(pred_cls)
                plt.savefig(f'{plot_dir}/test_cls_{i}_meta.png')
                plt.close()
                
        with open(res_filepath, 'wb') as save_f:
            pickle.dump(type_list, save_f, pickle.HIGHEST_PROTOCOL)
            
        return type_list
    
    
if __name__ == "__main__":
    import glob
    import matplotlib.pyplot as plt
    import numpy as np    
    import pickle
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts_dir", type=str, default='./mm_live_bench/artifacts/', help="Artifacts dir")
    parser.add_argument("--do_plot", action='store_true')
    
    args = parser.parse_args()
    classify_figure_type(args.artifacts_dir, None)    
    
    