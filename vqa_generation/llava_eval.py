import torch
from PIL import Image
import sys
import os
import argparse
import tqdm
import pandas as pd
import contextlib

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from evaluation_utils import construct_questions, try_to_fix_preds



# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode, remove_img_token):

        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode       
        self.remove_image_token = remove_img_token  

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image_path"]
        qs = line["Q"]                

        if not self.remove_image_token:            
            if self.model_config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

         
        options = line['options']
        options_str = "\n".join([f'{a}. {b}' for a, b in zip(['A', 'B', 'C', 'D'], options)])
        prompt_raw = f"{qs}\n{options_str}\nAnswer with the option's letter from the given choices directly."
        conv = conv_templates[self.conv_mode].copy()

        conv.append_message(conv.roles[0], prompt_raw)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size


    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, conv_mode, remove_img_token, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, conv_mode, remove_img_token)

    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_llava(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_name)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    questions = construct_questions(args.artifacts_dir, args.unfiltered_results_name, q_type='figure', gen_model='')[0]    
    
    if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:            
        args.conv_mode = conv_mode
            
    data_loader = create_data_loader(questions, args.artifacts_dir, tokenizer, image_processor, model.config, args.conv_mode, args.remove_image_token)
        
    answers = []
    
    for (input_ids, image_tensor, image_sizes), raw_data in tqdm.tqdm(zip(data_loader, questions), total=len(questions)):        
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        img_t = None if args.remove_image_token else image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)
        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=img_t, 
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

            llava_ans = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()        
            answers.append([raw_data['Q'], raw_data["image_path"], raw_data["gt"], llava_ans])
        except Exception as e:
            print(e)            
                
    df_answers = pd.DataFrame.from_records(data=answers, columns=['Question', 'img_filename', 'GT', 'LLaVA-Pred'])
    # Try to fix text results to only the option's letter
    df_answers['LLaVA-Pred-Fixed'] = df_answers['LLaVA-Pred'].apply(try_to_fix_preds)
    
    # Calculate Accuracy
    df_answers['acc'] = df_answers['LLaVA-Pred'] == df_answers['GT']    
    df_answers['acc-fixed'] = df_answers['LLaVA-Pred-Fixed'] == df_answers['GT']
    print(f'Accuracy: {df_answers["acc"].mean()}')
    print(f'Accuracy after correction: {df_answers["acc-fixed"].mean()}')
    
    df_answers.to_csv(args.results_name)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="liuhaotian/llava-v1.6-34b", help="Model name in HF format")
    parser.add_argument("--artifacts_dir", type=str, default='./mm_live_bench/artifacts', help="Artifacts dir")
    parser.add_argument("--unfiltered_results_name", type=str, default='./mm_live_bench/unfiltered_questions_for_eval.json', help="Results filename")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--results_name", type=str, default='llava_1_6_34B_results.csv', help="Results filename")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--remove_image_token", action='store_true')

    args = parser.parse_args()
    
    eval_llava(args) 
