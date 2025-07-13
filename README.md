# LiveXiv - A Multi-Modal Live Benchmark Based on Arxiv Papers Content

Welcome to our GitHub repository! This repository is based on the ideas introduced in

[Shabtay, Nimrod, Felipe Maia Polo, Sivan Doveh, Wei Lin, M. Jehanzeb Mirza, Leshem Chosen, Mikhail Yurochkin et al. "LiveXiv--A Multi-Modal Live Benchmark Based on Arxiv Papers Content." arXiv preprint arXiv:2410.10783 (2024).](https://arxiv.org/abs/2410.10783)

## Installation

To use the code in this repository, clone the repo and create a conda environment using:

```
conda env create --file=environment.yaml
conda activate sloth
```
## Data

Our data can be found on [HuggingFace](https://huggingface.co/datasets/LiveXiv/LiveXiv).

## VQA Generation
Go to `vqa_generation`
```
python main.py artifacts_dir=<artifacts_dir>
```
Comments:
* There are many configuration options inside `vqa_generation/confing/conf.yaml` so be sure to check it before you start.
* Api-keys for GPT and/or claude are needed to run the code.
* Additional installation is required - `pip install vqa_generation/requirements.txt`
* You will need to clone and install the llava repo to work with the stand-alone blind filtering (you can switch to any other model with minimal changes)

## Efficient evaluation

###  Quick start

If you are interested in checking how our efficient eval method works in practice, please check [this notebook](https://github.com/NimrodShabtay/LiveXiv/blob/main/notebooks/efficient_eval_demo.ipynb).


### Reproducing results from the paper

Please check our [notebooks](https://github.com/NimrodShabtay/LiveXiv/tree/main/notebooks).


## Citing

If you find LiveXiv useful for your research and applications, please cite using this BibTeX:
```
@misc{shabtay2024livexivmultimodallive,
      title={LiveXiv -- A Multi-Modal Live Benchmark Based on Arxiv Papers Content}, 
      author={Nimrod Shabtay and Felipe Maia Polo and Sivan Doveh and Wei Lin and M. Jehanzeb Mirza and Leshem Chosen and Mikhail Yurochkin and Yuekai Sun and Assaf Arbelle and Leonid Karlinsky and Raja Giryes},
      year={2024},
      eprint={2410.10783},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.10783}, 
}
```

