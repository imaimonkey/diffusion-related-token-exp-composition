# Saber: An Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for Diffusion Language Model
[![arXiv](https://img.shields.io/badge/arXiv-2510.18165-b31b1b.svg)](https://arxiv.org/abs/2510.18165)

Our work introduces Saber, a training-free sampling algorithm for diffusion language models that enhances code generation by adaptively accelerating inference and incorporating backtracking, thereby improving output quality and speed while narrowing the performance gap with autoregressive models.

## Installation environment
```shell
conda create -n saber python=3.11
pip install -r requirements.txt
```
or
```shell
uv venv saber --python 3.11
source saber/bin/activate
uv pip install -r requirements.txt
```

## Evaluation of Saber
Firstly, make sure you have downloaded the model. Your model path should match the file in configs.
Secondly, We provided the humaneval, mbpp, humaneval et, and mbpp et datasets used in our experiment. Due to GitHub's file size restrictions, if you wish to test livecodebench, you will need to download the dataset yourself. And ensure that it is on the correct path.
Finally, execute the following command to evaluate.
```shell
python eval.py --config ./configs/humaneval.yaml
```
If you want to test other methods, change the method in the yaml file.
For the humaneval and MBPP datasets, our code will print pass@1 And steps. For the livecodebench dataset, our code will save the generated results, and you need to run the evaluation program yourself

## Citation
```
@article{dong2025saber,
  title={Saber: An Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for Diffusion Language Model},
  author={Yihong Dong and Zhaoyu Ma and Xue Jiang and Zhiyuan Fan and Jiaru Qian and Yongmin Li and Jianha Xiao and Zhi Jin and Rongyu Cao and Binhua Li and Fei Huang and Yongbin Li and Ge Li},
  journal={arXiv preprint arXiv:2510.18165},
  year={2025}
}
```