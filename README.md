# Emotion Recognition in User-generated Videos with Long-Range Correlation-Aware Network

## Installation 
The codes are based on VideoMAE V2. We express our respect for their outstanding work. To prepare the environment, please follow the following instructions.
```
## install pytorch
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia

## install other requirements
pip install -r requirements.txt
```

## Datasets
The used datasets are provided in [VideoEmotion-8](https://drive.google.com/drive/folders/0B5peJ1MHnIWGd3pFbzMyTG5BSGs?resourcekey=0-hZ1jo5t1hIauRpYhYIvWYA) and [Ekman-6](https://github.com/kittenish/Frame-Transformer-Network). The train/test splits in both two datasets follow the official procedure. To prepare the data, you can refer to VideoMAE V2 for a general guideline.

## Model
We now provide the model weights in the following [link](https://pan.baidu.com/s/1LjO4nqA0z4qMD-CvVtjAsw?pwd=CHOW).

## Eval
You can easily evaluate the model by running the script below. 
```
bash eval.sh
```
To get the fusion result, please run the test.py.

Please email to jinchow@gnnu.edu.cn if you are interested in our work. We would be fairly pleased if you find the code useful and cite the following paper.
```
@article{yi2024emotion,
  title={Emotion recognition in user-generated videos with long-range correlation-aware network},
  author={Yi, Yun and Zhou, Jin and Wang, Hanli and Tang, Pengjie and Wang, Min},
  journal={IET Image Processing},
  year={2024},
  publisher={Wiley Online Library}
}
```
