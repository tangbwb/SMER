# Ensemble Diversity Facilitates Adversarial Transferability
We provide an example on MI-FGSM of SMER method.
## Datasets And models
To run the code, you should download pre-trained models and the datasets. 

You need to access the pretrained [model](https://huggingface.co/).

Please unzip the data and place the [data](https://www.kaggle.com/google-brain/nips-2017-adversarial-learning-development-set) to your directory.

## Requirements
 We implement the experiments on NVIDIA A6000 GPU:

 - pytorch = 2.0.1
 - torchvision = 0.15.2
 - pillow = 10.0.1
 - timm
 - yaml
 - tqdm


## File Description
- 'release.py': Main file to call attack methods, generate adversarial examples and evaluate.

## Experiments
It is recommended you to conduct the experiments as following:

'''
python release.py --image-dir YOUR_IMAGE_DIRECTORY --image-info YOUR_IMAGE_CSV
'''

## Acknowledgements
We appreciate the contribution in [AdaEA](https://github.com/CHENBIN99/AdaEA).  
Please refer to our work as following:  
@inproceedings{  
	title = {Ensemble Diversity Facilitates Adversarial Transferability},  
	author = {Bowen, Tang and Wang, Zheng and Yi, Bin and Qi, Dou and Yang, Yang and Shen, Heng Tao},  
	booktitle = The IEEE/CVF Conference on Computer Vision and Pattern Recognition 2024,  
	year = {2024}  
	}
