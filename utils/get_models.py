import os
import timm
import yaml
import torch
from torchvision import transforms
from utils.AverageMeter import AccuracyMeter
# checkpoint yaml file
yaml_path = '../configs/checkpoint.yaml'
adv_yaml_path = 'checkpoint_adv.yaml'
def get_models(args, device):
    metrix = {}
    with open(os.path.join(args.root_path, 'utils', yaml_path), 'r', encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    print('🌟\tBuilding models...')
    models = {}
    if args.dataset == 'imagenet_compatible':
        for key, value in yaml_data.items():
            models[key] = timm.create_model(value['full_name'], pretrained=True, num_classes=1000).to(device)
            models[key].eval()
            if 'inc' in key or 'vit' in key or 'bit' in key:
                models[key] = torch.nn.Sequential(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), models[key])
            else:
                models[key] = torch.nn.Sequential(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                models[key])
            metrix[key] = AccuracyMeter()
            print(f'⭐\tload {key} successfully')
    else:
        raise NotImplemented
    return models, metrix

