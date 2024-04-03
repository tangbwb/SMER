import torch
from tqdm import tqdm
from utils.get_dataset import get_dataset
from utils.get_models import get_models
from utils.tools import same_seeds, get_project_path
import argparse
from attack_method import MI_FGSM_SMER
def get_args():
    parser = argparse.ArgumentParser(description='benchmark of cifar10')
    parser.add_argument('--seed', type=int, default=1789)
    parser.add_argument('--dataset', type=str, default='imagenet_compatible',help='imagenet_compatible')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='the batch size when training')
    parser.add_argument('--image-size', type=int, default=224,
                        help='image size of the dataloader')
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--attack_method', type=str,default='MI_FGSM_SMER')
    parser.add_argument('--image-dir', type=str,)
    parser.add_argument('--image-info', type=str,
                        default='')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='gpu_id')
    # attack parameters
    parser.add_argument('--eps', type=float, default=16.0)
    parser.add_argument('--alpha', type=float, default=1.6)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--momentum', type=float, default=1.0,
                        help='default momentum value')
    parser.add_argument('--beta', type=float, default=10)

    args = parser.parse_args()
    return args


def main(args):
    cur_attack={'MI_FGSM_SMER':MI_FGSM_SMER}
    device = torch.device(f'cuda:{args.gpu_id}')
    # dataset
    dataloader = get_dataset(args)
    # models
    models, metrix = get_models(args, device=device)
    ens_model = ['vit_t','deit_t', 'resnet18','inc_v3']
    print(f'ens model: {ens_model}')
    ens_models=[models[i] for i in ens_model]
    for idx, (data,label,_) in enumerate(tqdm(dataloader)):
        n = label.size(0)
        data, label = data.to(device), label.to(device)
        adv_exp = cur_attack[args.attack_method](ens_models,data, label,args=args)
        # adv_exp=data
        for model_name, model in models.items():
            with torch.no_grad():
                r_clean = model(data)
                r_adv = model(adv_exp)
            # clean
            pred_clean = r_clean.max(1)[1]
            correct_clean = (pred_clean == label).sum().item()
            # adv
            pred_adv = r_adv.max(1)[1]
            correct_adv = (pred_adv == label).sum().item()
            metrix[model_name].update(correct_clean, correct_adv, n)
    print('-' * 73)
    print('|\tModel name\t|\tNat. Acc. (%)\t|\tAdv. Acc. (%)\t|\tASR. (%)\t|')
    for model_name, _ in models.items():
        print(f"|\t{model_name.ljust(10, ' ')}\t"
              f"|\t{str(round(metrix[model_name].clean_acc * 100, 2)).ljust(13, ' ')}\t"
              f"|\t{str(round(metrix[model_name].adv_acc * 100, 2)).ljust(13, ' ')}\t"
              f"|\t{str(round(metrix[model_name].attack_rate * 100, 2)).ljust(8, ' ')}\t|")
    print('-' * 73)


if __name__ == '__main__':
    args = get_args()
    same_seeds(args.seed)
    root_path = get_project_path()
    setattr(args, 'root_path', root_path)
    main(args)
