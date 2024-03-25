import torch
from torchvision import transforms
from utils.datasets import LoadDataset
def get_dataset(args):
    if args.dataset == 'imagenet_compatible':
        setattr(args, 'num_classes', 1000)
        transform_test = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size), antialias=True),
            transforms.ToTensor(),
        ])
        test_set=LoadDataset(image_root=args.image_dir, info_dir=args.image_info, transform = transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                                  num_workers=args.num_worker)
    else:
        raise NotImplemented

    return test_loader
