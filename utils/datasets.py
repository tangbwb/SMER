import csv
from os.path import join
from torch.utils.data import Dataset
from PIL import Image

class LoadDataset(Dataset):
    def __init__(self, image_root: str, info_dir: str, transform = None) -> None:
        super(LoadDataset).__init__()
        self.image_root = image_root
        self.info_dir = info_dir
        self.image_name = []
        self.image_true_label = []
        self.transform = transform
        self.image_target_label = []
        with open(self.info_dir) as info_csv:
            reader = csv.reader(info_csv)
            next(reader)
            for i in reader:
                self.image_name.append(i[0])
                self.image_true_label.append(int(i[6]))
                self.image_target_label.append(int(i[7]))

    def __getitem__(self, index: int) -> any:
        image = Image.open(join(self.image_root, self.image_name[index]+'.png')).convert('RGB')
        true_label= self.image_true_label[index]
        target_label= self.image_target_label[index]

        if self.transform:
            image = self.transform(image)
        return  image, true_label-1,target_label-1
    
    def __len__(self) -> int:
        return len(self.image_name)