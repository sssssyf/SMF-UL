
from torchvision import transforms
from transforms.co_transforms import get_co_transforms
from transforms import sep_transforms
from datasets.flow_datasets import HSI_Raw



def get_dataset(all_cfg):
    cfg = all_cfg.data

    input_transform = transforms.Compose([
        sep_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
    ])

    co_transform = get_co_transforms(aug_args=all_cfg.data_aug)



    if cfg.type == 'HSI_Raw':
        train_set = HSI_Raw(cfg.root_sintel_raw, n_frames=cfg.train_n_frames,
                              transform=input_transform, co_transform=co_transform)
        valid_set=[]


    else:
        raise NotImplementedError(cfg.type)
    return train_set, valid_set