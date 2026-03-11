from torchvision import datasets, transforms
import config
import tools.strings as strings
import torch
import h5py
from torch.utils.data import DataLoader



def init_dataloaders(cfg):
    train_dataloader = build_dataloader(**cfg[strings.DATALOADERS_CFG][strings.TRAIN])
    val_dataloader = build_dataloader(**cfg[strings.DATALOADERS_CFG][strings.VAL])
    return train_dataloader, val_dataloader


def build_dataset(dataset_id, transform, *args, **kwargs):
    if dataset_id == 'MNIST':
        dataset = datasets.MNIST(root=config.DATASETS_PATH, transform=transform, *args, **kwargs)
    elif dataset_id == 'CIFAR10':
        dataset = datasets.CIFAR10(root=config.DATASETS_PATH, transform=transform, *args, **kwargs)
    elif dataset_id == 'IMAGENET1K':
        dataset = datasets.ImageNet(config.DATASETS_PATH / 'imagenet', transform=transform, *args, **kwargs)
    elif dataset_id == 'OMNIGLOT':
        dataset = datasets.Omniglot(root=config.DATASETS_PATH, transform=transform, *args, **kwargs, download=True)
    elif dataset_id == '3DSHAPES':
        dataset = build_3dshapes(transform, *args, **kwargs)
    elif dataset_id == 'POKEMON':
        dataset = build_pokemon(transform, *args, **kwargs)
    else:
        raise NameError(f'unknown dataset {dataset_id}')
    return dataset


def build_dataloader(dataset_cfg, dataloader_cfg, transform_cfg):
    transform = build_transform(transform_cfg)
    dataset = build_dataset(transform=transform, **dataset_cfg)
    dataloader = DataLoader(dataset=dataset, **dataloader_cfg)
    return dataloader


def build_pokemon(transform, train, *args, **kwargs):
    if train:
        dataset = datasets.ImageFolder(root=config.DATASETS_PATH / 'pokemon' / 'train', transform=transform)
    else:
        dataset = datasets.ImageFolder(root=config.DATASETS_PATH / 'pokemon' / 'val', transform=transform)
    return dataset

def build_3dshapes(transform, train, *args, **kwargs):
    images, labels = load_shapes3d_to_ram()
    g = torch.Generator().manual_seed(123)

    images = images[torch.randperm(images.size(0), generator=g)]
    labels = labels[torch.randperm(labels.size(0), generator=g)]
    train_size = int(0.95 * images.size(0))
    if train:
        return Shapes3D(transform=transform, images=images[0:train_size], labels=labels[0:train_size])
    else:
        return Shapes3D(transform=transform, images=images[train_size:], labels=labels[train_size:0])

def load_shapes3d_to_ram():
    with h5py.File(config.DATASETS_PATH / '3dshapes.h5', "r") as f:
        images = torch.from_numpy(f["images"][:])  # (480k, 64, 64, 3)
        labels = torch.from_numpy(f["labels"][:])

    # convert to channels-first float here
    images = images.permute(0, 3, 1, 2).float() / 255
    return images, labels.float()


def build_transform(transform_cfg):
    transform_list = []
    transform_keys =  sorted(list(transform_cfg.keys()), key=int)
    for key in transform_keys:
        transform_list.append(create_transform(**transform_cfg[key]))
    return transforms.Compose(transform_list)

def create_transform(class_name, *args, **kwargs):
    transform_cls = getattr(transforms, class_name)
    transform = transform_cls(*args, **kwargs)
    return transform


class Shapes3D(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]




if __name__ == '__main__':
    build_pokemon(None, True)