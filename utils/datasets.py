# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LSGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

"""Code for getting the data loaders."""

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils.lmdb_datasets import LMDBDataset
from thirdparty.lsun import LSUN, LSUNClass
from thirdparty.mnist_sdf import MNISTSDFDataset
from thirdparty.afhq.dataset import ImageFolderDataset
import os
import urllib
from scipy.io import loadmat
from torch.utils.data import Dataset
from PIL import Image
from torch._utils import _accumulate
from torchvision.transforms.functional import InterpolationMode


class Binarize(object):
    """ This class introduces a binarization transformation
    """
    def __call__(self, pic):
        return torch.Tensor(pic.size()).bernoulli_(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CropCelebA64(object):
    """ This class applies cropping for CelebA64. This is a simplified implementation of:
    https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
    """
    def __call__(self, pic):
        new_pic = pic.crop((15, 40, 178 - 15, 218 - 30)) # PIL Image object
        return new_pic

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_interpolation_mode(interpolation='bilinear'):
    if interpolation == 'bilinear':
        return InterpolationMode.BILINEAR
    elif interpolation == 'bicubic':
        return InterpolationMode.BICUBIC
    else:
        raise NotImplementedError

def get_antialias(antialias):
    if type(antialias) == str:
        return antialias == 'true'
    elif type(antialias) == bool:
        return antialias
    else:
        raise NotImplementedError

def get_loaders(args, eval_mode=False, num_workers=8):
    """Get data loaders for required dataset."""
    return get_loaders_eval(args.dataset,
                            args.data,
                            args.distributed,
                            args.batch_size_per_gpu if not eval_mode else args.eval_batch_size_per_gpu,
                            centered=args.centered if hasattr(args, 'centered') else False,
                            super_resolution=args.super_resolution if hasattr(args, 'super_resolution') else False,
                            interpolation=get_interpolation_mode(args.interpolation) if hasattr(args, 'interpolation') else InterpolationMode.BILINEAR,
                            antialias=get_antialias(args.antialias) if hasattr(args, 'antialias') else None,
                            num_workers=num_workers,
                            )


def download_omniglot(data_dir):
    filename = 'chardata.mat'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    url = 'https://raw.github.com/yburda/iwae/master/datasets/OMNIGLOT/chardata.mat'

    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(url, filepath)
        print('Downloaded', filename)

    return


def load_omniglot(data_dir):
    download_omniglot(data_dir)

    data_path = os.path.join(data_dir, 'chardata.mat')

    omni = loadmat(data_path)
    train_data = 255 * omni['data'].astype('float32').reshape((28, 28, -1)).transpose((2, 1, 0))
    test_data = 255 * omni['testdata'].astype('float32').reshape((28, 28, -1)).transpose((2, 1, 0))

    train_data = train_data.astype('uint8')
    test_data = test_data.astype('uint8')

    return train_data, test_data


class OMNIGLOT(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        d = self.data[index]
        img = Image.fromarray(d)
        return self.transform(img), 0     # return zero as label.

    def __len__(self):
        return len(self.data)


class TensorDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index: int):
        sample = self.data[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, 0

    def __len__(self):
        return self.data.shape[0]


def get_datasets(dataset, root, augment=True, drop_last_train=True, shuffle_train=True,
                     binarize_binary_datasets=True, centered=False, super_resolution=False, interpolation=InterpolationMode.BILINEAR, antialias=None):
    if dataset == 'cifar10':
        num_classes = 10
        image_size = 32
        num_channels = 3
        train_transform, valid_transform = _data_transforms_cifar10(centered=centered)
        train_transform = train_transform if augment else valid_transform
        train_data = dset.CIFAR10(
            root=os.path.join(root, 'cifar10'), train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(
            root=os.path.join(root, 'cifar10'), train=False, download=True, transform=valid_transform)
    elif dataset == 'mnist':
        image_size = 32
        num_classes = 10
        num_channels = 1
        train_transform, valid_transform = _data_transforms_mnist(binarize_binary_datasets, centered=centered)
        train_transform = train_transform if augment else valid_transform
        train_data = dset.MNIST(
            root=root, train=True, download=True, transform=train_transform)
        valid_data = dset.MNIST(
            root=root, train=False, download=True, transform=valid_transform)
    elif dataset.startswith('mnistsdf'):
        if dataset.startswith('mnistsdfimg'):
            return_image = True
        else:
            return_image = False
        image_size = int(dataset.split('_')[1])
        num_classes = None
        num_channels = 1
        train_data = MNISTSDFDataset(
            root, train=True, download=True, size=(image_size, image_size), interpolation='bilinear', antialias=None, return_image=return_image)
        valid_data = MNISTSDFDataset(
            root, train=False, download=True, size=(image_size, image_size), interpolation='bilinear', antialias=None, return_image=return_image)
        #total_examples = len(train_data)
        #train_size = int(0.9 * total_examples)   # use 90% for training
        #train_data, valid_data = random_split_dataset(train_data, [train_size, total_examples - train_size])
    elif dataset == 'omniglot':
        raise NotImplementedError('Image size is not specified')
        image_size = None
        num_classes = 0
        num_channels = 1
        download_omniglot(root)
        train_transform, valid_transform = _data_transforms_mnist(binarize_binary_datasets, centered=centered)
        train_transform = train_transform if augment else valid_transform
        train_data, valid_data = load_omniglot(root)
        train_data = OMNIGLOT(train_data, train_transform)
        valid_data = OMNIGLOT(valid_data, valid_transform)
    elif dataset.startswith('vorticity'):
        vor3 = torch.load(os.path.join(root, 'vorticity', 'vorticity3.pt'))['vorticity']
        vor3 = vor3[:,1000:5001:10, :, :]
        vor3 = vor3.reshape(4010, 256, 256)
        vor4 = torch.load(os.path.join(root, 'vorticity', 'vorticity4.pt'))['vorticity']
        vor4 = vor4[:,1000:5001:10, :, :]
        vor4 = vor4.reshape(4010, 256, 256)
        vor = torch.cat([vor3, vor4], dim=0)[:,None]
        labels = torch.zeros(vor.shape[0]).long()
        num_classes = 1
        num_channels = 1
        if dataset == 'vorticity':
            image_size = 256
            train_transform, valid_transform = None, None
        else:
            image_size = int(dataset.split('_')[1])
            train_transform, valid_transform = _data_transforms_vorticity(image_size)
        train_data = TensorDataset(vor, transform=train_transform)
        valid_data = TensorDataset(vor, transform=valid_transform)
    elif dataset.startswith('celeba'):
        if dataset == 'celeba_64':
            image_size = 64
            num_classes = 40
            num_channels = 3
            train_transform, valid_transform = _data_transforms_celeba64(image_size, centered=centered, interpolation=interpolation, antialias=antialias)
            train_transform = train_transform if augment else valid_transform
            train_data = LMDBDataset(root=root, name='celeba64', train=True, transform=train_transform, is_encoded=True)
            valid_data = LMDBDataset(root=root, name='celeba64', train=False, transform=valid_transform, is_encoded=True)
        elif dataset in {'celeba_256'}:
            image_size = int(dataset.split('_')[1])
            num_classes = 1
            num_channels = 3
            train_transform, valid_transform = _data_transforms_generic(image_size, centered=centered, interpolation=interpolation, antialias=antialias)
            train_transform = train_transform if augment else valid_transform
            train_data = LMDBDataset(root=root, name='celeba', train=True, transform=train_transform)
            valid_data = LMDBDataset(root=root, name='celeba', train=False, transform=valid_transform)
        else:
            raise NotImplementedError
    elif dataset.startswith('lsun'):
        if dataset.startswith('lsun_bedroom'):
            image_size = int(dataset.split('_')[-1])
            num_classes = 1
            num_channels = 3
            train_transform, valid_transform = _data_transforms_lsun(image_size, centered=centered, interpolation=interpolation, antialias=antialias)
            train_transform = train_transform if augment else valid_transform
            train_data = LSUN(root=root, classes=['bedroom_train'], transform=train_transform)
            valid_data = LSUN(root=root, classes=['bedroom_val'], transform=valid_transform)
        elif dataset.startswith('lsun_church'):
            image_size = int(dataset.split('_')[-1])
            num_classes = 1
            num_channels = 3
            train_transform, valid_transform = _data_transforms_lsun(image_size, centered=centered, interpolation=interpolation, antialias=antialias)
            train_transform = train_transform if augment else valid_transform
            train_data = LSUN(root=root, classes=['church_outdoor_train'], transform=train_transform)
            valid_data = LSUN(root=root, classes=['church_outdoor_val'], transform=valid_transform)
        elif dataset.startswith('lsun_tower'):
            image_size = int(dataset.split('_')[-1])
            num_classes = 1
            num_channels = 3
            train_transform, valid_transform = _data_transforms_lsun(image_size, centered=centered, interpolation=interpolation, antialias=antialias)
            train_transform = train_transform if augment else valid_transform
            train_data = LSUN(root=root, classes=['tower_train'], transform=train_transform)
            valid_data = LSUN(root=root, classes=['tower_val'], transform=valid_transform)
        elif dataset.startswith('lsun_cat'):
            image_size = int(dataset.split('_')[-1])
            num_classes = 1
            num_channels = 3
            train_transform, valid_transform = _data_transforms_lsun(image_size, centered=centered, interpolation=interpolation, antialias=antialias)
            train_transform = train_transform if augment else valid_transform
            data = LSUNClass(root=root + '/cat', transform=train_transform)
            total_examples = len(data)
            train_size = int(0.9 * total_examples)   # use 90% for training
            train_data, valid_data = random_split_dataset(data, [train_size, total_examples - train_size])
        else:
            raise NotImplementedError
    elif dataset.startswith('imagenet_org'):
        image_size = int(dataset.split('_')[-1])
        num_classes = 1000
        num_channels = 3
        train_transform, valid_transform = _data_transforms_lsun(image_size, centered=centered, interpolation=interpolation, antialias=antialias)
        train_transform = train_transform if augment else valid_transform
        train_data = imagenet_lmdb_dataset(root=os.path.join(root, 'train'), transform=train_transform)
        valid_data = imagenet_lmdb_dataset(root=os.path.join(root, 'val'), transform=valid_transform)
    elif dataset.startswith('imagenet'):
        image_size = int(dataset.split('_')[1])
        num_classes = 1000
        num_channels = 3
        # assert root.replace('/', '')[-3:] == dataset.replace('/', '')[-3:], 'the size should match'
        # train_transform, valid_transform = _data_transforms_generic(image_size, centered=centered, interpolation=interpolation, antialias=antialias)
        # train_transform = train_transform if augment else valid_transform
        # train_data = LMDBDataset(root=root, name='imagenet-oord', train=True, transform=train_transform)
        # valid_data = LMDBDataset(root=root, name='imagenet-oord', train=False, transform=valid_transform)
        train_data = ImageFolderDataset(path=os.path.join(root, 'imagenet-{:d}x{:d}.zip'.format(image_size, image_size)), use_labels=False, resolution=image_size, xflip=True, to_01=True)
        valid_data = ImageFolderDataset(path=os.path.join(root, 'imagenet-{:d}x{:d}.zip'.format(image_size, image_size)), use_labels=False, resolution=image_size, xflip=False, to_01=True)
        train_data, _ = random_split_dataset(train_data, [len(train_data), 0])
        valid_data, _ = random_split_dataset(valid_data, [len(valid_data), 0]) # valid data will has 0 length for now
    elif dataset.startswith('ffhq'):
        # image_size = 256
        # num_classes = 1
        # num_channels = 3
        # train_transform, valid_transform = _data_transforms_generic(image_size, centered=centered, interpolation=interpolation, antialias=antialias)
        # train_transform = train_transform if augment else valid_transform
        # train_data = LMDBDataset(root=root, name='ffhq', train=True, transform=train_transform)
        # valid_data = LMDBDataset(root=root, name='ffhq', train=False, transform=valid_transform)
        image_size = int(dataset.split('_')[1])
        num_classes = 1
        num_channels = 3
        train_data = ImageFolderDataset(path=os.path.join(root, 'images{:d}x{:d}.zip'.format(image_size, image_size)), use_labels=False, resolution=image_size, xflip=True, to_01=True)
        valid_data = ImageFolderDataset(path=os.path.join(root, 'images{:d}x{:d}.zip'.format(image_size, image_size)), use_labels=False, resolution=image_size, xflip=False, to_01=True)
        train_data, _ = random_split_dataset(train_data, [len(train_data), 0])
        valid_data, _ = random_split_dataset(valid_data, [len(valid_data), 0]) # valid data will has 0 length for now
    elif dataset.startswith('afhqv2'):
        if dataset.startswith('afhqv2_cat'):
            image_size = int(dataset.split('_')[-1])
            num_classes = 1
            num_channels = 3
            train_data = ImageFolderDataset(path=os.path.join(root, 'afhqv2cat-{:d}x{:d}.zip'.format(image_size, image_size)), use_labels=False, resolution=image_size, xflip=True, to_01=True)
            valid_data = ImageFolderDataset(path=os.path.join(root, 'afhqv2cat-{:d}x{:d}.zip'.format(image_size, image_size)), use_labels=False, resolution=image_size, xflip=False, to_01=True)
            train_data, _ = random_split_dataset(train_data, [len(train_data), 0])
            valid_data, _ = random_split_dataset(valid_data, [len(valid_data), 0]) # valid data will has 0 length for now
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if super_resolution:
        train_data = SuperResolutionDatasetWrapper(train_data, image_size//2)
        valid_data = SuperResolutionDatasetWrapper(valid_data, image_size//2)

    # set image_size and num_channels to data
    train_data.image_size = image_size
    valid_data.image_size = image_size
    train_data.num_channels = num_channels
    valid_data.num_channels = num_channels

    return train_data, valid_data, num_classes

def get_loaders_eval(dataset, root, distributed, batch_size, augment=True, drop_last_train=True, shuffle_train=True,
                     binarize_binary_datasets=True, centered=False, super_resolution=False, interpolation=InterpolationMode.BILINEAR, antialias=None, num_workers=8):
    # get datasets
    train_data, valid_data, num_classes = get_datasets(dataset, root, augment, drop_last_train, shuffle_train, binarize_binary_datasets, centered, super_resolution, interpolation, antialias)

    # distributed sampler
    train_sampler, valid_sampler = None, None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)

    #num_workers = 8
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        shuffle=(train_sampler is None) and shuffle_train,
        sampler=train_sampler, pin_memory=True, num_workers=num_workers, drop_last=drop_last_train)

    # used for debugging
    if num_workers == 0:
        print('***** num_workers is zero *****')

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler, pin_memory=True, num_workers=1, drop_last=False)

    return train_queue, valid_queue, num_classes


def random_split_dataset(dataset, lengths, seed=0):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    g = torch.Generator()
    g.manual_seed(seed)

    indices = torch.randperm(sum(lengths), generator=g)
    return [torch.utils.data.Subset(dataset, indices[offset - length:offset])
            for offset, length in zip(_accumulate(lengths), lengths)]


def _data_transforms_cifar10(centered=False):
    """Get data transforms for cifar10."""
    T_train = [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    T_valid = [transforms.ToTensor()]
    if centered:
        T_train.append(transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))
        T_valid.append(transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))

    train_transform = transforms.Compose(T_train)
    valid_transform = transforms.Compose(T_valid)

    return train_transform, valid_transform

def _data_transforms_mnist(binarize, centered=False):
    """Get data transforms for mnist."""
    T = [transforms.Pad(padding=2), transforms.ToTensor()]
    if binarize:
        T.append(Binarize())
    if centered:
        T.append(transforms.Normalize((0.5), (0.5)))

    train_transform = transforms.Compose(T)
    valid_transform = transforms.Compose(T)

    return train_transform, valid_transform

def _data_transforms_stacked_mnist(binarize, centered=False):
    """Get data transforms for stackmnist."""
    T = [transforms.Pad(padding=2), transforms.ToTensor()]
    if binarize:
        T.append(Binarize())
    if centered:
        T.append(transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))

    train_transform = transforms.Compose(T)
    valid_transform = transforms.Compose(T)

    return train_transform, valid_transform

def _data_transforms_generic(size, centered=False, interpolation=InterpolationMode.BILINEAR, antialias=None):
    T_train = [
        transforms.Resize(size, interpolation=interpolation, antialias=antialias),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    T_valid = [
        transforms.Resize(size, interpolation=interpolation, antialias=antialias),
        transforms.ToTensor(),
    ]
    if centered:
        T_train.append(transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))
        T_valid.append(transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))

    train_transform = transforms.Compose(T_train)
    valid_transform = transforms.Compose(T_valid)

    return train_transform, valid_transform

def _data_transforms_celeba64(size, centered=False, interpolation=InterpolationMode.BILINEAR, antialias=None):
    T_train = [
        CropCelebA64(),
        transforms.Resize(size, interpolation=interpolation, antialias=antialias),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    T_valid = [
        CropCelebA64(),
        transforms.Resize(size, interpolation=interpolation, antialias=antialias),
        transforms.ToTensor(),
    ]
    if centered:
        T_train.append(transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))
        T_valid.append(transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))

    train_transform = transforms.Compose(T_train)
    valid_transform = transforms.Compose(T_valid)

    return train_transform, valid_transform

def _data_transforms_lsun(size, centered=False, interpolation=InterpolationMode.BILINEAR, antialias=None):
    T_train = [
        transforms.Resize(size, interpolation=interpolation, antialias=antialias),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    T_valid = [
        transforms.Resize(size, interpolation=interpolation, antialias=antialias),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ]
    if centered:
        T_train.append(transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))
        T_valid.append(transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))

    train_transform = transforms.Compose(T_train)
    valid_transform = transforms.Compose(T_valid)

    return train_transform, valid_transform

def _data_transforms_vorticity(size, interpolation=InterpolationMode.BICUBIC, antialias=True):
    T = [transforms.Resize(size, interpolation=interpolation, antialias=antialias)]

    train_transform = transforms.Compose(T)
    valid_transform = transforms.Compose(T)

    return train_transform, valid_transform

################################################################################
# SuperResolutionDatasetWrapper
################################################################################
def search_to_tensor(trsfms:list):
    n = len(trsfms)
    for i in range(n-1, -1, -1):
        if isinstance(trsfms[i], transforms.ToTensor):
            return i, n

class SuperResolutionDatasetWrapper(Dataset):
    def __init__(self, dataset, new_size:int, interpolation=InterpolationMode.BILINEAR, antialias=None):
        self.dataset = dataset
        assert isinstance(self.dataset.transform, transforms.Compose)
        assert isinstance(self.dataset.transform.transforms, list)
        ind_start, ind_end = search_to_tensor(self.dataset.transform.transforms)
        self.output_transform = transforms.Compose(
                [transforms.Resize(new_size, interpolation=interpolation, antialias=antialias)] + self.dataset.transform.transforms[ind_start:ind_end])
        self.input_transform = transforms.Compose(self.dataset.transform.transforms[ind_start:ind_end])
        self.dataset.transform.transforms = self.dataset.transform.transforms[:ind_start]

    def __getitem__(self, index):
        data, target = self.dataset.__getitem__(index)
        return self.input_transform(data), self.output_transform(data), target

    def __add__(self, *args, **kwargs):
        raise NotImplementedError()

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return "SuperResolutionDatasetWrapper: \n  " + self.dataset.__repr__() + "\n" + self.input_transform.__repr__() + "\n" + self.output_transform.__repr__()


################################################################################
# ImageNet - LMDB
################################################################################

import io
import os
import lmdb
import torch
from torchvision import datasets
from PIL import Image


def lmdb_loader(path, lmdb_data):
    # In-memory binary streams
    with lmdb_data.begin(write=False, buffers=True) as txn:
        bytedata = txn.get(path.encode('ascii'))
    img = Image.open(io.BytesIO(bytedata))
    return img.convert('RGB')


def imagenet_lmdb_dataset(
        root, transform=None, target_transform=None,
        loader=lmdb_loader):
    if root.endswith('/'):
        root = root[:-1]
    pt_path = os.path.join(
        root + '_faster_imagefolder.lmdb.pt')
    lmdb_path = os.path.join(
        root + '_faster_imagefolder.lmdb')
    if os.path.isfile(pt_path) and os.path.isdir(lmdb_path):
        print('Loading pt {} and lmdb {}'.format(pt_path, lmdb_path))
        data_set = torch.load(pt_path)
    else:
        data_set = datasets.ImageFolder(
            root, None, None, None)
        torch.save(data_set, pt_path, pickle_protocol=4)
        print('Saving pt to {}'.format(pt_path))
        print('Building lmdb to {}'.format(lmdb_path))
        env = lmdb.open(lmdb_path, map_size=1e12)
        with env.begin(write=True) as txn:
            for path, class_index in data_set.imgs:
                with open(path, 'rb') as f:
                    data = f.read()
                txn.put(path.encode('ascii'), data)
    data_set.lmdb_data = lmdb.open(
        lmdb_path, readonly=True, max_readers=1, lock=False, readahead=False,
        meminit=False)
    # reset transform and target_transform
    data_set.samples = data_set.imgs
    data_set.transform = transform
    data_set.target_transform = target_transform
    data_set.loader = lambda path: loader(path, data_set.lmdb_data)
    return data_set
