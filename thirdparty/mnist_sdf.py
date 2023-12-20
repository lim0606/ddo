"""
Copied and modified from https://github.com/vsitzmann/metasdf/blob/56d0c6e411d23523bd638aa0ad37c0c2c9dbdb9d/MNISTMetaSDFDemo.ipynb
"""
import numpy as np
import scipy
import scipy.ndimage
import torch
import torchvision
from torchvision import transforms

from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode


def get_interpolation_mode(interpolation='bilinear'):
    if interpolation == 'bilinear':
        return InterpolationMode.BILINEAR
    elif interpolation == 'bicubic':
        return InterpolationMode.BICUBIC
    else:
        raise NotImplementedError

def get_mgrid(sidelen):
    # Generate 2D pixel coordinates from an image of sidelen x sidelen
    pixel_coords = np.stack(np.mgrid[:sidelen,:sidelen], axis=-1)[None,...].astype(np.float32)
    pixel_coords /= sidelen
    pixel_coords -= 0.5
    pixel_coords = torch.Tensor(pixel_coords).view(-1, 2)
    return pixel_coords

class SignedDistanceTransform:
    def __call__(self, img_tensor):
        # Threshold.
        img_tensor[img_tensor<0.5] = 0.
        img_tensor[img_tensor>=0.5] = 1.

        # Compute signed distances with distance transform
        img_tensor = img_tensor.numpy()

        neg_distances = scipy.ndimage.morphology.distance_transform_edt(img_tensor)
        sd_img = img_tensor - 1.
        sd_img = sd_img.astype(np.uint8)
        signed_distances = scipy.ndimage.morphology.distance_transform_edt(sd_img) - neg_distances
        signed_distances /= float(img_tensor.shape[1])
        signed_distances = torch.Tensor(signed_distances)

        return signed_distances, torch.Tensor(img_tensor)

class SDFDataset(Dataset):
    def __init__(self, dataset, size:tuple=(256, 256)):
        self.img_dataset = dataset
        self.sdf = SignedDistanceTransform()
        self.im_size = size
        self.meshgrid = get_mgrid(size[0])

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, item):
        items = self.img_dataset.__getitem__(item)
        if type(items) in [list, tuple]:
            img = items[0]
        sz = list(img.shape)
        assert sz[1] == self.im_size[0] and sz[2] == self.im_size[1]

        signed_distance_img, binary_image = self.sdf(img)

        coord_values = self.meshgrid.reshape(-1, 2)
        signed_distance_values = signed_distance_img.reshape((-1, 1))

        return signed_distance_values, coord_values


class MNISTSDFDataset(Dataset):
    def __init__(self,
                 root,
                 train:bool,
                 download:bool=True,
                 size:tuple=(256, 256),
                 interpolation:str='bilinear',
                 antialias:bool=None,
                 return_image:bool=False,
                ):
        self.transform = transforms.Compose([
            transforms.Resize(size,
                              interpolation=get_interpolation_mode(interpolation),
                              antialias=antialias,
                              ),
            transforms.ToTensor(),
            SignedDistanceTransform(),
        ])
        self.img_dataset = torchvision.datasets.MNIST(root, train=train, download=download)
        self.meshgrid = get_mgrid(size[0])
        self.im_size = size
        self.return_image = return_image

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, item):
        img, digit_class = self.img_dataset[item]

        signed_distance_img, binary_image = self.transform(img)

        if self.return_image:
            return signed_distance_img, digit_class

        coord_values = self.meshgrid.reshape(-1, 2)
        signed_distance_values = signed_distance_img.reshape((-1, 1))

        return signed_distance_values, coord_values

        # indices = torch.randperm(coord_values.shape[0])
        # support_indices = indices[:indices.shape[0]//2]
        # query_indices = indices[indices.shape[0]//2:]

        # meta_dict = {'context': (coord_values[support_indices], signed_distance_values[support_indices]), 'query': (coord_values[query_indices], signed_distance_values[query_indices]), 'all': (coord_values, signed_distance_values)}

        # return meta_dict
