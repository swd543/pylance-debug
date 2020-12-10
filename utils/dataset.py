#!/usr/bin/env python3

import SimpleITK as sitk
import torch
import os
import glob
import numpy as np
from torch._C import dtype

class SitkToNumpy(object):
    def __init__(self) -> None:
        pass
    def __call__(self, sample: sitk.Image):
        return np.moveaxis(sitk.GetArrayFromImage(sample), 0, -1)

class OAR_bounding_box_one(object):
    def __init__(self) -> None:
        pass
    def __call__(self, sample: sitk.Image):
        lsif = sitk.LabelShapeStatisticsImageFilter()
        lsif.Execute(sample)
        boundingBox = np.array(lsif.GetBoundingBox(1))
        xs, ys, zs, xl, yl, zl = boundingBox
        to_return = np.zeros(sample.GetSize(), dtype=np.int)
        to_return[xs:xs+xl, ys:ys+yl, zs:zs+zl] = 1
        # GetImageFromArray swaps channel axis
        to_return=np.moveaxis(to_return, -1, 0)
        return sitk.GetImageFromArray(to_return)

class MinMaxNormalize(object):
    def __init__(self, outputMinimum=0, outputMaximum=1, **kwargs):
        self.kwargs = kwargs
        self.outputMinimum = outputMinimum
        self.outputMaximum = outputMaximum

    def __call__(self, sample: sitk.Image):
        return sitk.IntensityWindowing(sample, outputMinimum=self.outputMinimum, outputMaximum=self.outputMaximum, **self.kwargs)

class Brain2019(torch.utils.data.Dataset):
    """Some Information about Miccai2015"""
    def __init__(self, train=True, base_dir=f'{os.environ.get("WORK")}/downloads/Brain2019', transform=None, label_transform=None, valid_labels=None):
        super(Brain2019, self).__init__()
        self.transform = transform
        self.label_transform = label_transform
        self.train = train

        self.reader = sitk.ImageFileReader()
        self.reader.SetImageIO('NrrdImageIO')
        self.reader.SetNumberOfThreads(4)

        sample_file_paths = glob.glob(f'{base_dir}/*')
        _paths_all = {f'{i}/img.nrrd':{os.path.splitext(n)[0]:f'{i}/structures/{n}' for n in os.listdir(f'{i}/structures')} for i in sample_file_paths}
        self._oars = max(_paths_all.values(), key=len).keys()
        self._paths_train = {k:v for k,v in _paths_all.items() if len(v)==len(self._oars)}
        self._paths_valid = {k:v for k,v in _paths_all.items() if len(v)!=len(self._oars)}
        if valid_labels is not None:
            # If all elements in valid_labels are present, then only add to list
            self._paths_valid = {k:v for k,v in self._paths_valid.items() if all(elem in v.keys() for elem in valid_labels)}
        self._paths = self._paths_train if self.train else self._paths_valid

    def _get_image_from_path(self, path):
        self.reader.SetFileName(path)
        img = self.reader.Execute()
        return img

    def __getitem__(self, index):
        sample_path = list(self._paths.keys())[index]
        img = self._get_image_from_path(sample_path)
        segs = {n:self._get_image_from_path(s) for n, s in self._paths[sample_path].items()}
        if self.label_transform is not None:
            segs = {k:self.label_transform(v) for k,v in segs.items()}
        if self.transform is not None:
            img = self.transform(img)
            segs = {k:self.transform(v) for k,v in segs.items()}
        return img, segs

    def __len__(self):
        return len(self._paths)

if __name__ == "__main__":
    from torchvision import transforms
    tfs = transforms.Compose([
        MinMaxNormalize(),
        SitkToNumpy(),
        transforms.ToTensor()
    ])
    a = Brain2019(transform=tfs, label_transform=OAR_bounding_box_one())[0]
    img, segs = a
    print(img.shape, segs['Chiasm'].shape)
    a = Brain2019(transform=tfs, train=False, valid_labels=['Chiasm','Mandible'])
    print(len(a))
    for i,s in a:
        print(s.keys())
