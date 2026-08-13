import random

from torchvision import transforms
from PIL import Image
import os
import torch
import torch.nn.functional as F
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json

from pathlib import Path

# import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np

torch.multiprocessing.set_sharing_strategy('file_system')


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".JPG", ".JPEG")


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("L")), 
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

def _list_images(folder: str):
    paths = []
    for ext in IMG_EXTS:
        paths += glob.glob(os.path.join(folder, f"*{ext}"))
    return sorted(paths)


def _mvtec_sample_stem(path: str) -> str:
    """Return the shared stem for a MVTec test image and its ``*_mask`` GT."""
    stem = Path(path).stem
    return stem[:-5] if stem.endswith("_mask") else stem


class MVTecDataset_v2(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase='test', mode='test', global_cls_map=None):
        self.transform = transform
        self.gt_transform = gt_transform
        assert phase in ('train', 'test')
        self.phase = phase
        self.mode = mode
        self.root = root
        
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        
        # Build the class map.
        self.cls_name_to_id = self._build_class_mapping(global_cls_map)
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()

    def _build_class_mapping(self, global_cls_map=None):
        """Build the defect-name-to-ID map."""
        # Prefer the supplied global map.
        if global_cls_map is not None:
            return global_cls_map
        
        # Otherwise, build the map from the data directory.
        data_dir = os.path.join(self.root, 'train')  # Build the map from the train directory.
        if not os.path.isdir(data_dir):
            data_dir = os.path.join(self.root, 'test')  # Fall back to the test directory.
        
        defect_names = []
        for name in sorted(os.listdir(data_dir)):
            ddir = os.path.join(data_dir, name)
            if os.path.isdir(ddir) and name != 'good':
                defect_names.append(name)
        
        # Use 0 for normal and 1..K for defects.
        return {n: (i+1) for i, n in enumerate(sorted(set(defect_names)))}

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        if not os.path.isdir(self.img_path):
            raise FileNotFoundError(f"Image path not found: {self.img_path}")

        for defect_type in sorted(os.listdir(self.img_path)):
            cls_dir = os.path.join(self.img_path, defect_type)
            if not os.path.isdir(cls_dir):
                continue
                
            img_paths = _list_images(cls_dir)
            
            if self.phase == 'train':
                if defect_type == 'good':
                    # Normal sample, label 0.
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend([0] * len(img_paths))
                    tot_labels.extend([0] * len(img_paths))
                    tot_types.extend(['good'] * len(img_paths))
                else:
                    # Include defective samples according to mode.
                    if self.mode == 'train_with_anomalies':
                        label = self.cls_name_to_id.get(defect_type, 1)
                        img_tot_paths.extend(img_paths)
                        gt_tot_paths.extend([0] * len(img_paths))
                        tot_labels.extend([label] * len(img_paths))
                        tot_types.extend([defect_type] * len(img_paths))
            else:  # test phase
                if defect_type == 'good':
                    # Normal test sample.
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend([0] * len(img_paths))
                    tot_labels.extend([0] * len(img_paths))
                    tot_types.extend(['good'] * len(img_paths))
                else:
                    # Defective test sample.
                    gdir = os.path.join(self.gt_path, defect_type)
                    gt_map = {}
                    if os.path.isdir(gdir):
                        for gp in _list_images(gdir):
                            gt_map[_mvtec_sample_stem(gp)] = gp
                    
                    label = self.cls_name_to_id.get(defect_type, 1)
                    for ip in img_paths:
                        stem = Path(ip).stem
                        gpath = gt_map.get(stem, 0)
                        img_tot_paths.append(ip)
                        gt_tot_paths.append(gpath)
                        tot_labels.append(label)
                        tot_types.append(defect_type)

        assert len(img_tot_paths) == len(gt_tot_paths), "图像和ground truth数量不匹配!"
        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], int(self.labels[idx]), self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)  # transform should convert PIL images to torch.Tensor.
        
        # img is now a torch.Tensor.
        if label == 0 or gt == 0:
            H, W = img.shape[1], img.shape[2]
            gt = torch.zeros([1, H, W], dtype=torch.float32)
        else:
            gt_img = Image.open(gt)
            gt = self.gt_transform(gt_img)  # gt_transform should return a torch.Tensor.

        # Check spatial dimensions.
        if img.shape[1:] != gt.shape[1:]:
            # Resize GT if needed.
            gt = F.interpolate(gt.unsqueeze(0), size=img.shape[1:], mode='nearest').squeeze(0)
        
        return img, gt, label, img_path

    def get_class_mapping(self):
        return self.cls_name_to_id.copy()

class MVTecTrainWithAnomalies(torch.utils.data.Dataset):
    """
    mode:
      - 'train'                : normal samples only, label=0
      - 'train_with_anomalies' : normal (0) and defects (1..K)
    """
    def __init__(self, root, transform=None, gt_transform=None, phase='train', mode='train', global_cls_map=None):
        self.root = root
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase
        self.mode = mode

        train_dir = os.path.join(root, 'train')
        assert os.path.isdir(train_dir), f"train dir not found: {train_dir}"

        # normal
        normal_dir = os.path.join(train_dir, 'good')
        normal_imgs = _list_images(normal_dir) if os.path.isdir(normal_dir) else []
        normal_samples = [(p, 0) for p in normal_imgs]

        # anomalies
        anomaly_samples = []
        defect_names = []
        for name in sorted(os.listdir(train_dir)):
            ddir = os.path.join(train_dir, name)
            if not os.path.isdir(ddir) or name == 'good':
                continue
            imgs = _list_images(ddir)
            if imgs:
                defect_names.append(name)
                anomaly_samples.extend([(p, name) for p in imgs])

        if mode == 'train':
            self.samples = normal_samples
        elif mode == 'train_with_anomalies':
            if global_cls_map is None:
                cls_name_to_id = {n:(i+1) for i,n in enumerate(sorted(set(defect_names)))}
            else:
                cls_name_to_id = global_cls_map
            self.samples = normal_samples + [(p, cls_name_to_id[n]) for p,n in anomaly_samples]
        else:
            raise ValueError(f"mode should be 'train' or 'train_with_anomalies', but got {mode}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(label)

class MVTecDataset(torch.utils.data.Dataset):
    """
    phase: 'train' or 'test'
      - train: good=0, defect=1; returns a zero mask
      - test : match test and ground_truth by filename stem
    """
    def __init__(self, root, transform, gt_transform, phase):
        self.transform = transform
        self.gt_transform = gt_transform
        assert phase in ('train', 'test')
        self.phase = phase
        self.root = root
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()
        self.cls_idx = 0

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        if not os.path.isdir(self.img_path):
            raise FileNotFoundError(f"Image path not found: {self.img_path}")

        for defect_type in sorted(os.listdir(self.img_path)):
            cls_dir = os.path.join(self.img_path, defect_type)
            if not os.path.isdir(cls_dir):
                continue
            img_paths = _list_images(cls_dir)
            if self.phase == 'train':
                if defect_type == 'good':
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend([0] * len(img_paths))
                    tot_labels.extend([0] * len(img_paths))
                    tot_types.extend(['good'] * len(img_paths))
                else:
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend([0] * len(img_paths))
                    tot_labels.extend([1] * len(img_paths))
                    tot_types.extend([defect_type] * len(img_paths))
            else:  # test
                if defect_type == 'good':
                    for p in img_paths:
                        img_tot_paths.append(p)
                        gt_tot_paths.append(0)
                        tot_labels.append(0)
                        tot_types.append('good')
                else:
                    # map gt by stem
                    gdir = os.path.join(self.gt_path, defect_type)
                    gt_map = {}
                    if os.path.isdir(gdir):
                        for gp in _list_images(gdir):
                            gt_map[_mvtec_sample_stem(gp)] = gp
                    for ip in img_paths:
                        stem = Path(ip).stem
                        gpath = gt_map.get(stem, 0)
                        img_tot_paths.append(ip)
                        gt_tot_paths.append(gpath)
                        tot_labels.append(1)
                        tot_types.append(defect_type)

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], int(self.labels[idx]), self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0 or gt == 0:
            H, W = img.size()[-2], img.size()[-1]
            gt = torch.zeros([1, H, W], dtype=torch.float32)
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"
        return img, gt, label, img_path


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
            self.gt_path = os.path.join(root, 'ground_truth')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = gt.convert('L')  # Ensure grayscale for ground truth
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class RealIADDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase):
        self.img_path = os.path.join(root, 'realiad_1024', category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase

        json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons', category + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []

        data_set = class_json[phase]
        for sample in data_set:
            self.img_paths.append(os.path.join(root, 'realiad_1024', category, sample['image_path']))
            label = sample['anomaly_class'] != 'OK'
            if label:
                self.gt_paths.append(os.path.join(root, 'realiad_1024', category, sample['mask_path']))
            else:
                self.gt_paths.append(None)
            self.labels.append(label)
            self.types.append(sample['anomaly_class'])

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.types = np.array(self.types)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.phase == 'train':
            return img, label

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path

