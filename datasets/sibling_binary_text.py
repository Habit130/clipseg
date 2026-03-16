import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as nnf


class SiblingBinaryTextDataset(object):

    def __init__(self, split, image_size=352, mask='text', normalize=True, data_root=None):
        super().__init__()

        if split not in {'train', 'test'}:
            raise ValueError(f'unsupported split: {split}')
        if mask != 'text':
            raise ValueError('SiblingBinaryTextDataset only supports mask="text"')

        repo_root = Path(__file__).resolve().parents[1]
        default_root = repo_root.parent / 'dataset'
        self.data_root = Path(data_root).expanduser().resolve() if data_root is not None else default_root.resolve()
        self.split = split
        self.image_size = image_size
        self.mask = mask
        self.normalize = normalize
        self.negative_prob = 0.0

        index_path = self.data_root / f'{split}.json'
        if not index_path.is_file():
            raise FileNotFoundError(f'dataset index not found: {index_path}')

        self.samples = json.loads(index_path.read_text(encoding='utf-8'))
        if not isinstance(self.samples, list) or len(self.samples) == 0:
            raise ValueError(f'empty or invalid dataset index: {index_path}')

        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]

    def __len__(self):
        return len(self.samples)

    def _resolve(self, relative_path):
        path = self.data_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f'dataset asset not found: {path}')
        return path

    def _load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        image_tensor = nnf.interpolate(
            image_tensor,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=True,
        )[0]

        if self.normalize:
            image_tensor = (image_tensor - self.mean) / self.std

        return image_tensor

    def _load_mask(self, mask_path):
        mask = Image.open(mask_path).convert('L')
        mask_np = (np.array(mask) != 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
        mask_tensor = nnf.interpolate(mask_tensor, size=(self.image_size, self.image_size), mode='nearest')[0]
        return mask_tensor

    def __getitem__(self, index):
        sample = self.samples[index]

        caption = sample.get('caption')
        if not isinstance(caption, list) or len(caption) < 3:
            raise ValueError(f'sample {sample.get("id", index)} is missing caption[2]')

        image = self._load_image(self._resolve(sample['image']))
        mask = self._load_mask(self._resolve(sample['mask']))

        return (image, caption[2]), (mask, torch.zeros(0), index)
