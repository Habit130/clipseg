import json
from os.path import dirname, isfile, join, realpath

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as nnf


class PlantSegDataset(object):

    def __init__(self, split, image_size=352, mask='text', caption_index=3, data_root=None, normalize=True):
        super().__init__()

        if mask != 'text':
            raise ValueError('PlantSegDataset only supports mask="text"')

        self.split = split
        self.image_size = image_size
        self.mask = mask
        self.caption_index = caption_index
        self.normalize = normalize
        self.negative_prob = 0.0
        self.base_dir = realpath(data_root or join(dirname(__file__), '..', '..', 'plantseg'))

        manifest_path = join(self.base_dir, 'main.json')
        if not isfile(manifest_path):
            raise FileNotFoundError(
                f'PlantSeg manifest was not found at {manifest_path}. '
                'Expected the dataset directory to live next to the repository root.'
            )

        with open(manifest_path, 'r', encoding='utf-8') as handle:
            manifest = json.load(handle)

        self.samples = [sample for sample in manifest if sample['split'] == split]
        if len(self.samples) == 0:
            raise ValueError(f'No PlantSeg samples were found for split "{split}"')

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def _load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image_tensor = nnf.interpolate(
            image_tensor,
            (self.image_size, self.image_size),
            mode='bilinear',
            align_corners=True,
        )[0]

        if self.normalize:
            image_tensor = (image_tensor - self.mean) / self.std

        return image_tensor

    def _load_mask(self, mask_path):
        mask = Image.open(mask_path).convert('L')
        mask_np = (np.array(mask) > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).view(1, 1, *mask_np.shape)
        mask_tensor = nnf.interpolate(mask_tensor, (self.image_size, self.image_size), mode='nearest')[0]
        return mask_tensor

    def __getitem__(self, index):
        sample = self.samples[index]

        image_path = join(self.base_dir, sample['image'])
        mask_path = join(self.base_dir, sample['mask'])
        caption = sample['caption'][self.caption_index]

        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        return (image, caption), (mask, torch.zeros(0), index)
