import os
import re
import sys
from abc import abstractmethod
from reproducibility.utils.cacher import cache_hit_or_miss, cache_numpy_object
from reproducibility.embedders.internal_datasets import CLIPImageDataset
from torch.utils.data import DataLoader
from torch.utils import model_zoo
from torchvision.models.resnet import ResNet, model_urls as resnet_urls, BasicBlock, Bottleneck
from torchvision.models.densenet import DenseNet, model_urls as densenet_urls
import torch.nn.functional as F
import numpy as np
from torch import nn

class FeaturesInterface(object):
    @abstractmethod
    def n_features(self):
        pass

import torch
from torch.hub import download_url_to_file

try:
    from requests.utils import urlparse
    from requests import get as urlopen
    requests_available = True
except ImportError:
    requests_available = False
    from urllib.request import urlopen
    from urllib.parse import urlparse
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # defined below


def _remove_prefix(s, prefix):
    if s.startswith(prefix):
        s = s[len(prefix):]
    return s


def clean_state_dict(state_dict, prefix, filter=None):
    if filter is None:
        filter = lambda *args: True
    return {_remove_prefix(k, prefix): v for k, v in state_dict.items() if filter(k)}


def load_dox_url(url, filename, model_dir=None, map_location=None, progress=True):
    r"""Adapt to fit format file of mtdp pre-trained models
    """
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        sys.stderr.flush()
        download_url_to_file(url, cached_file, None, progress=progress)
    return torch.load(cached_file, map_location=map_location)



MTDRN_URLS = {
    "resnet50": ("https://dox.uliege.be/index.php/s/kvABLtVuMxW8iJy/download", "resnet50-mh-best-191205-141200.pth")
}


class NoHeadResNet(ResNet, FeaturesInterface):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x

    def n_features(self):
        return [b for b in list(self.layer4[-1].children()) if hasattr(b, 'num_features')][-1].num_features


def build_resnet(download_dir, pretrained=None, arch="resnet50", model_class=NoHeadResNet, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        arch (str): Type of densenet (among: resnet18, resnet34, resnet50, resnet101 and resnet152)
        pretrained (str|None): If "imagenet", returns a model pre-trained on ImageNet. If "mtdp" returns a model
                              pre-trained in multi-task on digital pathology data. Otherwise (None), random weights.
        model_class (nn.Module): Actual resnet module class
    """
    params = {
        "resnet18": [BasicBlock, [2, 2, 2, 2]],
        "resnet34": [BasicBlock, [3, 4, 6, 3]],
        "resnet50": [Bottleneck, [3, 4, 6, 3]],
        "resnet101": [Bottleneck, [3, 4, 23, 3]],
        "resnet152":  [Bottleneck, [3, 8, 36, 3]]
    }
    model = model_class(*params[arch], **kwargs)
    if isinstance(pretrained, str):
        if pretrained == "imagenet":
            url = resnet_urls[arch]  # default imagenet
            state_dict = model_zoo.load_url(url)
        elif pretrained == "mtdp":
            if arch not in MTDRN_URLS:
                raise ValueError("No pretrained weights for multi task pretraining with architecture '{}'".format(arch))
            url, filename = MTDRN_URLS[arch]
            state_dict = load_dox_url(url, filename, model_dir=download_dir, map_location="cpu")
            state_dict = clean_state_dict(state_dict, prefix="features.", filter=lambda k: not k.startswith("heads."))
        else:
            raise ValueError("Unknown pre-training source")
        model.load_state_dict(state_dict)
    return model

MTDP_URLS = {
    "densenet121": ("https://dox.uliege.be/index.php/s/G72InP4xmJvOrVp/download", "densenet121-mh-best-191205-141200.pth")
}


class NoHeadDenseNet(DenseNet, FeaturesInterface):
    def forward(self, x):
        return F.adaptive_avg_pool2d(self.features(x), (1, 1))

    def n_features(self):
        return self.features[-1].num_features


def build_densenet(download_dir, pretrained=False, arch="densenet121", model_class=NoHeadDenseNet, **kwargs):
    r"""Densenet-XXX model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        arch (str): Type of densenet (among: densenet121, densenet169, densenet201 and densenet161)
        pretrained (str|None): If "imagenet", returns a model pre-trained on ImageNet. If "mtdp" returns a model pre-trained
                           in multi-task on digital pathology data. Otherwise (None), random weights.
        model_class (nn.Module): Actual densenet module class
    """
    params = {
        "densenet121": {"num_init_features": 64, "growth_rate": 32, "block_config": (6, 12, 24, 16)},
        "densenet169": {"num_init_features": 64, "growth_rate": 32, "block_config": (6, 12, 32, 32)},
        "densenet201": {"num_init_features": 64, "growth_rate": 32, "block_config": (6, 12, 48, 32)},
        "densenet161": {"num_init_features": 96, "growth_rate": 48, "block_config": (6, 12, 36, 24)}
    }
    model = model_class(**(params[arch]), **kwargs)
    if isinstance(pretrained, str):
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        if pretrained == "imagenet":
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            state_dict = model_zoo.load_url(densenet_urls[arch])
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
        elif pretrained == "mtdp":
            if arch not in MTDP_URLS:
                raise ValueError("No pretrained weights for multi task pretraining with architecture '{}'".format(arch))
            url, filename = MTDP_URLS[arch]
            state_dict = load_dox_url(url, filename, model_dir=download_dir, map_location="cpu")
            state_dict = clean_state_dict(state_dict, prefix="features.", filter=lambda k: not k.startswith("heads."))
        else:
            raise ValueError("Unknown pre-training source")
        model.load_state_dict(state_dict)
    return model


class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class DenseNetEmbedder:
    def __init__(self, model, preprocess, name, backbone):
        self.model = model
        self.preprocess = preprocess
        self.name = name
        self.backbone = backbone

    def image_embedder(self, list_of_images, device="cuda", num_workers=1, batch_size=32, additional_cache_name=""):
        # additional_cache_name: name of the validation dataset (e.g., Kather_7K)
        hit_or_miss = cache_hit_or_miss(self.name + "img" + additional_cache_name, self.backbone)

        if hit_or_miss is not None:
            return hit_or_miss
        else:
            hit = self.embed_images(list_of_images, device=device, num_workers=num_workers, batch_size=batch_size)
            cache_numpy_object(hit, self.name + "img" + additional_cache_name, self.backbone)
            return hit

    def embed_images(self, list_of_images, device="cuda", num_workers=1, batch_size=32):
        dataset = CLIPImageDataset(list_of_images, self.preprocess)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        all_embs = []
        for batch_X in tqdm(dataloader):
            batch_X = batch_X.to(device)
            embeddings = self.model(batch_X).detach().float().squeeze()
            embeddings = embeddings.detach().cpu().numpy()
            all_embs.append(embeddings)
        return np.concatenate(all_embs)


