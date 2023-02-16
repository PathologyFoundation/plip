import torch
import clip
from embedders.plip import CLIPEmbedder
from embedders.mudipath import build_densenet
from torchvision import transforms
from embedders.mudipath import DenseNetEmbedder

class EmbedderFactory:

    def __init__(self):
        pass
    
    def factory(self, name, path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if name == "plip":
            model, preprocess = clip.load("ViT-B/32", device=device)
            model.load_state_dict(torch.load(path))
            return CLIPEmbedder(model, preprocess, name, path)

        elif name == "clip":
            model, preprocess = clip.load("ViT-B/32", device=device)

            return CLIPEmbedder(model, preprocess, name, path)

        elif name == "mudipath":
            backbone = build_densenet(download_dir="/oak/stanford/groups/jamesz/fede/medical_clip/pathology_notebooks/",
                                      pretrained="mtdp")  # TODO fixed path
            backbone.num_feats = backbone.n_features()
            backbone.forward_type = "image"
            backbone = backbone.to(device)
            backbone.eval()
            image_preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
            ])
            return DenseNetEmbedder(backbone, image_preprocess, name, path)