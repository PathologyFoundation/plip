from embedders.plip import CLIPEmbedder
import torch
import clip


class EmbedderFactory:

    def __init__(self):
        pass
    
    def factory(self, name, path):
        if name == "plip":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)
            model.load_state_dict(torch.load(path))
            return CLIPEmbedder(model, preprocess, name, path)

        elif name == "clip":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)

            return CLIPEmbedder(model, preprocess, name, path)