from embedders.plip import CLIPEmbedder
import torch
import clip


class EmbedderFactory:

    def __init__(self):
        pass
    
    def factory(self, name, path):
        if name == "plip":

            return
        elif name == "clip":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load(path, device=device)

            return CLIPEmbedder(model, preprocess, name, path)