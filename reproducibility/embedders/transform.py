from torchvision.transforms import (
    RandomAffine,
    RandomPerspective,
    RandomAutocontrast,
    RandomEqualize,
    RandomRotation,
    RandomCrop,
    RandomHorizontalFlip
)
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _train_transform(first_resize=512,
                    n_px=224,
                    ):
    return Compose([
        Resize([first_resize], interpolation=InterpolationMode.BICUBIC),# Desired output size. If size is a sequence like (h, w), output size will be matched to this. If size is an int, smaller edge of the image will be matched to this number. i.e, if height > width, then image will be rescaled to (size * height / width, size).
        RandomCrop([n_px]),
        RandomHorizontalFlip(),
        RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
            shear=(-15, 15, -15, 15),
            interpolation=InterpolationMode.BILINEAR,
            fill=127,
        ),
        RandomPerspective(
            distortion_scale=0.3,
            p=0.3,
            interpolation=InterpolationMode.BILINEAR,
            fill=127,
        ),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
