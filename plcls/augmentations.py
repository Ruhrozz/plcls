import warnings
from typing import List, Optional, Tuple

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image


def get_transforms(
    imgsz: Optional[Tuple[int, int]] = None,
    augmentations: str = "none",
    to_tensor: bool = True,
    augs_prob: float = 0.05,
    normalize: bool = True,
    return_lambda: bool = True,
):
    """Get transforms for model training.

    imgsz: Optional[Tuple[int, int]]
        Height and width.
    augmentations: str
        One of ['none', 'light', 'medium', 'heavy']. Now only 'light' is implemented.
    to_tensor: bool
        Add "ToTensorV2" transform.
    """
    augs: List = []

    if imgsz is not None:
        augs.append(A.Resize(*imgsz))

    if augmentations.lower() == "none":
        pass
    elif augmentations.lower() == "light":
        augs.extend(get_light_augmentations(p=augs_prob))
    else:
        raise RuntimeError(f"Available augmentations [light].\nGot{augmentations}")

    if normalize:
        augs.append(A.Normalize())

    if to_tensor:
        augs.append(ToTensorV2())

    compose = A.Compose(augs)

    def compose_fn(image, compose=compose):
        if isinstance(image, Image.Image):
            return compose(image=pil_to_numpy_transform(image))["image"]

        return compose(image=image)["image"]

    if return_lambda:
        return compose_fn

    return compose


def get_light_augmentations(p: float) -> List:
    return [
        A.Blur(p=p),
        A.MedianBlur(p=p),
        A.ToGray(p=p),
        A.CLAHE(p=p),
        A.RandomBrightnessContrast(p=p),
        A.RandomGamma(p=p),
        A.ImageCompression(quality_lower=50, p=p),
    ]


def pil_to_numpy_transform(image: Image.Image):
    """Supposed to be x2.5 faster then np.array(PIL.Image.Image)"""
    if not isinstance(image, Image.Image):
        warnings.warn("Found non PIL image in `pil_to_numpy`")
        return image

    image.load()
    # unpack data
    e = Image._getencoder(image.mode, "raw", image.mode)
    e.setimage(image.im)

    # NumPy buffer for the result
    shape, typestr = Image._conv_type_shape(image)
    data = np.empty(shape, dtype=np.dtype(typestr))
    mem = data.data.cast("B", (data.data.nbytes,))

    bufsize, s, offset = 65536, 0, 0
    while not s:
        _, s, d = e.encode(bufsize)
        mem[offset : offset + len(d)] = d
        offset += len(d)

    if s < 0:
        raise RuntimeError(f"encoder error {s} in tobytes")

    return data
