"""Image preprocessing mirroring ``notebooks/02_first_model.ipynb``.

Notebook behavior (cell "Read a processed RGB image"):
* processed dataset images are 256x256 RGB PNGs, verified strictly.

Inference behavior for arbitrary user uploads (this module):
* accept a file path, ``PIL.Image``, or ``numpy`` array
* convert to RGB, resize to 256x256 (LANCZOS, same filter as notebook 01),
  scale to [0, 1], normalize with ImageNet mean/std.

Third-party imports (PIL/numpy/torch) are done lazily inside functions so
that pure helpers (padding / patch-grid math) remain importable and testable
without any dependency installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Union

from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, PROCESSED_SIZE

if TYPE_CHECKING:
    import numpy as _np
    import torch as _torch
    from PIL import Image as _PILImage

logger = logging.getLogger(__name__)

ImageLike = Union[str, Path, "_PILImage.Image", "_np.ndarray"]


# ---------------------------------------------------------------------------
# Pure helpers (no third-party dependency needed)
# ---------------------------------------------------------------------------
def compute_padding(
    image_height: int, image_width: int, patch_height: int, patch_width: int
) -> tuple[int, int, int, int]:
    """Replicate the notebook edge-padding rule for DINOv2 patch embedding.

    Returns ``(pad_right, pad_left_unused, padded_height, padded_width)`` as
    ``(pad_right, pad_bottom, padded_height, padded_width)``.
    """
    if image_height <= 0 or image_width <= 0:
        raise ValueError(
            f"Image dimensions must be positive, got ({image_height}, {image_width})."
        )
    if patch_height <= 0 or patch_width <= 0:
        raise ValueError(
            f"Patch dimensions must be positive, got ({patch_height}, {patch_width})."
        )
    pad_bottom = (-image_height) % patch_height
    pad_right = (-image_width) % patch_width
    return pad_right, pad_bottom, image_height + pad_bottom, image_width + pad_right


def compute_patch_grid(
    padded_height: int, padded_width: int, patch_height: int, patch_width: int
) -> tuple[int, int]:
    """Number of patches along (height, width) for a padded image."""
    if padded_height % patch_height != 0 or padded_width % patch_width != 0:
        raise ValueError(
            f"Padded size ({padded_height}, {padded_width}) is not divisible by "
            f"patch size ({patch_height}, {patch_width})."
        )
    return padded_height // patch_height, padded_width // patch_width


# ---------------------------------------------------------------------------
# Image I/O + tensor conversion (require PIL / numpy / torch at call time)
# ---------------------------------------------------------------------------
def load_rgb_image(source: ImageLike) -> _PILImage.Image:
    """Open *source* and return a 256x256 RGB ``PIL.Image``.

    * ``str``/``Path``: file must exist and be a decodable image.
    * ``PIL.Image``: converted to RGB and resized.
    * ``numpy`` array: ``(H, W, 3)`` uint8, converted to RGB and resized.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for image loading.") from exc

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}.")
        try:
            image = Image.open(path)
            image.load()
        except (OSError, ValueError) as exc:
            raise ValueError(f"Unable to decode image file: {path} ({exc}).") from exc
    elif isinstance(source, Image.Image):
        image = source
    else:  # numpy array
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("numpy is required for ndarray inputs.") from exc
        if not isinstance(source, np.ndarray):
            raise TypeError(
                f"Unsupported image input type: {type(source).__name__}. "
                "Expected a file path, PIL.Image, or numpy array."
            )
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError(
                f"ndarray images must have shape (H, W, 3), got {source.shape}."
            )
        image = Image.fromarray(source)

    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    return image


def read_processed_rgb(path: str | Path) -> _np.ndarray:
    """Strict reader for already-preprocessed 256x256 RGB dataset images.

    Exact replica of the notebook ``read_rgb`` helper (raises on unexpected
    mode/size instead of resizing).
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for image loading.") from exc
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Image file not found: {file_path}.")
    try:
        import numpy as np

        with Image.open(file_path) as image:
            image.load()
            if image.mode != "RGB":
                raise ValueError(f"Expected RGB image: {file_path}.")
            if image.size != (PROCESSED_SIZE[1], PROCESSED_SIZE[0]):
                raise ValueError(
                    f"Unexpected processed image size {image.size}: {file_path}."
                )
            return np.array(image, copy=True)
    except (OSError, ValueError) as exc:
        if isinstance(exc, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Unable to decode image file: {file_path} ({exc}).") from exc


def image_tensor(source: ImageLike) -> _torch.Tensor:
    """Convert *source* to a normalized ``(3, 256, 256)`` float tensor.

    Mirrors the notebook ``image_tensor`` helper: ``/255`` then
    ``(x - mean) / std`` with ImageNet statistics.
    """
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise ImportError("numpy and torch are required for preprocessing.") from exc

    pixels = np.array(load_rgb_image(source), copy=True)
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


def prepare_image(source: ImageLike) -> _torch.Tensor:
    """Alias of :func:`image_tensor` (public entry point for inference)."""
    return image_tensor(source)


def pad_for_patch_embedding(
    images: _torch.Tensor, pad_right: int, pad_bottom: int
) -> _torch.Tensor:
    """Right/bottom replicate-pad a ``(B, C, H, W)`` batch for patch embedding.

    Mirrors the notebook ``F.pad(images, (0, pad_right, 0, pad_bottom),
    mode="replicate")`` call, which preserves all original pixels when 256
    is not divisible by the DINOv2 patch size 14.
    """
    try:
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("torch is required for padding.") from exc
    if images.ndim != 4:
        raise ValueError(f"Expected a 4D batch tensor, got shape {tuple(images.shape)}.")
    if pad_right < 0 or pad_bottom < 0:
        raise ValueError(
            f"Padding must be non-negative, got ({pad_right}, {pad_bottom})."
        )
    if pad_right == 0 and pad_bottom == 0:
        return images
    return F.pad(images, (0, pad_right, 0, pad_bottom), mode="replicate")
