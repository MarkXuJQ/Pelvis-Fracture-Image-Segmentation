import numpy as np
import cv2
from typing import TypeVar, Optional
from PIL import Image
import albumentations as A
from pathlib import Path
import seaborn as sns

T = TypeVar("T", bound=np.number)
SampleArg = tuple[T, T] | T


CATEGORIES: dict[str, int] = {
    "SA": 1,
    "LI": 2,
    "RI": 3,
}

LABELS: dict[int, str] = {v: k for k, v in CATEGORIES.items()}


def sample(x: SampleArg) -> T:
    return np.random.uniform(x[0], x[1]) if isinstance(x, tuple) else x


class Dropout(A.PixelDropout):
    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        return img


class CoarseDropout(A.CoarseDropout):
    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        return img


def gaussian_contrast_fn(
    images: np.ndarray,
    alpha: float | tuple[float, float] = (0.6, 1.4),
    sigma: float | tuple[float, float] = (0.1, 0.5),
    max_value: float = 1,
):
    original_type = images.dtype
    images = images.astype(np.float32) / max_value

    N, H, W, C = images.shape
    if isinstance(alpha, tuple):
        alpha = np.random.uniform(alpha[0], alpha[1])
    if isinstance(sigma, tuple):
        s = np.random.uniform(sigma[0], sigma[1]) * min(H, W)
    else:
        s = sigma * min(H, W)

    mu_x = np.random.uniform(0, H, size=N)
    mu_y = np.random.uniform(0, W, size=N)
    xs, ys = np.meshgrid(
        np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij"
    )
    xdiff = xs[:, :, None] - mu_x[None, None, :]
    ydiff = ys[:, :, None] - mu_y[None, None, :]
    distance_squared = xdiff**2 + ydiff**2
    h = np.exp(-distance_squared / (2 * s * s))
    hmax = np.max(h, axis=(0, 1), keepdims=True)
    hmap = h / hmax  # in [0, 1]
    alpha_map = hmap * (alpha - 1) + 1
    images = 0.5 + (images - 0.5) * alpha_map
    images = np.clip(images, 0, 1)
    images = (images * max_value).astype(original_type)
    return images


def gaussian_contrast_aug(
    alpha: float | tuple[float, float] = (0.6, 1.4),
    sigma: float | tuple[float, float] = (0.1, 0.5),
    max_value: float = 1,
) -> A.Lambda:
    """Nonuniform contrast augmentation.

    Adjust the contrast by scaling each pixel with value `v` at `x` to
    `0.5 + (v - 0.5) * exp(-(x - mu)**2 / (2 * sigma**2)))`

    Args:
        alpha (float or tuple of float): Alpha of the nonuniform contrast
            augmentation. If a tuple is provided, the value will be randomly
            selected from the range.
        sigma (float or tuple of float): Standard deviation of the Gaussian
            kernel, as a fraction of the (smaller) image size. If a tuple is provided, the value
            will be randomly selected from the range.

    Returns:
        imgaug.augmenters.Lambda: The augmenter.
    """
    if isinstance(alpha, tuple):
        assert len(alpha) == 2
        assert alpha[0] <= alpha[1]

    if isinstance(sigma, tuple):
        assert len(sigma) == 2
        assert sigma[0] <= sigma[1]
        sigma = np.random.uniform(sigma[0], sigma[1])

    def f_image(image, **kwargs):
        # Images are in NHWC
        return gaussian_contrast_fn(np.array([image]), alpha, sigma, max_value=max_value)[0]

    def f_id(x, **kwargs):
        return x

    return A.Lambda(
        image=f_image,
        mask=f_id,
        keypoint=f_id,
        bbox=f_id,
        name="gaussian_contrast",
    )


def neglog_fn(images: np.ndarray, epsilon: float = 0.001) -> np.ndarray:
    """Take the negative log transform of an intensity image.

    Args:
        image (np.ndarray): [N,H,W,C] array of intensity images.
        epsilon (float, optional): positive offset from 0 before taking the logarithm.

    Returns:
        np.ndarray: the image or images after a negative log transform.
    """

    # shift image to avoid invalid values
    images += images.min(axis=(1, 2), keepdims=True) + epsilon

    # negative log transform
    images = -np.log(images)

    return images


def neglog_aug(epsilon: float = 0.001) -> A.Lambda:
    """Take the negative log transform of an intensity image.

    Args:
    """

    def f_image(images: np.ndarray, **kwargs) -> np.ndarray:
        return neglog_fn(images, epsilon)

    def f_id(x, **kwargs):
        return x

    return A.Lambda(
        image=f_image,
        mask=f_id,
        keypoint=f_id,
        bbox=f_id,
        name="neglog",
    )


def window_(
    images: np.ndarray,
    lower: SampleArg = 0.01,
    upper: SampleArg = 0.99,
    convert: bool = True,
) -> np.ndarray:
    """Apply a random window to an intensity image.

    Args:
        images (np.ndarray): [H,W,C] image
        upper (float, optional): The upper quantile of the window. Defaults to 0.99.
        lower (float, optional): The lower quantile of the window. Defaults to 0.01.

    Returns:
        np.ndarray: the image or images after having a random window applied.
    """
    eps = 1e-7
    upper = sample(upper)
    upper = np.quantile(images, upper)

    lower = sample(lower)
    lower = np.quantile(images, lower)

    if upper == lower:
        upper = images.max()
        lower = images.min()

    images = images - lower
    images = images / (upper - lower + eps)
    images = np.clip(images, 0, 1)

    if convert:
        images = (images * 255).astype(np.uint8)
    return images


def window(
    lower: SampleArg = 0.01,
    upper: SampleArg = 0.99,
    convert: bool = True,
):
    """Apply a random window to intensity images.

    Args:
        upper (float, optional): The upper quantile of the window. Defaults to 0.99.
        lower (float, optional): The lower quantile of the window. Defaults to 0.01.

    Returns:
        np.ndarray: the image or images after having a random window applied.
    """

    def _window(images: np.ndarray, **kwargs) -> np.ndarray:
        return window_(images, upper, lower, convert=convert)

    def f_id(x, **kwargs):
        return x

    return A.Lambda(
        image=_window,
        mask=f_id,
        keypoint=f_id,
        bbox=f_id,
        name="window",
    )


def build_augmentation(train: bool = True, img_size: int = 448) -> A.SomeOf:
    """Build an augmentation pipeline.

    Args:
        train: Whether to build an augmentation for training or testing. If True, the wrapped
            function is used to get the training augmentations.

        annotations: Whether the dataset contains annotations.
        image_size: The size to resize images to. If None, no resizing is done.
        normalize: Whether to normalize the image to [-1, 1].

    """
    if not train:
        return A.Compose(
            [neglog_aug(), window(0.01, 0.95, convert=False), A.Resize(img_size, img_size)]
        )

    return A.Compose(
        [
            neglog_aug(),
            window((0, 0.05), (0.95, 1.0), convert=True),
            A.Resize(img_size, img_size),
            A.CLAHE(clip_limit=(1, 4), p=0.5),
            A.InvertImg(p=0.5),
            A.SomeOf(
                [
                    A.OneOf(
                        [
                            A.GaussianBlur((3, 5)),
                            A.MotionBlur(blur_limit=(3, 5)),
                            A.MedianBlur(blur_limit=5),
                        ],
                    ),
                    A.OneOf(
                        [
                            A.Sharpen(alpha=(0.2, 0.5)),
                            A.Emboss(alpha=(0.2, 0.5)),
                        ],
                    ),
                    A.OneOf(
                        [
                            A.MultiplicativeNoise(multiplier=(0.9, 1.1)),
                            A.HueSaturationValue(
                                hue_shift_limit=20,
                                sat_shift_limit=30,
                                val_shift_limit=20,
                            ),
                            A.RandomBrightnessContrast(
                                brightness_limit=(-0.4, 0.2), contrast_limit=(-0.4, 0.2)
                            ),
                            gaussian_contrast_aug(
                                alpha=(0.6, 1.4), sigma=(0.1, 0.5), max_value=255
                            ),
                        ],
                    ),
                    A.RandomToneCurve(scale=0.1),
                    A.OneOf(
                        [
                            A.RandomShadow(),
                            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08),
                        ],
                    ),
                    A.OneOf(
                        [
                            Dropout(dropout_prob=0.05),
                            CoarseDropout(
                                max_holes=12,
                                max_height=24,
                                max_width=24,
                                min_holes=4,
                                min_height=4,
                                min_width=4,
                            ),
                        ],
                        p=3,
                    ),
                ],
                n=np.random.randint(0, 5),
                replace=False,
            ),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),  # Normalize to [0, 1]
        ],
    )


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def _shift(category_id: int, fragment_id: int) -> int:
    return 10 * (category_id - 1) + fragment_id


def masks_to_seg(masks: np.ndarray, category_ids: list[int], fragment_ids: list[int]) -> np.ndarray:
    """Convert masks to a binary-encoded multi-label segmentation.

    Binarizes the segmentation at each pixel by left shifting the one-hot mask by
    10 * (category_id - 1) + (fragment_id)

    Args:
        masks (np.ndarray): [n, h, w] boolean masks.
        category_ids (list[int]): [n] integer category IDs, in SA (1), LI (2) or RI (3).
        fragment_ids (list[int]): [n] integer fragment IDs, in [1,10].

    Returns:
        np.ndarray: [h, w] uint32 segmentation, where each pixel is a 32-bit integer encoding the
            whether the

    """

    seg = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.uint32)
    masks = masks.astype(np.uint32)
    for mask, category_id, fragment_id in zip(masks, category_ids, fragment_ids):
        seg = np.bitwise_or(seg, np.left_shift(mask, _shift(category_id, fragment_id)))
    return seg


def seg_to_masks(seg: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    """Convert a binary-encoded multi-label segmentation to masks."""
    category_ids = []
    fragment_ids = []
    masks = []
    for category_id in CATEGORIES.values():
        for fragment_id in range(1, 11):
            mask = np.right_shift(seg, _shift(category_id, fragment_id)) & 1
            if mask.sum() > 0:
                masks.append(mask)
                category_ids.append(category_id)
                fragment_ids.append(fragment_id)

    return np.array(masks), category_ids, fragment_ids


def load_masks(path: Path) -> tuple[np.ndarray, list[int], list[int]]:
    seg = np.array(Image.open(path))
    return seg_to_masks(seg)


def neglog_window(image: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    """Take the negative log transform of an intensity image.

    Args:
        image (np.ndarray): a single 2D image.
        epsilon (float, optional): positive offset from 0 before taking the logarithm.

    Returns:
        np.ndarray: the image or images after a negative log transform, scaled to [0, 1]
    """
    image = np.array(image)
    shape = image.shape
    if len(shape) == 2:
        image = image[np.newaxis, :, :]

    # shift image to avoid invalid values
    image += image.min(axis=(1, 2), keepdims=True) + epsilon

    # negative log transform
    image = -np.log(image)

    # linear interpolate to range [0, 1]
    image_min = image.min(axis=(1, 2), keepdims=True)
    image_max = image.max(axis=(1, 2), keepdims=True)
    if np.any(image_max == image_min):
        print(
            f"mapping constant image to 0. This probably indicates the projector is pointed away from the volume."
        )
        image[:] = 0
        if image.shape[0] > 1:
            print("TODO: zeroed all images, even though only one might be bad.")
    else:
        image = (image - image_min) / (image_max - image_min)

    if np.any(np.isnan(image)):
        print(f"got NaN values from negative log transform.")

    if len(shape) == 2:
        return image[0]
    else:
        return image


def as_uint8(image: np.ndarray) -> np.ndarray:
    """Convert the image to uint8.

    Args:
        image (np.ndarray): the image to convert.

    Returns:
        np.ndarray: the converted image.
    """
    if image.dtype in [np.float16, np.float32, np.float64]:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    elif image.dtype == bool:
        image = image.astype(np.uint8) * 255
    elif image.dtype != np.uint8:
        print(f"Unknown image type {image.dtype}. Converting to uint8.")
        image = image.astype(np.uint8)
    return image


def as_float32(image: np.ndarray) -> np.ndarray:
    """Convert the image to float32.

    Args:
        image (np.ndarray): the image to convert.

    Returns:
        np.ndarray: the converted image.
    """
    if image.dtype in [np.float16, np.float32, np.float64]:
        image = image.astype(np.float32)
    elif image.dtype == bool:
        image = image.astype(np.float32)
    elif image.dtype != np.uint8:
        print(f"Unknown image type {image.dtype}. Converting to float32.")
        image = image.astype(np.float32)
    else:
        image = image.astype(np.float32) / 255
    return image


def visualize_drr(image: np.ndarray) -> np.ndarray:
    """Process a raw DRR for visualization.

    Args:
        image (np.ndarray): The raw float32 DRR."""
    # Cast to uint8
    image = neglog_window(image)
    image = as_uint8(image)

    # apply clahe and invert
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = 255 - image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def draw_masks(
    image: np.ndarray,
    masks: np.ndarray,
    alpha: float = 0.3,
    threshold: float = 0.5,
    names: Optional[list[str]] = None,
    colors: Optional[np.ndarray] = None,
    palette: str = "hls",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Draw contours of masks on an image (copy).

    Args:
        image (np.ndarray): the image to draw on.
        masks (np.ndarray): the masks to draw. [num_masks, H, W] array of masks.
    """

    image = as_float32(image)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    if colors is None:
        colors = np.array(sns.color_palette(palette, masks.shape[0]))
        if seed is not None:
            np.random.seed(seed)
        colors = colors[np.random.permutation(colors.shape[0])]

    image *= 1 - alpha
    for i, mask in enumerate(masks):
        bool_mask = mask > threshold

        image[bool_mask] = colors[i] * alpha + image[bool_mask] * (1 - alpha)

        contours, _ = cv2.findContours(
            bool_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        image = as_uint8(image)
        cv2.drawContours(image, contours, -1, (255 * colors[i]).tolist(), 1)
        image = as_float32(image)

    image = as_uint8(image)

    fontscale = 0.75 / 512 * image.shape[0]
    thickness = max(int(1 / 256 * image.shape[0]), 1)

    if names is not None:
        for i, mask in enumerate(masks):
            bool_mask = mask > threshold
            ys, xs = np.argwhere(bool_mask).T
            if len(ys) == 0:
                continue
            y = (np.min(ys) + np.max(ys)) / 2
            x = (np.min(xs) + np.max(xs)) / 2
            image = cv2.putText(
                image,
                names[i],
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontscale,
                (255 * colors[i]).tolist(),
                thickness,
                cv2.LINE_AA,
            )

    return image


def visualize_sample(image, masks, category_ids, fragment_ids):
    """Visualize the image and masks."""
    names = [
        f"{LABELS[category_id]}-{fragment_id}"
        for category_id, fragment_id in zip(category_ids, fragment_ids)
    ]
    image = visualize_drr(image)
    return draw_masks(image, masks, names=names, seed=0)


class Dataset:
    def __init__(self, root: Path, split: str, img_size: int = 448):
        self.root = Path(root).expanduser()
        self.split = split
        self.img_size = img_size
        assert self.split in ["train", "val", "test"]

        self.input_dir = self.root / self.split / "input" / "images" / "x-ray"
        self.output_dir = self.root / self.split / "output" / "images" / "x-ray"

        self.image_paths = sorted(self.input_dir.glob("*.tif"))

    def __len__(self, index: int):
        image_path = self.image_paths[index]
        seg_path = self.output_dir / image_path.name

        image = load_image(image_path)
        masks, category_ids, fragment_ids = load_masks(seg_path)
        track_ids = [
            1000 * cat_id + fragment_id for cat_id, fragment_id in zip(category_ids, fragment_ids)
        ]

        # Augmentation
        aug = build_augmentation(train=self.split == "train")
        augmented = aug(image=image, masks=masks, category_ids=track_ids)

        image = augmented["image"]
        masks = augmented["masks"]
        track_ids = augmented["category_ids"]
        category_ids = [track_id // 1000 for track_id in track_ids]
        fragment_ids = [track_id % 1000 for track_id in track_ids]

        return image, masks, category_ids, fragment_ids


if __name__ == "__main__":
    import shutil
    import imageio.v3 as iio

    root = Path("/home/killeen/datasets/OneDrive/datasets/PENGWIN")
    image_path = root / Path("test/input/images/x-ray/122_0350.tif")
    mask_path = root / Path("test/output/images/x-ray/122_0350.tif")

    shutil.copy(str(mask_path), "images/seg1.tif")

    # tiff to masks
    image = load_image(image_path)
    masks, category_ids, fragment_ids = load_masks(mask_path)
    print(category_ids, fragment_ids)
    print(masks.shape)

    vis_image = visualize_sample(image, masks, category_ids, fragment_ids)
    vis_path = Path("images/sample_original.png")
    cv2.imwrite(str(vis_path), vis_image)
    print(f"Wrote image to {vis_path}")

    # masks to tiff
    seg_cycle = masks_to_seg(masks, category_ids, fragment_ids)
    seg_path = Path("images/seg2.tif")
    iio.imwrite(seg_path, seg_cycle)
    # Image.fromarray(seg_cycle).save(seg_path)
    print(f"Wrote segmentation to {seg_path}")

    # Images/seg2.tif and Images/seg1.tif should be the same
    masks, category_ids, fragment_ids = load_masks(seg_path)
    print(category_ids, fragment_ids)
    vis_image = visualize_sample(image, masks, category_ids, fragment_ids)
    cv2.imwrite("images/sample_cycle.png", vis_image)
    print(f"Wrote image to images/sample_cycle.png")
