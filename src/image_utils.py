from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


#找出文件夹里的图片文件
def list_images(folder: Path, limit: int | None = None) -> list[Path]:
    """Return image paths in a stable order."""
    paths = [
        p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if limit is not None:
        paths = paths[:limit]
    return paths


#从图片中间裁成正方形
def center_crop_square(image: Image.Image) -> Image.Image:
    """Crop the image to a centered square."""
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


#读取图片并转成PyTorch能用的格式
def load_image_tensor(path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    """Load an RGB image as a 1x3xHxW float tensor in [0, 1]."""
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = center_crop_square(image)
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    array = np.asarray(image).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


#把tensor转回普通图片
def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a 1x3xHxW or 3xHxW tensor in [0, 1] to a PIL image."""
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = tensor.detach().clamp(0, 1).cpu()
    array = (tensor.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    return Image.fromarray(array)


#保存tensor格式的图片
def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(tensor).save(path)


#把预测结果整理成几行文字
def format_predictions(predictions: Iterable[tuple[int, str, float]]) -> list[str]:
    lines = []
    for rank, (idx, name, prob) in enumerate(predictions, start=1):
        lines.append(f"{rank}. {idx}: {name} ({prob:.2%})")
    return lines


#尝试加载一个可用字体
def _load_font(size: int) -> Any:
    for font_name in ("Arial.ttf", "Helvetica.ttc", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


#生成一张包含原图、结果图和预测标签的证据图
def make_prediction_card(
    original: torch.Tensor,
    adversarial: torch.Tensor,
    original_predictions: list[tuple[int, str, float]],
    adversarial_predictions: list[tuple[int, str, float]],
    target_index: int,
    target_name: str,
    output_path: Path,
) -> None:
    """Create a PNG summary that can be used as a prediction screenshot."""
    original_image = tensor_to_pil(original).resize((320, 320), Image.Resampling.LANCZOS)
    adversarial_image = tensor_to_pil(adversarial).resize((320, 320), Image.Resampling.LANCZOS)

    width = 920
    height = 560
    card = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(card)
    title_font = _load_font(24)
    body_font = _load_font(17)
    small_font = _load_font(15)

    draw.text((24, 18), "VGG19 Prediction Evidence", fill=(20, 20, 20), font=title_font)
    draw.text(
        (24, 54),
        f'Target label: {target_index} - "{target_name}"',
        fill=(70, 70, 70),
        font=body_font,
    )

    card.paste(original_image, (24, 104))
    card.paste(adversarial_image, (360, 104))
    draw.text((24, 432), "Original image", fill=(20, 20, 20), font=body_font)
    draw.text((360, 432), "Generated adversarial image", fill=(20, 20, 20), font=body_font)

    draw.text((690, 104), "Before attack", fill=(20, 20, 20), font=body_font)
    y = 134
    for line in format_predictions(original_predictions):
        draw.text((690, y), line, fill=(60, 60, 60), font=small_font)
        y += 26

    draw.text((690, 286), "After attack", fill=(20, 20, 20), font=body_font)
    y = 316
    for line in format_predictions(adversarial_predictions):
        color = (0, 120, 70) if str(target_index) in line.split(":", 1)[0] else (60, 60, 60)
        draw.text((690, y), line, fill=color, font=small_font)
        y += 26

    success = adversarial_predictions[0][0] == target_index
    status = "SUCCESS: top-1 prediction is the target label." if success else "NOT TOP-1 YET: tune steps/weights and rerun."
    draw.rectangle((24, 486, 896, 532), fill=(230, 248, 239) if success else (255, 244, 224))
    draw.text((42, 498), status, fill=(20, 90, 60) if success else (140, 80, 20), font=body_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    card.save(output_path)
