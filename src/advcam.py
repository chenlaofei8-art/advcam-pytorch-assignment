from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, cast

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


STYLE_LAYERS = ("0", "5", "10", "19", "28")
CONTENT_LAYERS = ("21",)


#保存攻击时用到的几个参数
@dataclass
class AttackConfig:
    steps: int = 400
    lr: float = 0.03
    lambda_adv: float = 5.0
    alpha_style: float = 80000.0
    beta_content: float = 1.0
    tv_weight: float = 0.00001
    log_every: int = 50
    early_stop_confidence: float = 0.90


#自动选择用CPU、GPU还是苹果芯片加速
def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


#按照ImageNet的格式标准化图片
def normalize_for_vgg(tensor: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=tensor.device, dtype=tensor.dtype)
    std = IMAGENET_STD.to(device=tensor.device, dtype=tensor.dtype)
    return (tensor - mean) / std


#加载预训练好的VGG19分类模型
def build_vgg19(device: torch.device) -> tuple[models.VGG, list[str]]:
    """Build an ImageNet-pretrained VGG19 classifier."""
    categories: list[str] = []
    try:
        weights = models.VGG19_Weights.IMAGENET1K_V1
        model = models.vgg19(weights=weights)
        categories = list(weights.meta.get("categories", []))
    except Exception:
        model = models.vgg19(pretrained=True)

    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    features = cast(nn.Sequential, model.features)

    for layer in features:
        if isinstance(layer, nn.ReLU):
            layer.inplace = False
    return model, categories


#找到目标类别在ImageNet里的编号
def resolve_target_index(
    categories: Sequence[str],
    target_label: str = "cinema",
    target_index: int | None = None,
) -> tuple[int, str]:
    if target_index is not None:
        name = categories[target_index] if categories and target_index < len(categories) else f"class_{target_index}"
        return target_index, name

    query = target_label.lower()
    for idx, name in enumerate(categories):
        lowered = name.lower()
        if query in lowered or "movie theater" in lowered or "movie theatre" in lowered:
            return idx, name

    fallback = 498
    name = categories[fallback] if categories and fallback < len(categories) else "cinema / movie theater"
    return fallback, name


#从VGG19中取出风格层和内容层特征
class StyleContentExtractor(nn.Module):
    #初始化要提取哪些中间层
    def __init__(
        self,
        features: nn.Sequential,
        style_layers: Sequence[str] = STYLE_LAYERS,
        content_layers: Sequence[str] = CONTENT_LAYERS,
    ) -> None:
        super().__init__()
        self.features = features
        self.style_layers = set(style_layers)
        self.content_layers = set(content_layers)
        self.needed_layers = self.style_layers | self.content_layers

    #前向传播时保存指定层的输出
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in self.needed_layers:
                outputs[name] = x
        return outputs


#计算特征之间的相关性，用来表示风格
def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = features.shape
    flat = features.view(batch, channels, height * width)
    gram = torch.bmm(flat, flat.transpose(1, 2))
    return gram / (channels * height * width)


#让生成图片不要有太多碎噪声
def total_variation_loss(image: torch.Tensor) -> torch.Tensor:
    horizontal = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    vertical = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    return horizontal + vertical


#返回模型预测概率最高的几个类别
@torch.no_grad()
def predict_topk(
    model: nn.Module,
    image: torch.Tensor,
    categories: Sequence[str],
    k: int = 5,
) -> list[tuple[int, str, float]]:
    logits = model(normalize_for_vgg(image))
    probabilities = torch.softmax(logits, dim=1)[0]
    values, indices = probabilities.topk(k)
    results = []
    for index, probability in zip(indices.tolist(), values.tolist()):
        name = categories[index] if categories and index < len(categories) else f"class_{index}"
        results.append((index, name, probability))
    return results


#核心函数：不断修改图片，让它带风格并骗过模型
def optimize_adversarial_image(
    model: models.VGG,
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    target_index: int,
    categories: Sequence[str],
    config: AttackConfig,
    image_name: str = "image",
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    features = cast(nn.Sequential, model.features)
    extractor = StyleContentExtractor(features).to(content_image.device).eval()
    target = torch.tensor([target_index], device=content_image.device)

    with torch.no_grad():
        style_features = extractor(normalize_for_vgg(style_image))
        style_targets = {layer: gram_matrix(style_features[layer]).detach() for layer in STYLE_LAYERS}
        content_features = extractor(normalize_for_vgg(content_image))
        content_targets = {layer: content_features[layer].detach() for layer in CONTENT_LAYERS}

    adversarial = content_image.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([adversarial], lr=config.lr)
    history: list[dict[str, float]] = []

    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)

        logits = model(normalize_for_vgg(adversarial))
        adv_loss = F.cross_entropy(logits, target)

        generated_features = extractor(normalize_for_vgg(adversarial))
        style_loss = torch.zeros((), device=content_image.device)
        for layer in STYLE_LAYERS:
            style_loss = style_loss + F.mse_loss(gram_matrix(generated_features[layer]), style_targets[layer])

        content_loss = torch.zeros((), device=content_image.device)
        for layer in CONTENT_LAYERS:
            content_loss = content_loss + F.mse_loss(generated_features[layer], content_targets[layer])

        tv_loss = total_variation_loss(adversarial)
        total_loss = (
            config.lambda_adv * adv_loss
            + config.alpha_style * style_loss
            + config.beta_content * content_loss
            + config.tv_weight * tv_loss
        )

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            adversarial.clamp_(0.0, 1.0)
            probabilities = torch.softmax(logits, dim=1)[0]
            target_confidence = float(probabilities[target_index].detach().cpu())
            top1 = int(probabilities.argmax().detach().cpu())

        should_log = step == 1 or step % config.log_every == 0 or step == config.steps
        if should_log:
            top1_name = categories[top1] if categories and top1 < len(categories) else str(top1)
            print(
                f"[{image_name}] step {step:04d}/{config.steps} "
                f"total={total_loss.item():.4f} adv={adv_loss.item():.4f} "
                f"style={style_loss.item():.6f} content={content_loss.item():.4f} "
                f"target_conf={target_confidence:.3f} top1={top1} ({top1_name})"
            )
            history.append(
                {
                    "step": float(step),
                    "total_loss": float(total_loss.detach().cpu()),
                    "adv_loss": float(adv_loss.detach().cpu()),
                    "style_loss": float(style_loss.detach().cpu()),
                    "content_loss": float(content_loss.detach().cpu()),
                    "tv_loss": float(tv_loss.detach().cpu()),
                    "target_confidence": target_confidence,
                    "top1": float(top1),
                }
            )

        if top1 == target_index and target_confidence >= config.early_stop_confidence:
            print(f"[{image_name}] early stop: target reached with confidence {target_confidence:.3f}")
            break

    return adversarial.detach(), history
