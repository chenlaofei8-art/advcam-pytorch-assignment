from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm

from src.advcam import (
    AttackConfig,
    build_vgg19,
    choose_device,
    optimize_adversarial_image,
    predict_topk,
    resolve_target_index,
)
from src.image_utils import (
    list_images,
    load_image_tensor,
    make_prediction_card,
    save_tensor_image,
)


#读取命令行里的运行参数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate style-transfer adversarial examples for VGG19."
    )
    parser.add_argument("--content-dir", type=Path, default=Path("data/content"))
    parser.add_argument("--style-image", type=Path, default=Path("data/style/style.jpg"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--target-label", type=str, default="cinema")
    parser.add_argument("--target-index", type=int, default=None)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--lambda-adv", type=float, default=5.0)
    parser.add_argument("--alpha-style", type=float, default=80000.0)
    parser.add_argument("--beta-content", type=float, default=1.0)
    parser.add_argument("--tv-weight", type=float, default=0.00001)
    parser.add_argument("--early-stop-confidence", type=float, default=0.90)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


#检查原图和风格图有没有准备好
def ensure_inputs(content_dir: Path, style_image: Path, num_images: int) -> list[Path]:
    if not content_dir.exists():
        raise FileNotFoundError(f"Content image folder not found: {content_dir}")
    if not style_image.exists():
        raise FileNotFoundError(f"Style image not found: {style_image}")

    images = list_images(content_dir, limit=num_images)
    if len(images) < num_images:
        raise RuntimeError(
            f"Need {num_images} input images in {content_dir}, but found {len(images)}."
        )
    return images


#把每张图的最终预测结果写成表格
def write_summary(summary_path: Path, rows: list[dict[str, str | int | float]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "original_top1_index",
        "original_top1_label",
        "original_top1_confidence",
        "adversarial_top1_index",
        "adversarial_top1_label",
        "adversarial_top1_confidence",
        "target_index",
        "target_label",
        "success",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


#主流程：加载图片、攻击模型、保存结果
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    image_paths = ensure_inputs(args.content_dir, args.style_image, args.num_images)
    device = choose_device(args.device)
    print(f"Using device: {device}")

    model, categories = build_vgg19(device)
    target_index, target_name = resolve_target_index(
        categories, target_label=args.target_label, target_index=args.target_index
    )
    print(f'Target class: {target_index} - "{target_name}"')

    output_dir = args.output_dir
    adversarial_dir = output_dir / "adversarial_images"
    original_dir = output_dir / "original_images"
    card_dir = output_dir / "prediction_cards"
    log_dir = output_dir / "logs"
    for folder in (adversarial_dir, original_dir, card_dir, log_dir):
        folder.mkdir(parents=True, exist_ok=True)

    style_tensor = load_image_tensor(args.style_image, args.image_size, device)
    save_tensor_image(style_tensor, output_dir / "style_reference.png")

    config = AttackConfig(
        steps=args.steps,
        lr=args.lr,
        lambda_adv=args.lambda_adv,
        alpha_style=args.alpha_style,
        beta_content=args.beta_content,
        tv_weight=args.tv_weight,
        log_every=args.log_every,
        early_stop_confidence=args.early_stop_confidence,
    )

    rows: list[dict[str, str | int | float]] = []

    for image_path in tqdm(image_paths, desc="Attacking images"):
        stem = image_path.stem
        content_tensor = load_image_tensor(image_path, args.image_size, device)
        original_predictions = predict_topk(model, content_tensor, categories, k=5)

        adversarial_tensor, history = optimize_adversarial_image(
            model=model,
            content_image=content_tensor,
            style_image=style_tensor,
            target_index=target_index,
            categories=categories,
            config=config,
            image_name=stem,
        )
        adversarial_predictions = predict_topk(model, adversarial_tensor, categories, k=5)

        save_tensor_image(content_tensor, original_dir / f"{stem}_original.png")
        save_tensor_image(adversarial_tensor, adversarial_dir / f"{stem}_adversarial.png")
        make_prediction_card(
            original=content_tensor,
            adversarial=adversarial_tensor,
            original_predictions=original_predictions,
            adversarial_predictions=adversarial_predictions,
            target_index=target_index,
            target_name=target_name,
            output_path=card_dir / f"{stem}_prediction_card.png",
        )

        with (log_dir / f"{stem}_loss_history.csv").open("w", newline="", encoding="utf-8") as file:
            fieldnames = [
                "step",
                "total_loss",
                "adv_loss",
                "style_loss",
                "content_loss",
                "tv_loss",
                "target_confidence",
                "top1",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

        original_top1 = original_predictions[0]
        adversarial_top1 = adversarial_predictions[0]
        rows.append(
            {
                "image": image_path.name,
                "original_top1_index": original_top1[0],
                "original_top1_label": original_top1[1],
                "original_top1_confidence": original_top1[2],
                "adversarial_top1_index": adversarial_top1[0],
                "adversarial_top1_label": adversarial_top1[1],
                "adversarial_top1_confidence": adversarial_top1[2],
                "target_index": target_index,
                "target_label": target_name,
                "success": adversarial_top1[0] == target_index,
            }
        )

    write_summary(output_dir / "summary.csv", rows)
    successes = sum(1 for row in rows if row["success"])
    print(f"Done. Successes: {successes}/{len(rows)}")
    print(f"Generated adversarial images: {adversarial_dir}")
    print(f"Prediction cards: {card_dir}")
    print(f"Summary CSV: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
