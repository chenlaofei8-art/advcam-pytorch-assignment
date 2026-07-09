# Adversarial Examples Using Style Transfer

This project implements a simplified PyTorch version of the AdvCam idea:
generate adversarial examples that are visually stylized while fooling a
pre-trained ImageNet VGG19 classifier.

The VGG19 model is kept fixed. Only the input image is optimized.

## Method

For each content image, the script optimizes an adversarial image with three
main losses:

- `L_adv`: targeted adversarial loss, encouraging VGG19 to classify the image
  as `cinema / movie theater`.
- `L_style`: style-transfer loss, encouraging the generated image to match the
  chosen style image.
- `L_content`: content loss, encouraging the generated image to preserve the
  original image content.

The total loss is:

```text
L_total = lambda_adv * L_adv + alpha * L_style + beta * L_content + tv_weight * L_tv
```

`L_tv` is a small smoothness term. This implementation focuses on the digital
setting and does not include physical-world EOT transformations.

## Install

Create the Python environment and install dependencies once:

```bash
bash scripts/setup_env.sh
```

For later terminal sessions, activate the environment with:

```bash
source .venv/bin/activate
```

The first run may download the ImageNet-pretrained VGG19 weights from
TorchVision.

## Prepare Data

Put 10 ImageNet images in:

```text
data/content/
```

Put one style image at:

```text
data/style/style.jpg
```

A simple abstract style image can be created with:

```bash
python scripts/create_demo_style.py
```

## Run

Run the attack on the 10 images in `data/content/`:

```bash
python run_attack.py --num-images 10 --target-label cinema --steps 400
```

If some images are not classified as the target class, try increasing the attack
strength:

```bash
python run_attack.py --num-images 10 --target-label cinema --steps 700 --lambda-adv 8.0 --alpha-style 50000
```

## Outputs

The script writes:

```text
results/
  adversarial_images/      generated adversarial images
  original_images/         resized original images used by the script
  prediction_cards/        screenshot-like PNGs with before/after predictions
  logs/                    loss history for each image
  style_reference.png      resized style reference
  summary.csv              prediction summary
```

The `prediction_cards/` images can be included as screenshots of the predicted
labels.

## Notes

- This code uses the ImageNet-pretrained TorchVision VGG19 model.
- The target is resolved from TorchVision's ImageNet labels by searching for
  `cinema`. In standard ImageNet-1K labels, the target index is 498.
- The attack is stochastic only in minor implementation details; results may
  differ slightly across devices.
- For best speed, use CUDA or Apple Silicon MPS if available.
