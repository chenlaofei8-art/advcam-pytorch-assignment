from __future__ import annotations

from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFilter


#生成一张简单的抽象风格图
def main() -> None:
    random.seed(7)
    out = Path("data/style/style.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)

    size = 512
    image = Image.new("RGB", (size, size), (235, 235, 230))
    draw = ImageDraw.Draw(image, "RGBA")

    palette = [
        (36, 92, 140, 95),
        (218, 91, 64, 90),
        (244, 183, 74, 80),
        (66, 145, 105, 85),
        (116, 78, 145, 70),
    ]

    for _ in range(180):
        x = random.randint(-80, size)
        y = random.randint(-80, size)
        w = random.randint(40, 180)
        h = random.randint(20, 140)
        color = random.choice(palette)
        draw.ellipse((x, y, x + w, y + h), fill=color)

    for _ in range(45):
        points = []
        start_x = random.randint(0, size)
        start_y = random.randint(0, size)
        for i in range(8):
            points.append((start_x + i * 30, start_y + random.randint(-60, 60)))
        draw.line(points, fill=random.choice(palette), width=random.randint(4, 12))

    image = image.filter(ImageFilter.GaussianBlur(radius=2.0))
    image.save(out, quality=95)
    print(f"Created demo style image: {out}")


if __name__ == "__main__":
    main()
