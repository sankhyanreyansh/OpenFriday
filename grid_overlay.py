"""
Visual Coordinate Grid & Set-of-Marks (SoM) Overlay for FRIDAY.
Draws subtle 0-1000 normalized coordinate grid lines and high-contrast numbered
Set-of-Marks (SoM) element badges directly over screenshots for 100% VLM grounding accuracy.
"""

from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont
from accessibility_tree import UIElementInfo


def draw_visual_coordinate_grid(image: Image.Image) -> Image.Image:
    """
    Overlays an unobtrusive normalized coordinate grid (0-1000)
    with tick marks every 100 units to eliminate spatial coordinate hallucinations.
    """
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = base.size
    grid_color = (255, 255, 255, 36)       # Subtle translucent grid line
    text_color = (255, 255, 255, 140)      # Muted text label
    font = ImageFont.load_default()

    # Draw vertical grid lines & X-axis labels (0 to 1000)
    for i in range(1, 10):
        norm_val = i * 100
        x = int((norm_val / 1000.0) * w)
        draw.line([(x, 0), (x, h)], fill=grid_color, width=1)
        draw.text((x + 3, 4), str(norm_val), fill=text_color, font=font)

    # Draw horizontal grid lines & Y-axis labels (0 to 1000)
    for i in range(1, 10):
        norm_val = i * 100
        y = int((norm_val / 1000.0) * h)
        draw.line([(0, y), (w, y)], fill=grid_color, width=1)
        draw.text((4, y + 2), str(norm_val), fill=text_color, font=font)

    return Image.alpha_composite(base, overlay).convert("RGB")


def draw_set_of_marks_overlay(
    image: Image.Image,
    elements: List[UIElementInfo],
    logical_w: int,
    logical_h: int,
    apply_grid: bool = True
) -> Image.Image:
    """
    Renders high-contrast numbered Set-of-Marks (SoM) badges over interactive UI elements
    and optionally overlays the 0-1000 coordinate grid.
    Handles Retina display scaling seamlessly.
    """
    base = image.convert("RGBA")
    if apply_grid:
        base = draw_visual_coordinate_grid(base).convert("RGBA")

    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    img_w, img_h = base.size
    scale_x = float(img_w) / float(max(1, logical_w))
    scale_y = float(img_h) / float(max(1, logical_h))

    font = ImageFont.load_default()

    # Color palette for SoM badges
    badge_bg = (124, 58, 237, 230)        # Vibrant Purple-Violet
    badge_border = (255, 255, 255, 240)    # Crisp White Border
    badge_text = (255, 255, 255, 255)      # White text
    box_border = (139, 92, 246, 120)       # Translucent element bounding box
    box_fill = (139, 92, 246, 25)          # Very subtle highlight fill

    for elem in elements:
        x, y, w, h = elem.bounds

        # Map logical coordinates to physical image pixels
        px = int(round(x * scale_x))
        py = int(round(y * scale_y))
        pw = int(round(w * scale_x))
        ph = int(round(h * scale_y))

        if pw < 4 or ph < 4:
            continue

        # 1. Subtle element bounding box outline
        draw.rectangle([px, py, px + pw, py + ph], outline=box_border, fill=box_fill, width=1)

        # 2. Numbered Pill Badge (e.g. "[1]")
        tag_str = str(elem.element_id)
        tag_w = max(18, len(tag_str) * 8 + 10)
        tag_h = 16

        # Position badge at top-left of the element (clamp within image boundaries)
        bx = max(2, min(px, img_w - tag_w - 2))
        by = max(2, min(py - tag_h - 2 if py >= tag_h + 4 else py + 2, img_h - tag_h - 2))

        # Badge shadow
        draw.rounded_rectangle([bx + 1, by + 1, bx + tag_w + 1, by + tag_h + 1], radius=4, fill=(0, 0, 0, 160))
        # Badge background
        draw.rounded_rectangle([bx, by, bx + tag_w, by + tag_h], radius=4, fill=badge_bg, outline=badge_border, width=1)
        # Badge text centered
        draw.text((bx + (tag_w - len(tag_str) * 6) // 2, by + 2), tag_str, fill=badge_text, font=font)

    return Image.alpha_composite(base, overlay).convert("RGB")
