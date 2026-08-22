"""
Visual Coordinate Grid Overlay for FRIDAY.
Draws a subtle 0-1000 normalized coordinate grid with tick marks every 100 units
over full-screen screenshots to eliminate spatial coordinate hallucinations in VLMs.
"""

from PIL import Image, ImageDraw, ImageFont


def draw_visual_coordinate_grid(image: Image.Image) -> Image.Image:
    """
    Overlays an unobtrusive normalized coordinate grid (0-1000)
    with tick marks every 100 units to eliminate spatial coordinate hallucinations.
    """
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = base.size
    grid_color = (255, 255, 255, 40)       # 15% opacity white line
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
