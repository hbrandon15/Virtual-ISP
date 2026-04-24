import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# -- PIPELINE DIAGRAM --
# Generates a flowchart of the Virtual ISP pipeline from light source to final image.

STEPS = [
    {
        "label": "Light",
        "sublabel": "Photons hit the sensor",
        "color": "#FFF176",
        "text_color": "#333333",
    },
    {
        "label": "Camera Sensor",
        "sublabel": "Bayer mosaic — each pixel captures\none color channel (R, G, or B)",
        "color": "#FFCC80",
        "text_color": "#333333",
    },
    {
        "label": "1. Decode & Extract",
        "sublabel": "Read RAW bayer array, CFA layout,\nblack/white levels, WB multipliers",
        "color": "#90CAF9",
        "text_color": "#1a237e",
    },
    {
        "label": "2. Linearize",
        "sublabel": "Subtract black level, normalize to [0, 1]\nusing white level",
        "color": "#90CAF9",
        "text_color": "#1a237e",
    },
    {
        "label": "3. Label Map",
        "sublabel": "Tile 2×2 CFA pattern to full image size\nBuild boolean masks for R, G, B channels",
        "color": "#90CAF9",
        "text_color": "#1a237e",
    },
    {
        "label": "4. Demosaic (Bilinear)",
        "sublabel": "Interpolate missing color values per pixel\nusing nearest-neighbor averaging",
        "color": "#A5D6A7",
        "text_color": "#1b5e20",
    },
    {
        "label": "5. White Balance",
        "sublabel": "Normalize WB multipliers to green channel\nScale each RGB channel by its gain",
        "color": "#A5D6A7",
        "text_color": "#1b5e20",
    },
    {
        "label": "6. Color Space Conversion",
        "sublabel": "Apply 3×3 CCM to each pixel\n(identity matrix — see README)",
        "color": "#A5D6A7",
        "text_color": "#1b5e20",
    },
    {
        "label": "7. Gamma / Tone Mapping",
        "sublabel": "Apply sRGB transfer function\n(linear below 0.0031308, power curve above)",
        "color": "#CE93D8",
        "text_color": "#4a148c",
    },
    {
        "label": "Final RGB Image",
        "sublabel": "Convert to uint8 [0, 255]\nDisplay and save",
        "color": "#EF9A9A",
        "text_color": "#b71c1c",
    },
]

BOX_WIDTH = 4.2
BOX_HEIGHT = 0.85
STEP_GAP = 1.18
ARROW_GAP = 0.12
FIG_WIDTH = 7
FIG_HEIGHT = len(STEPS) * STEP_GAP + 1.2


def draw_box(ax, cx, cy, step):
    box = FancyBboxPatch(
        (cx - BOX_WIDTH / 2, cy - BOX_HEIGHT / 2),
        BOX_WIDTH, BOX_HEIGHT,
        boxstyle="round,pad=0.08",
        facecolor=step["color"],
        edgecolor="#888888",
        linewidth=1.2,
        zorder=2,
    )
    ax.add_patch(box)

    ax.text(cx, cy + 0.17, step["label"],
            ha="center", va="center",
            fontsize=10, fontweight="bold",
            color=step["text_color"], zorder=3)

    ax.text(cx, cy - 0.18, step["sublabel"],
            ha="center", va="center",
            fontsize=7.5, color="#555555",
            linespacing=1.4, zorder=3)


def draw_arrow(ax, cx, top_y, bottom_y):
    ax.annotate(
        "", xy=(cx, bottom_y + BOX_HEIGHT / 2 + ARROW_GAP),
        xytext=(cx, top_y - BOX_HEIGHT / 2 - ARROW_GAP),
        arrowprops=dict(arrowstyle="-|>", color="#666666", lw=1.4),
        zorder=1,
    )


fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
ax.set_xlim(0, FIG_WIDTH)
ax.set_ylim(0, FIG_HEIGHT)
ax.axis("off")

ax.set_facecolor("#F9F9F9")
fig.patch.set_facecolor("#F9F9F9")

cx = FIG_WIDTH / 2
top_padding = 0.7

for i, step in enumerate(STEPS):
    cy = FIG_HEIGHT - top_padding - i * STEP_GAP
    draw_box(ax, cx, cy, step)

    if i < len(STEPS) - 1:
        next_cy = FIG_HEIGHT - top_padding - (i + 1) * STEP_GAP
        draw_arrow(ax, cx, cy, next_cy)

ax.text(cx, FIG_HEIGHT - 0.25, "Virtual ISP Pipeline",
        ha="center", va="top",
        fontsize=13, fontweight="bold", color="#222222")

legend_items = [
    mpatches.Patch(color="#FFF176", label="Input"),
    mpatches.Patch(color="#90CAF9", label="Raw Processing"),
    mpatches.Patch(color="#A5D6A7", label="Color Correction"),
    mpatches.Patch(color="#CE93D8", label="Tone Mapping"),
    mpatches.Patch(color="#EF9A9A", label="Output"),
]
ax.legend(handles=legend_items, loc="lower right",
          fontsize=8, framealpha=0.85, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("pipeline_diagram.png", dpi=150, bbox_inches="tight", pad_inches=0.2)
plt.show()
