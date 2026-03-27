# Virtual ISP

A hands-on project to build a simplified Image Signal Processing (ISP) pipeline from scratch. This project implements each pipeline stage manually using NumPy for pixel manipulation, with `rawpy` for RAW file decoding and `matplotlib` for display.

## Pipeline

The pipeline processes a RAW `.ARW` (Sony) file through the following stages:

1. **Decode & Extract** — Read the Bayer mosaic, CFA layout, black/white levels, white balance multipliers, and the camera-to-XYZ color matrix using `rawpy`.
2. **Linearize** — Subtract black level and normalize to `[0, 1]` using the white level.
3. **Label Map** — Tile the 2×2 CFA pattern to full image size and build per-channel boolean masks (R, G, B).
4. **Demosaic (Bilinear)** — Interpolate missing color values at each pixel position using nearest-neighbor averaging via vectorized NumPy shifts.
5. **White Balance** — Normalize camera white balance multipliers relative to green, then scale each RGB channel.
6. **Color Space Conversion** — Apply a 3×3 Color Correction Matrix (CCM) to each pixel. Currently uses an identity matrix; cam-to-XYZ and XYZ-to-sRGB matrices are extracted but not yet composed.
7. **Gamma / Tone Mapping** — Apply the sRGB transfer function (linear below 0.0031308, power curve above).

## Dependencies

- Python 3.x
- NumPy
- rawpy
- matplotlib

## Usage

Place a `.ARW` file at `./imgs/AKG02229.ARW` and run:

```bash
python main.py
```

## Goals

- Learn the fundamentals of digital image processing.
- Gain hands-on experience with RAW image data.
- Understand and implement each ISP stage from scratch.

## License

MIT License
