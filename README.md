# Virtual ISP

A hands-on project to build a simplified Image Signal Processing (ISP) pipeline from scratch using only NumPy for pixel manipulation. This project avoids high-level OpenCV functions to provide a deeper understanding of the underlying image processing steps.

## Project Workflow

1. **Find a RAW image:**
   - Download a `.dng` or `.arw` file (common in digital photography).
2. **Demosaicing:**
   - Implement an algorithm to convert the Bayer pattern (RGGB) into a full RGB image using NumPy.
3. **White Balance:**
   - Apply white balance (e.g., for 5500K daylight) by scaling the Red and Blue channels to neutralize color casts.
4. **Gamma Correction:**
   - Apply a non-linear gamma curve to the data for natural-looking output.

## Getting Started

- Python 3.x
- NumPy

## Goals

- Learn the fundamentals of digital image processing.
- Gain hands-on experience with RAW image data.
- Understand and implement demosaicing, white balance, and gamma correction from scratch.

## License

MIT License
