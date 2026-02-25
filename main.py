import numpy as np
import rawpy
import matplotlib.pyplot as plt

# Post Process a RAW image


def display_arw_image(file_path):
    with rawpy.imread(file_path) as raw:
        rgb = raw.postprocess()  # convert sensor data into a standard RGB image
    plt.imshow(rgb)
    plt.title('ARW Image')
    plt.axis('off')
    plt.show()

# Decode & extract data


def decode_arw_image(file_path):
    with rawpy.imread(file_path) as raw:
        bayer = raw.raw_image_visible.copy()  # 2D Bayer mosaic (decoded RAW data)
        cfa = raw.raw_pattern.copy()  # CFA layout, [[0,1], [1,2]]
        black = np.array(raw.black_level_per_channel)
        white = raw.white_level
        color_desc = raw.color_desc

    # print("bayer shape:", bayer.shape, "dtype:", bayer.dtype)
    # print("cfa pattern:\n", cfa)
    # print("black levels:", black, "white level:", white)
    return bayer, cfa, black, white, color_desc


def linearize_bayer(bayer, black_level, white_level):
    black_scalar = float(black_level[0])
    linear = (bayer.astype(np.float32) - black_scalar) / \
        (white_level - black_scalar)
    return np.clip(linear, 0.0, 1.0) # limit linear values between 0 and 1


bayer, cfa, black, white, color_desc = decode_arw_image('.\imgs\AKG02229.ARW')
linear = linearize_bayer(bayer, black, white) # each pixel is a sensor intensity fraction



