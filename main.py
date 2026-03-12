import numpy as np
import rawpy
import matplotlib.pyplot as plt

# Virtual ISP Pipeline
# 1. Decode & extract data
# 2. Linearize
# 3. Create label map
# 4. Demosaic
# 5. White balance
# 6. Color Space conversion
# 7. Gamma / tone mapping


# Post Process a RAW image
def display_arw_image(file_path):
    with rawpy.imread(file_path) as raw:
        rgb = raw.postprocess()  # convert sensor data into a standard RGB image
    plt.imshow(rgb)
    plt.title('ARW Image')
    plt.axis('off')
    plt.show()

# 1.) Decode & extract data


def decode_arw_image(file_path):
    with rawpy.imread(file_path) as raw:
        bayer = raw.raw_image_visible.copy()  # 2D Bayer mosaic (decoded RAW data)
        cfa = raw.raw_pattern.copy()  # CFA layout, [[0,1], [1,2]]
        black = np.array(raw.black_level_per_channel)
        white = raw.white_level
        color_desc = raw.color_desc
        white_balance_multipliers = raw.camera_whitebalance

    return bayer, cfa, black, white, color_desc, white_balance_multipliers

# 2.) Find linear


def linearize_bayer(bayer, black_level, white_level):
    black_scalar = float(black_level[0])
    linear = (bayer.astype(np.float32) - black_scalar) / \
        (white_level - black_scalar)
    return np.clip(linear, 0.0, 1.0)  # limit linear values between 0 and 1


# 3.) Create label map
def build_rgb_masks(bayer, cfa, color_desc):
    # get image height and width
    h, w = bayer.shape

    # a.) repeat 2x2 CFA tile to full image size
    full_ids = np.tile(cfa, (h // 2 + 1, w // 2 + 1))[:h, :w]

    # b.) Map CFA indices to channel names using color_desc, then create per-channel boolean masks
    red_mask = (full_ids == color_desc.index(b'R'))
    green_mask = (full_ids == color_desc.index(b'G'))
    blue_mask = (full_ids == color_desc.index(b'B'))
    return red_mask, green_mask, blue_mask

# 4.) Demosaic - Bilinear


def demosaic_bilinear(linear_bayer, red_mask, green_mask, blue_mask):

    # create a blank red channel
    h, w = linear_bayer.shape

    R = np.zeros((h, w), dtype=np.float32)
    R[red_mask] = linear_bayer[red_mask]
    # interpolate red channel
    R = interpolate_channel(R, red_mask)

    # repeat for green

    G = np.zeros((h, w), dtype=np.float32)
    G[green_mask] = linear_bayer[green_mask]
    G = interpolate_channel(G, green_mask)

    # repeat for blue

    B = np.zeros((h, w), dtype=np.float32)
    B[blue_mask] = linear_bayer[blue_mask]
    B = interpolate_channel(B, blue_mask)

    rgb_linear = np.stack([R, G, B], axis=-1)

    return rgb_linear


def interpolate_channel(values, known_mask):
    out = values.copy()
    h, w = values.shape

    neighbor_sum = np.zeros_like(values)
    neighbor_count = np.zeros_like(values)

    neighbor_sum[1:, :] += values[:-1, :]  # value from the top neighbor
    neighbor_sum[:-1, :] += values[1:, :]  # value from the bottom neighbor
    neighbor_sum[:, 1:] += values[:, :-1]  # value from left neighbor
    neighbor_sum[:, :-1] += values[:, 1:]  # value from right neighbor

    # need to convert known_mask to float
    mask = known_mask.astype(np.float32)

    neighbor_count[1:, :] += mask[:-1, :]
    neighbor_count[:-1, :] += mask[1:, :]
    neighbor_count[:, 1:] += mask[:, :-1]
    neighbor_count[:, :-1] += mask[:, 1:]

    # now we need to find the avg
    # first we need to ensure count > 0

    safe_count = np.where(neighbor_count > 0, neighbor_count, 1)
    avg = neighbor_sum / safe_count

    return out


def normalize_white_balance(wb_mult):

    red_balance, green_balance, blue_balance = wb_mult[0], wb_mult[1], wb_mult[2]

    # Scale balance based on green
    r_norm = red_balance / green_balance
    g_norm = green_balance / green_balance
    b_norm = blue_balance / green_balance

    n_wb = [r_norm, g_norm, b_norm]

    n_wb = np.round(n_wb, decimals=2)

    print(n_wb)

    return n_wb


def apply_white_balance(rgb_lienar, n_wb):
    # setup function to apply WB

    return


bayer, cfa, black, white, color_desc, whitebalance_mult = decode_arw_image(
    '.\imgs\AKG02229.ARW')
# each pixel is a sensor intensity fraction
linear = linearize_bayer(bayer, black, white)

# Create RGB masks
red_mask, green_mask, blue_mask = build_rgb_masks(bayer, cfa, color_desc)

# Create RGB linear
rgb_linear = demosaic_bilinear(linear, red_mask, green_mask, blue_mask)


# testing correct wb normalization:
normalize_white_balance(whitebalance_mult)
