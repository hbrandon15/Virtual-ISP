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

# -- DECODE AND EXTRACT DATA --


def decode_arw_image(file_path):
    with rawpy.imread(file_path) as raw:
        bayer = raw.raw_image_visible.copy()  # 2D Bayer mosaic (decoded RAW data)
        cfa = raw.raw_pattern.copy()  # CFA layout, [[0,1], [1,2]]
        black = np.array(raw.black_level_per_channel)
        white = raw.white_level
        color_desc = raw.color_desc
        white_balance_multipliers = raw.camera_whitebalance

        # !!color_matrix is often empty for Sony files!!
        # color_correction_matrix = raw.color_matrix
        # remove empty 4th row
        color_correction_matrix = raw.rgb_xyz_matrix[:3, :]

        # print(color_correction_matrix)

    return bayer, cfa, black, white, color_desc, white_balance_multipliers, color_correction_matrix

# -- FIND LINEAR --


def linearize_bayer(bayer, black_level, white_level):
    black_scalar = float(black_level[0])
    linear = (bayer.astype(np.float32) - black_scalar) / \
        (white_level - black_scalar)
    return np.clip(linear, 0.0, 1.0)  # limit linear values between 0 and 1


# -- CREATE LABEL MAP --


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

# -- DEMOSAIC - BILINEAR --


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

    out[~known_mask] = avg[~known_mask]

    return out

# -- CORRECT WHITE BALANCE --


def normalize_white_balance(wb_mult):

    red_balance, green_balance, blue_balance = wb_mult[0], wb_mult[1], wb_mult[2]

    # Scale balance based on green
    r_norm = red_balance / green_balance
    g_norm = green_balance / green_balance
    b_norm = blue_balance / green_balance

    n_wb = [r_norm, g_norm, b_norm]

    n_wb = np.round(n_wb, decimals=2)

    return n_wb


def apply_white_balance(rgb_linear, gains):
    # return result

    red_channel = rgb_linear[:, :, 0]
    green_channel = rgb_linear[:, :, 1]
    blue_channel = rgb_linear[:, :, 2]

    # multiply each channel by its gain
    red_corrected = red_channel * gains[0]
    green_corrected = green_channel * gains[1]
    blue_corrected = blue_channel * gains[2]

    # return clip to [0,1]
    rgb_wb = np.stack(
        [red_corrected, green_corrected, blue_corrected], axis=-1)

    return np.clip(rgb_wb, 0.0, 1.0)

# -- CORRECT COLOR SPACE --


def color_space_conversion(ccm, rgb_wb):
    h, w, c = rgb_wb.shape
    # first_pixel = rgb_wb[0, 0, :]  # shape (3,)
    # rgb_values = first_pixel.reshape(3, 1)  # new shape (3,1)

    # 1. Flatten the image to a list of RGB triplets
    # reshape(-1,3) will stack our data into rows for processing.
    pixels = rgb_wb.reshape(-1, 3)

    # 2. Apply CCM. We use .T (transpose) because our pixels are now rows
    # result shape will be (N,3)
    corrected_pixels = pixels @ ccm.T

    # 3. Put back into the image shape (H,W,3)
    result = corrected_pixels.reshape(h, w, 3)

    return np.clip(result, 0.0, 1.0)

# -- GAMMA & TONE MAPPING --


def apply_srgb_gamma(rgb_linear):

    # Apply sRGB transfer function
    rgb = np.where(rgb_linear <= 0.0031308,  # condition 
                   12.92 * rgb_linear, # if TRUE
                   1.055 * np.power(rgb_linear, 1.0 / 2.4) - 0.055 ) # if FALSE


    return np.clip(rgb, 0.0, 1.0)


# STEP 1: OBTAIN METADATA
bayer, cfa, black, white, color_desc, whitebalance_mult, color_correction_matrix = decode_arw_image(
    '.\imgs\AKG02229.ARW')
# STEP 2: OBTAIN LINEAR - each pixel is a sensor intensity fraction
linear = linearize_bayer(bayer, black, white)

# STEP 3: CREATE LABEL MAP - Create RGB masks
red_mask, green_mask, blue_mask = build_rgb_masks(bayer, cfa, color_desc)

# STEP 4: DEMOSAIC
# Create RGB linear
rgb_linear = demosaic_bilinear(linear, red_mask, green_mask, blue_mask)


# STEP 5: WHITE BALANCE
wb_gains = normalize_white_balance(whitebalance_mult)
# apply WB
rgb_wb = apply_white_balance(rgb_linear, wb_gains)

# STEP 6: COLOR CORRECTION
rgb_ccm = color_space_conversion(color_correction_matrix, rgb_wb)

# STEP 7: GAMMA AND TONE MAPPING
rgb_gamma = apply_srgb_gamma(rgb_ccm)

# CONVERT TO UINT8 FOR DISPLAY

