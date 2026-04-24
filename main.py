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


# -- PREVIEW --


def display_arw_image(file_path: str) -> None:
    """
    This function is only intended to preview a processed image using rawpy by converting sensor data into a standard RGB image.
    """
    with rawpy.imread(file_path) as raw:
        rgb = raw.postprocess()
    plt.imshow(rgb)
    plt.title('ARW Image')
    plt.axis('off')
    plt.show()


# -- DECODE AND EXTRACT DATA --


def decode_arw_image(file_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bytes, list, np.ndarray]:
    """
    Extracts raw sensor data and metadata from an ARW file.

    xyz_to_srgb @ cam_to_xyz is unusable standalone — libraw's rgb_xyz_matrix
    row sums: R=-0.47 (clips to 0), G=1.51, B=0.57 — severe green cast.
    Requires libraw's internal WB pre-multipliers and normalization to balance.
    A proper CCM needs either dcraw's hardcoded camera matrices or color checker calibration.
    See README for details.
    """
    with rawpy.imread(file_path) as raw:
        bayer = raw.raw_image_visible.copy()  # 2D Bayer mosaic (decoded RAW data)
        cfa = raw.raw_pattern.copy()          # CFA layout, [[0,1], [1,2]]
        black = np.array(raw.black_level_per_channel)
        white = raw.white_level
        color_desc = raw.color_desc
        white_balance_multipliers = raw.camera_whitebalance

        # !!color_matrix is often empty for Sony files!!
        # color_correction_matrix = raw.color_matrix
        # cam_to_xyz = raw.rgb_xyz_matrix[:3, :]  — unusable without dcraw matrices or color checker calibration

        ccm = np.eye(3)

    return bayer, cfa, black, white, color_desc, white_balance_multipliers, ccm


# -- LINEARIZE --


def linearize_bayer(bayer: np.ndarray, black_level: np.ndarray, white_level: int) -> np.ndarray:
    """
    Black level - baseline value the sensor reports in complete darkness.
    White level - saturation point. Any photon count above this level clips to the same max value.
    """
    black_scalar = float(black_level[0])
    linear = (bayer.astype(np.float32) - black_scalar) / \
        (white_level - black_scalar)

    return np.clip(linear, 0.0, 1.0)


# -- CREATE LABEL MAP --


def build_rgb_masks(bayer: np.ndarray, cfa: np.ndarray, color_desc: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a full array of a repeated (tile) color filter array the size of our original image.
    Map the color filter array to each color channel name using the incoming color_desc. All color channel masks will be a Boolean type. For the Red channel, if a pixel is red -> True.
    """
    h, w = bayer.shape

    full_ids = np.tile(cfa, (h // 2 + 1, w // 2 + 1))[:h, :w]

    red_mask = (full_ids == color_desc.index(b'R'))
    green_mask = (full_ids == color_desc.index(b'G'))
    blue_mask = (full_ids == color_desc.index(b'B'))

    return red_mask, green_mask, blue_mask


# -- DEMOSAIC --


def demosaic_bilinear(linear_bayer: np.ndarray, red_mask: np.ndarray, green_mask: np.ndarray, blue_mask: np.ndarray) -> np.ndarray:
    """
    Reconstructing incomplete color sample output from an image sensor overlaid with a color filter array into a full color image.
    Bilinear - estimating each missing pixel in both horizontal and vertical directions.
    """
    h, w = linear_bayer.shape

    R = np.zeros((h, w), dtype=np.float32)
    R[red_mask] = linear_bayer[red_mask]
    R = interpolate_channel(R, red_mask)

    G = np.zeros((h, w), dtype=np.float32)
    G[green_mask] = linear_bayer[green_mask]
    G = interpolate_channel(G, green_mask)

    B = np.zeros((h, w), dtype=np.float32)
    B[blue_mask] = linear_bayer[blue_mask]
    B = interpolate_channel(B, blue_mask)

    rgb_linear = np.stack([R, G, B], axis=-1)

    return rgb_linear


def interpolate_channel(values: np.ndarray, known_mask: np.ndarray) -> np.ndarray:
    """
    values is a full (h,w) float32 array with mostly zeros, with actual color values at their placed locations.
    known_mask is a boolean mask (red, green, blue) where it is only True where the selected color actually exists.
    We are looking at the same color of pixels nearby each and taking the average.
    """
    out = values.copy()

    neighbor_sum = np.zeros_like(values)
    neighbor_count = np.zeros_like(values)

    neighbor_sum[1:, :] += values[:-1, :]   # value from the top neighbor
    neighbor_sum[:-1, :] += values[1:, :]   # value from the bottom neighbor
    neighbor_sum[:, 1:] += values[:, :-1]   # value from left neighbor
    neighbor_sum[:, :-1] += values[:, 1:]   # value from right neighbor

    mask = known_mask.astype(np.float32)

    neighbor_count[1:, :] += mask[:-1, :]
    neighbor_count[:-1, :] += mask[1:, :]
    neighbor_count[:, 1:] += mask[:, :-1]
    neighbor_count[:, :-1] += mask[:, 1:]

    safe_count = np.where(neighbor_count > 0, neighbor_count, 1)
    avg = neighbor_sum / safe_count

    out[~known_mask] = avg[~known_mask]

    return out


# -- WHITE BALANCE --


def normalize_white_balance(wb_mult: list) -> np.ndarray:
    """
    Normalize camera white balance multipliers relative to the green channel.
    """
    red_balance, green_balance, blue_balance = wb_mult[0], wb_mult[1], wb_mult[2]

    r_norm = red_balance / green_balance
    g_norm = green_balance / green_balance
    b_norm = blue_balance / green_balance

    n_wb = np.round([r_norm, g_norm, b_norm], decimals=2)

    return n_wb


def apply_white_balance(rgb_linear: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """
    Multiply each RGB channel by its corresponding white balance gain and clip to [0, 1].
    """
    red_corrected = rgb_linear[:, :, 0] * gains[0]
    green_corrected = rgb_linear[:, :, 1] * gains[1]
    blue_corrected = rgb_linear[:, :, 2] * gains[2]

    rgb_wb = np.stack([red_corrected, green_corrected, blue_corrected], axis=-1)

    return np.clip(rgb_wb, 0.0, 1.0)


# -- COLOR SPACE CONVERSION --


def color_space_conversion(ccm: np.ndarray, rgb_wb: np.ndarray) -> np.ndarray:
    """
    1. Flatten the image to a list of RGB triplets.
    2. Apply CCM. We use .T (transpose) because our pixels are now rows.
    """
    h, w, c = rgb_wb.shape

    pixels = rgb_wb.reshape(-1, 3)           # reshape(-1,3) stacks data into rows for processing
    corrected_pixels = pixels @ ccm.T        # result shape will be (N,3)
    result = corrected_pixels.reshape(h, w, 3)

    return np.clip(result, 0.0, 1.0)


# -- GAMMA AND TONE MAPPING --


def apply_srgb_gamma(rgb_linear: np.ndarray) -> np.ndarray:
    """
    Apply the sRGB transfer function (gamma encoding) to a linear light image.
    """
    rgb = np.where(rgb_linear <= 0.0031308,
                   12.92 * rgb_linear,
                   1.055 * np.power(rgb_linear, 1.0 / 2.4) - 0.055)

    return np.clip(rgb, 0.0, 1.0)


def main():
    # STEP 1: OBTAIN METADATA
    bayer, cfa, black, white, color_desc, whitebalance_mult, ccm = decode_arw_image(
        '.\imgs\AKG02229.ARW')

    # STEP 2: LINEARIZE - normalize each pixel to a sensor intensity fraction
    linear = linearize_bayer(bayer, black, white)

    # STEP 3: CREATE LABEL MAP - create boolean masks per color channel
    red_mask, green_mask, blue_mask = build_rgb_masks(bayer, cfa, color_desc)

    # STEP 4: DEMOSAIC - reconstruct full RGB image from Bayer mosaic
    rgb_linear = demosaic_bilinear(linear, red_mask, green_mask, blue_mask)

    # STEP 5: WHITE BALANCE - normalize channel gains relative to green
    wb_gains = normalize_white_balance(whitebalance_mult)
    rgb_wb = apply_white_balance(rgb_linear, wb_gains)

    # STEP 6: COLOR CORRECTION (identity matrix — see decode_arw_image docstring)
    rgb_ccm = color_space_conversion(ccm, rgb_wb)

    # STEP 7: GAMMA AND TONE MAPPING - apply sRGB transfer function
    rgb_gamma = apply_srgb_gamma(rgb_ccm)

    # CONVERT TO UINT8 FOR DISPLAY
    rgb_display = (rgb_gamma * 255).astype(np.uint8)

    plt.imshow(np.rot90(rgb_display))
    plt.title('Final ISP Output')
    plt.axis('off')
    plt.savefig("isp_output.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == "__main__":
    main()
