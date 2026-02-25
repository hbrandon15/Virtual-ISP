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

    print("bayer shape:", bayer.shape, "dtype:", bayer.dtype)
    print("cfa pattern:\n", cfa)
    print("black levels:", black, "white level:", white)

    # compute linear sensor values
    # convert raw sensor codes into a clean 0-1 signal
    lin = np.clip((bayer - black) / (white - black), 0, 1) 
    
	# build color masks
    


decode_arw_image('.\imgs\AKG02229.ARW')
