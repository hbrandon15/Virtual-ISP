import numpy as np
import rawpy
import matplotlib.pyplot as plt

print("hello world!")


def display_arw_image(file_path):
    with rawpy.imread(file_path) as raw:

        rgb = raw.postprocess()  # convert sensor data into a standard RGB image
        # (includes: demosaicing, WB, basic color correction)
    plt.imshow(rgb)
    plt.title('ARW Image')
    plt.axis('off')
    plt.show()


# Example usage:
display_arw_image('.\imgs\AKG02229.ARW')
