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


# Test image:
display_arw_image('.\imgs\AKG02229.ARW')
