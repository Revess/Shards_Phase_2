from PIL import Image, ImageOps,ImageDraw
import numpy as np
import os,sys
import matplotlib.pyplot as plt
from math import sqrt, ceil

directory = os.path.join('..','..','data','datasets','ImprovedShardDrawrings','output','holwerda140-142')
index = 1
plt.figure(figsize=(16, 16))

for image in os.listdir(directory):
    im = Image.open(os.path.join(directory,image)).convert('L')
    im = ImageOps.invert(im)
    ax = plt.subplot(10, 12, index)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.imshow(im, cmap="Greys",interpolation='nearest',vmin=0,vmax=255)
    plt.title(str(image))
    index+=1
plt.tight_layout()
plt.show()