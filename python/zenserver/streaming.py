# TODO: use multiprocessing for video encoding
from PIL import Image
from io import BytesIO
import numpy as np


def encode(img, width, height, scale=2, quality=70):
    if scale != 1:
        img = np.frombuffer(img, dtype=np.uint8).reshape(height, width, 3)
        #tl = img[:-1:scale, :-1:scale] >> np.uint8(2)
        #tr = img[:-1:scale, +1::scale] >> np.uint8(2)
        #bl = img[:+1:scale, :-1:scale] >> np.uint8(2)
        #br = img[+1::scale, +1::scale] >> np.uint8(2)
        #img = tl + tr + bl + br
        img = img[::scale, ::scale]
        im = Image.fromarray(img)
    else:
        im = Image.new('RGB', (width, height))
        im.frombytes(img)
    with BytesIO() as f:
        im.save(f, 'jpeg', quality=quality)
        im = f.getvalue()
    return im
