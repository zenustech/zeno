# TODO: use multiprocessing for video encoding
from PIL import Image
from io import BytesIO


def encode(img, width, height):
    im = Image.new('RGB', (width, height))
    im.frombytes(img)
    with BytesIO() as f:
        im.save(f, 'jpeg')
        im = f.getvalue()
    return im
