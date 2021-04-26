# TODO: use multiprocessing for video encoding
from PIL import Image
from io import BytesIO


def start():
    return 'uri'


def push(img, width, height):
    im = Image.new('RGB', (width, height))
    im.frombytes(img)
    with BytesIO() as f:
        im.save(f, 'jpeg')
        im = f.getvalue()
    #print(im)


def stop():
    pass
