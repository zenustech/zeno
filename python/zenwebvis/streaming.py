from PIL import Image
from io import BytesIO
import numpy as np
import taichi as ti


g_img = None
g_res = (0, 0)


def decode(img):
    global g_img, g_res
    with BytesIO(img) as f:
        im = Image.open(f)
        g_res = im.width, im.height
        g_img = im.tobytes()


gui = ti.GUI()
def paint():
    if g_img is None:
        return
    img = np.frombuffer(g_img, dtype=np.uint8).reshape(g_res[1], g_res[0], 3)
    gui.set_image(ti.imresize(img, 512, 512))
    gui.show()
