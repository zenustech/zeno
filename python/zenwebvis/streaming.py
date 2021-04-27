from PIL import Image
from io import BytesIO
from OpenGL.GL import *


g_img = None
g_res = (0, 0)
g_tex = None


def decode(img):
    global g_img, g_res
    with BytesIO(img) as f:
        im = Image.open(f)
        g_res = im.width, im.height
        g_img = im.tobytes()


def initializeGL():
    global g_tex

    glClearColor(0.0, 0.0, 0.0, 0.0)
    glEnable(GL_TEXTURE_2D)

    textures = [0]
    glGenTextures(1, textures)
    g_tex = textures[0]


def paintGL(nx, ny):
    if g_img is None:
        return

    glViewport(0, 0, nx, ny)

    glClear(GL_COLOR_BUFFER_BIT)
    glBindTexture(GL_TEXTURE_2D, g_tex)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, 3,
            g_res[0], g_res[1], 0, GL_RGB, GL_UNSIGNED_BYTE, g_img)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-1.0, -1.0, 0.0)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(1.0, -1.0, 0.0)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(1.0, 1.0, 0.0)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(-1.0, 1.0, 0.0)
    glEnd()
