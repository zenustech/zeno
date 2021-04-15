from . import core
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import time


class App:
    def __init__(self, nx=960, ny=800):
        self.nx, self.ny = nx, ny

    @property
    def curr_frameid(self):
        return core.get_curr_frameid()

    @curr_frameid.setter
    def curr_frameid(self, value):
        return core.set_curr_frameid(value)

    def draw(self):
        self.nx = glutGet(GLUT_WINDOW_WIDTH)
        self.ny = glutGet(GLUT_WINDOW_HEIGHT)
        core.set_window_size(self.nx, self.ny)

        core.new_frame()
        glFlush()
        time.sleep(1 / 60)

    def mainloop(self):
        glutInit()
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA)
        glutInitWindowSize(self.nx, self.ny)
        glutCreateWindow('zenvis')
        glutDisplayFunc(self.draw)
        glutIdleFunc(self.draw)
        core.initialize()
        glutMainLoop()
        core.finalize()

App().mainloop()
