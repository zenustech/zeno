import OpenGL.GL as gl
import glfw


def key_callback(window, key, scancode, action, mods):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, 1)

succeed = glfw.init()
assert succeed
window = glfw.create_window(640, 480, "Hello World", None, None)
assert window
glfw.make_context_current(window)
glfw.set_key_callback(window, key_callback)
while not glfw.window_should_close(window):
    width, height = glfw.get_framebuffer_size(window)
    ratio = width / float(height)
    gl.glViewport(0, 0, width, height)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(-ratio, ratio, -1, 1, 1, -1)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    gl.glRotatef(glfw.get_time() * 50, 0, 0, 1)
    gl.glBegin(gl.GL_TRIANGLES)
    gl.glColor3f(1, 0, 0)
    gl.glVertex3f(-0.6, -0.4, 0)
    gl.glColor3f(0, 1, 0)
    gl.glVertex3f(0.6, -0.4, 0)
    gl.glColor3f(0, 0, 1)
    gl.glVertex3f(0, 0.6, 0)
    gl.glEnd()
    glfw.swap_buffers(window)
    glfw.poll_events()
glfw.terminate()
