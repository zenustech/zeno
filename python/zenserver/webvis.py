import OpenGL.GL as gl
import glfw
import json
import queue

import zenvis
from zenutils import go

from . import app, sockets
from . import streaming


@sockets.route('/webvisSocket')
def webvisSocket(ws):
    qw = queue.Queue()
    qw.wsclosed = False
    t = go(workerWebvisSocket, qw)

    while not ws.closed:
        data = ws.receive()
        if data is None:
            qw.put(None)
            break

        zenvis.upStat.update(json.loads(data))

        res = lambda: 0
        qw.put(res)
        qw.join()

        img = streaming.encode(res.img, res.width, res.height, 2)

        data = json.dumps(zenvis.dnStat)
        ws.send(data)
        ws.send(img)

    qw.wsclosed = True
    t.join()


def workerWebvisSocket(qw):
    #print('zenvis init')
    succeed = glfw.init()
    assert succeed
    window = glfw.create_window(1, 1, 'webvis context', None, None)
    assert window
    #glfw.hide_window(window)
    glfw.make_context_current(window)

    zenvis.initializeGL()

    while not qw.wsclosed:
        res = qw.get()
        if res is None:
            break

        width, height = zenvis.upStat['resolution']
        glfw.set_window_size(window, width, height)

        #print('zenvis paint', width, height)
        zenvis.uploadStatus()
        zenvis.paintGL()
        glfw.swap_buffers(window)
        glfw.poll_events()

        res.width, res.height = width, height
        res.img = gl.glReadPixels(0, 0, width, height,
                gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

        qw.task_done()

    #print('zenvis exit')
    glfw.terminate()
