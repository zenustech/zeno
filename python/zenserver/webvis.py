import OpenGL.GL as gl
from PIL import Image
from io import BytesIO
import numpy as np
import glfw
import json
import queue

import zenvis
from zenutils import go

from . import app, sockets


@sockets.route('/webvisSocket')
def webvisSocket(ws):
    qw = queue.Queue()
    t = go(workerWebvisSocket, qw)

    while not ws.closed:
        data = ws.receive()
        if data is None:
            qw.put(None)
            break

        zenvis.upStat.update(json.loads(data))

        res = type('', (), {})()
        qw.put(res)
        qw.join()

        print(res.jpeg)

        data = json.dumps(zenvis.dnStat)
        ws.send(data)

    t.join()


def workerWebvisSocket(qw):
    #print('zenvis init')
    succeed = glfw.init()
    assert succeed
    window = glfw.create_window(1, 1, 'webvis context', None, None)
    assert window
    glfw.make_context_current(window)
    #glfw.hide_window(window)

    zenvis.initializeGL()

    while True:
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

        img = gl.glReadPixels(0, 0, width, height,
                gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

        im = Image.new('RGB', (width, height))
        im.frombytes(img)
        with BytesIO() as f:
            im.save(f, 'jpeg')
            res.jpeg = f.getvalue()

        qw.task_done()

    #print('zenvis exit')
    glfw.terminate()
