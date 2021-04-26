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
            qw.put(False)
            break

        zenvis.upStat.update(json.loads(data))

        qw.put(True)
        qw.join()

        data = json.dumps(zenvis.dnStat)
        ws.send(data)

    t.join()


def workerWebvisSocket(qw):
    print('zenvis init')
    succeed = glfw.init()
    assert succeed
    window = glfw.create_window(640, 480, 'webvis context', None, None)
    assert window
    glfw.make_context_current(window)

    zenvis.initializeGL()

    while qw.get():
        width, height = zenvis.upStat['resolution']
        glfw.set_window_size(window, width, height)

        print('zenvis paint', width, height)
        zenvis.uploadStatus()
        zenvis.paintGL()
        glfw.swap_buffers(window)
        glfw.poll_events()

        qw.task_done()

    print('zenvis exit')
    glfw.terminate()
