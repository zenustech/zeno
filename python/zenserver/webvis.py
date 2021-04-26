import glfw
import json

import zenvis

from . import app, sockets


@sockets.route('/webvisSocket')
def webvisSocket(ws):
    print('zenvis init')

    succeed = glfw.init()
    assert succeed
    window = glfw.create_window(640, 480, "webvis context", None, None)
    assert window
    glfw.make_context_current(window)

    zenvis.initializeGL()

    while not ws.closed:
        data = ws.receive()
        if data is None:
            break

        zenvis.upStat.update(json.loads(data))
        width, height = zenvis.upStat['resolution']
        glfw.set_window_size(window, width, height)
        print('resize', width, height)

        print('zenvis paint')
        zenvis.uploadStatus()
        zenvis.paintGL()
        glfw.swap_buffers(window)
        glfw.poll_events()

        data = json.dumps(zenvis.dnStat)
        ws.send(data)

    print('zenvis exit')
    glfw.terminate()
