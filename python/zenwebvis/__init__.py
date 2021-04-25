import json
import urllib.request
import urllib.parse

import zenwebcfg
from zenutils import go

from . import streaming, websocket


status = {
    'frameid': 0,
    'next_frameid': -1,
    'solver_frameid': 0,
    'solver_interval': 0,
    'render_fps': 0,
    'resolution': (1, 1),
    'perspective': (),
    'playing': True,
}


def uploadStatus():
    websocket.send(json.dumps(status))


def initializeGL():
    streaming.open(zenwebcfg.baseurl)
    websocket.open(zenwebcfg.baseurl)

    websocket.onreceive = _onRecvCallback


def paintGL():
    streaming.paint()


def _onRecvCallback(data):
    status.update(json.loads(data))
