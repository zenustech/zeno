import json
import asyncio
import websockets

import zenwebcfg
from zenutils import go

from . import streaming


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


async def ws_startup(url):
    async with websockets.connect(url) as ws:

        while True:
            data = json.dumps(status)
            await ws.send(data)

            data = await ws.recv()
            if data is None:
                break

            status.update(json.loads(data))


def ws_open(url):
    go(asyncio.get_event_loop().run_until_complete, ws_startup(url))


def uploadStatus():
    print('upload', status)


def initializeGL():
    ws_open(zenwebcfg.wsurl + '/webvisSocket')
    streaming.open(zenwebcfg.rtmpurl)


def paintGL():
    streaming.paint()
