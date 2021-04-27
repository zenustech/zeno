import json
import asyncio
import websockets
import time

import zenwebcfg
from zenutils import go

from . import streaming


dnStat = {
    'frameid': 0,
    'solver_frameid': 0,
    'solver_interval': 0,
    'render_fps': 0,
}

upStat = {
    'next_frameid': -1,
    'resolution': (1, 1),
    'perspective': (),
    'playing': True,
}


async def ws_startup(url):
    async with websockets.connect(url) as ws:

        while True:
            data = json.dumps(upStat)
            await ws.send(data)

            data = await ws.recv()
            if data is None:
                break

            dnStat.update(json.loads(data))

            img = await ws.recv()
            if img is None:
                break
            streaming.decode(img)


def ws_open(url):
    go(asyncio.get_event_loop().run_until_complete, ws_startup(url))


def uploadStatus():
    pass


def initializeGL():
    ws_open(zenwebcfg.wsurl + '/webvisSocket')


def paintGL():
    streaming.paint()
