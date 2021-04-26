from flask import Flask, request
from flask_sockets import Sockets
import json

import zenapi
import zenvis


app = Flask(__name__)
sockets = Sockets(app)


@app.route('/launchGraph', methods=['POST'])
def launchGraph():
    graph = json.loads(request.form['graph'])
    nframes = request.form['nframes']

    t = zenapi.launchGraph(graph, nframes)
    return 'OK'


@app.route('/getDescriptors', methods=['GET'])
def getDescriptors():
    descs = zenapi.getDescriptors()
    return json.dumps(descs)


@sockets.route('/webvisSocket')
def webvisSocket(ws):
    while not ws.closed:
        data = ws.receive()
        if data is None:
            break

        zenvis.status.update(json.loads(data))

        data = json.dumps(zenvis.status)
        ws.send(data)
